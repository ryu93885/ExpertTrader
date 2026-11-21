import copy
import logging
import os
import shutil
from collections import deque

import numpy as np
import torch
import math
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


# 💡 学習中に観測されたcritic_lossの急激な発散(80万ステップ付近で数百万台まで
# 悪化し、actor_loss・ent_coefも同時に悪化する現象)への対策。
# stable-baselines3(2.9.0時点)のSACには勾配クリッピングが実装されていないため、
# SAC.train()をオーバーライドし、critic/actorそれぞれの逆伝播直後に
# torch.nn.utils.clip_grad_norm_ を追加したサブクラス。
#
# ロジック自体は stable-baselines3==2.9.0 の SAC.train() を可能な限り忠実に
# 踏襲しており、勾配クリッピングの追加以外の変更は行っていない
# (SB3のバージョンが大きく変わった場合はこの部分の追従が必要)。
#
# 値の選定について: GRAD_CLIP_MAX_NORM はモデルごとに細かくチューニングする
# ような繊細なハイパーパラメータではなく、「通常の学習を妨げない範囲で、
# 異常な外れ値バッチによる大暴走だけを防ぐ」ための緩めの安全弁として機能する。
# このため厳密な最適値探索は不要で、10.0という一般的な値を採用している。
GRAD_CLIP_MAX_NORM = 10.0
ENT_COEF_MIN = 0.01

# 💡 追加(2026-11-21の調査で判明した問題への対応): 学習後半でcritic_lossが数桁
# 悪化し、その状態のまま学習が終了して実バックテストで致命的な損失を出す事例が
# 発生した(2026-11-20の学習、timestep約261万〜299万で発覚)。test_portfolio.py
# 相当の環境ベース評価をチェックポイント(50,000ステップ)ごとに挟むと学習時間への
# 影響が無視できないため、代わりに既に計算済みのcritic_loss自体を「ほぼ無料の
# 監視指標」として使う。
#
# このSACインスタンスはtrain_freq=1/gradient_steps=1(SB3のSACデフォルト、
# train_portfolio_rl.py側で上書きしていない)のため、train()は環境1ステップごとに
# 呼ばれる。そのため基準値(baseline)のウィンドウは「train()呼び出し回数」で
# 考えればよく、下記の値は環境ステップ数にほぼ等しい(数千ステップ=数十秒程度)。
CRITIC_LOSS_BASELINE_WINDOW = 2000  # 基準値(移動平均)に使う直近のtrain()呼び出し回数
MIN_BASELINE_SAMPLES = 500  # これより少ない間は基準値が未成熟とみなし、急上昇の判定を行わない
CRITIC_LOSS_SPIKE_MULTIPLIER = 10.0  # 基準値のこの倍率を超えたら「急上昇」とみなす
RECOVERY_COOLDOWN_CALLS = 2000  # 復元・学習率減衰の直後、基準値再構築を再開するまでの猶予
LR_DAMPENING_FACTOR = 0.1  # 急上昇を検知するたびに学習率へ掛け合わせる減衰係数
MIN_LR_DAMPENING = 0.01  # 減衰係数の下限(繰り返し検知しても学習率が実質ゼロにならないように)


class SACWithGradClip(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 💡 追加(2026-11-21): critic_loss監視・自動復元の状態。
        # safe_snapshot_path_base/drive_safe_snapshot_dirは呼び出し側
        # (train_portfolio_rl.py)でモード別のパスに上書きすることを想定した既定値。
        self._critic_loss_baseline = deque(maxlen=CRITIC_LOSS_BASELINE_WINDOW)
        self._safe_snapshot = None
        self._lr_dampening = 1.0
        self._cooldown_remaining = 0
        self.safe_snapshot_path_base = "saved_rl_models/sac_safe_snapshot"
        self.drive_safe_snapshot_dir = "/content/drive/MyDrive/saved_models_rl"

    def _snapshot_state(self):
        """現在のパラメータ・オプティマイザ状態をCPU上にコピーして保持する。"""
        snapshot = {
            "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
            "actor_optimizer": copy.deepcopy(self.actor.optimizer.state_dict()),
            "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
            "critic_optimizer": copy.deepcopy(self.critic.optimizer.state_dict()),
            "critic_target": {k: v.detach().cpu().clone() for k, v in self.critic_target.state_dict().items()},
        }
        if self.log_ent_coef is not None:
            snapshot["log_ent_coef"] = self.log_ent_coef.detach().cpu().clone()
            snapshot["ent_coef_optimizer"] = copy.deepcopy(self.ent_coef_optimizer.state_dict())
        return snapshot

    def _restore_state(self, snapshot):
        """_snapshot_stateで保存した状態を、現在学習中のモデルに復元する。"""
        device = self.device
        self.actor.load_state_dict({k: v.to(device) for k, v in snapshot["actor"].items()})
        self.actor.optimizer.load_state_dict(snapshot["actor_optimizer"])
        self.critic.load_state_dict({k: v.to(device) for k, v in snapshot["critic"].items()})
        self.critic.optimizer.load_state_dict(snapshot["critic_optimizer"])
        self.critic_target.load_state_dict({k: v.to(device) for k, v in snapshot["critic_target"].items()})
        if "log_ent_coef" in snapshot:
            with torch.no_grad():
                self.log_ent_coef.data.copy_(snapshot["log_ent_coef"].to(device))
            self.ent_coef_optimizer.load_state_dict(snapshot["ent_coef_optimizer"])

    def _save_safe_snapshot_to_disk(self):
        """
        復元直後(=現在の状態が安全な状態そのもの)にSB3標準形式で保存する。
        test_portfolio.pyの--model-pathでそのまま検証できるようにするため、
        self.save()(SB3標準のzip形式)を使う。
        """
        save_dir = os.path.dirname(self.safe_snapshot_path_base)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.save(self.safe_snapshot_path_base)
        vecnorm_path = f"{self.safe_snapshot_path_base}_vecnormalize.pkl"
        if self._vec_normalize_env is not None:
            self._vec_normalize_env.save(vecnorm_path)

        if os.path.exists("/content/drive/MyDrive"):
            try:
                os.makedirs(self.drive_safe_snapshot_dir, exist_ok=True)
                model_zip = f"{self.safe_snapshot_path_base}.zip"
                shutil.copy2(model_zip, os.path.join(self.drive_safe_snapshot_dir, os.path.basename(model_zip)))
                if os.path.exists(vecnorm_path):
                    shutil.copy2(
                        vecnorm_path,
                        os.path.join(self.drive_safe_snapshot_dir, os.path.basename(vecnorm_path)),
                    )
            except Exception as e:
                logging.warning(f"⚠️ 安全なスナップショットのGoogle Driveへのバックアップに失敗しました: {e}")

    def _handle_critic_loss_monitoring(self, mean_critic_loss):
        """
        critic_lossの急上昇を検知し、(B)直近の安定状態への自動復元 + 保存と、
        (C)学習率を下げての学習継続を行う。環境を使った評価を一切行わないため、
        追加の計算コストはほぼゼロ。
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return

        if len(self._critic_loss_baseline) >= MIN_BASELINE_SAMPLES:
            baseline = float(np.mean(self._critic_loss_baseline))
            if baseline > 0 and mean_critic_loss > baseline * CRITIC_LOSS_SPIKE_MULTIPLIER:
                logging.warning(
                    f"⚠️ critic_lossが直近の基準値の{CRITIC_LOSS_SPIKE_MULTIPLIER}倍を超えて"
                    f"急上昇しました(今回={mean_critic_loss:.6f} / 基準={baseline:.6f})。"
                    "学習が不安定化した兆候のため、直近の安定していた状態に復元し、"
                    "学習率を下げて継続します。"
                )
                if self._safe_snapshot is not None:
                    self._restore_state(self._safe_snapshot)
                    self._save_safe_snapshot_to_disk()
                else:
                    logging.warning("   復元可能な安定状態がまだ記録されていないため、復元はスキップします。")
                self._lr_dampening = max(MIN_LR_DAMPENING, self._lr_dampening * LR_DAMPENING_FACTOR)
                logging.warning(f"   学習率の減衰係数を{self._lr_dampening:.4f}に設定しました。")
                self._cooldown_remaining = RECOVERY_COOLDOWN_CALLS
                self._critic_loss_baseline.clear()
                return

        self._safe_snapshot = self._snapshot_state()
        self._critic_loss_baseline.append(mean_critic_loss)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)
        # 💡 追加(2026-11-21): critic_loss急上昇の検知後に設定される永続的な学習率
        # 減衰係数を、LinearSchedule本来の出力に重ねて適用する。_update_learning_rate
        # がスケジュール本来の値で上書きするため、その直後に掛け合わせる必要がある。
        if self._lr_dampening != 1.0:
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self._lr_dampening

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # 💡 追加: exp()適用後の値をENT_COEF_MINでクランプする。
                # ent_coef_loss自体は元のlog_ent_coef(クランプ前)で計算するため、
                # auto-tuningの勾配計算そのものは変更しない。
                ent_coef = torch.clamp(torch.exp(self.log_ent_coef.detach()), min=ENT_COEF_MIN)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                # 💡 追加: log_ent_coef(学習対象パラメータ)自体もENT_COEF_MINに
                # 対応する下限でクランプする。exp後の値だけをクランプすると
                # パラメータ自体は際限なく下がり続け、Adamのモーメンタムが
                # 偏った状態のままになるため、パラメータそのものを制限する。
                with torch.no_grad():
                    self.log_ent_coef.data.clamp_(min=math.log(ENT_COEF_MIN))

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # 💡 追加: 勾配クリッピング(critic)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_MAX_NORM)
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # 💡 追加: 勾配クリッピング(actor)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_MAX_NORM)
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        mean_critic_loss = float(np.mean(critic_losses))
        self._handle_critic_loss_monitoring(mean_critic_loss)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", mean_critic_loss)
        self.logger.record("train/lr_dampening", self._lr_dampening)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
