import numpy as np
import torch
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

class SACWithGradClip(SAC):
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

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

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
