import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import logging

class FXTradingEnv(gym.Env):
    """
    MultimodalFXmodel (PyTorch) の予測結果と、厳選されたMTFインジケーターを融合した
    強化学習トレーディング環境（ハイブリッド型・報酬正規化・ランダムスタート実装版）
    ※TP/SLアクション学習、利確ボーナス、保有時間ペナルティ実装版
    ※コントラクトサイズ・為替レート（クロス円等）対応版
    """
    def __init__(self, dataset, trained_model, initial_balance=1000000, max_lot_size=1.0, transaction_cost=0.0001, price_col="M15_close", raw_features=None, raw_data_df=None, max_steps=1000, contract_size=100000, exchange_rate=1.0, risk_scaler=None):
        super(FXTradingEnv, self).__init__()
        logging.info("fx_env:ver204.1")
        self.dataset = dataset
        self.model = trained_model
        self.raw_data = raw_data_df
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # train.py が保存する risk_scalers_{mode}.json (symbol毎の mean/std) を受け取り、
        # モデルの生出力（Zスコア）を実際の target_risk_pct スケールへ逆変換するために使う。
        # evaluate.py / predict.py と同じ逆変換規約に揃える（未指定時は無変換=mean0/std1）。
        risk_scaler = risk_scaler or {}
        self.risk_mean = risk_scaler.get("mean", 0.0)
        self.risk_std = risk_scaler.get("std", 1.0)

        self.initial_balance = initial_balance
        self.max_lot_size = max_lot_size
        self.transaction_cost = transaction_cost 
        self.price_col = price_col 
        self.max_steps = max_steps 
        self.equity = self.initial_balance       # 現在の純資産
        self.prev_equity = self.initial_balance  # 1ステップ前の純資産（報酬計算用）
        
        # ロット計算とリスク管理用の設定
        self.base_max_lot = 1.0  # 初期資金の時の最大ロット（本番環境に合わせて調整）
        self.bankruptcy_threshold = self.initial_balance * 0.5  # 50%を失ったら破産

        
        self.contract_size = contract_size
        self.exchange_rate = exchange_rate

        self.consecutive_losses = 0.0
        self.last_evaluation_equity = initial_balance
        self.evaluation_interval = 100
        if raw_features is None:
            self.raw_features = ["M5_RSI", "M15_ema20_diff", "M30_ADX"]
        else:
            self.raw_features = raw_features

        # Action: [Lot Size, TP Ratio, SL Ratio]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_dim = 7 + len(self.raw_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset_env_variables()

    def reset_env_variables(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_position_size = 0.0 
        self.entry_price = 0.0
        self.current_episode_step = 0 
        self.holding_steps = 0 
        self.consecutive_losses = 0.0 
        self.last_evaluation_equity = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.trades_in_interval = 0
        self.current_risk_val = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env_variables()
        
        safe_max_start = len(self.dataset) - self.max_steps - 1
        if safe_max_start > self.dataset.seq_length:
            self.current_step = np.random.randint(self.dataset.seq_length, safe_max_start)
        else:
            self.current_step = self.dataset.seq_length
        
        obs = self._get_observation()
        while obs is None and self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            obs = self._get_observation()
            
        if obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        
            
        info = {"balance": self.balance, 
                "equity": self.equity,
                "position": self.current_position_size,
                }
        return obs, info

    def _get_observation(self):
        res = self.dataset[self.current_step]
        if res is None:
            return None
            
        image_tensor, tabular_tensor, _, _ = res
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        tabular_tensor = tabular_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            class_out, risk_out = self.model(image_tensor, tabular_tensor)
            probs = torch.softmax(class_out, dim=1).cpu().numpy()[0]
            risk_val_scaled = risk_out.cpu().numpy()[0][0]

        # train.py が学習時にZスコア化した target_risk_pct を、実際の%スケールへ逆変換する
        # （evaluate.py / predict.py と同じ risk_val * std + mean の規約）。
        risk_val = risk_val_scaled * self.risk_std + self.risk_mean

        self.current_risk_val = risk_val

        balance_ratio = self.balance / self.initial_balance
        current_price = self.raw_data.iloc[self.current_step][self.price_col]
        
        unrealized_pnl_ratio = 0.0
        if self.current_position_size != 0:
            # 💡 修正: 決済通貨での損益を計算し、口座通貨に換算
            unrealized_pnl_quote = (current_price - self.entry_price) * self.current_position_size * self.contract_size
            unrealized_pnl_account = unrealized_pnl_quote * self.exchange_rate
            unrealized_pnl_ratio = unrealized_pnl_account / self.balance

        position_signal = float(self.current_position_size)

        hybrid_obs = [
            probs[0], probs[1], probs[2],
            risk_val,
            balance_ratio,
            unrealized_pnl_ratio,
            position_signal
        ]

        for col in self.raw_features:
            val = self.raw_data.iloc[self.current_step][col]
            hybrid_obs.append(float(val))

        return np.array(hybrid_obs, dtype=np.float32)

    def step(self, action):
        current_price = self.raw_data.iloc[self.current_step][self.price_col]
        
        # ========================================================
        # 1. 資金管理：アナライザの予測(risk_val)に基づくロット計算
        # ========================================================
        # 現在の執行足に対応するATRを取得
        base_tf = self.price_col.split("_")[0]
        atr_col = f"{base_tf}_ATR"
        current_atr = self.raw_data.iloc[self.current_step][atr_col]

        # 💡 基本的な想定損失幅をアナライザの出力(risk_val)から計算
        # ※ risk_valがATRに対する乗数（ボラティリティ係数）として機能する想定
        base_sl_distance = current_atr * self.current_risk_val
        
        # 💡 絶対上限をATRの3倍に制限
        max_sl_distance = current_atr * 3.0
        sl_distance = min(base_sl_distance, max_sl_distance)
        
        # 0割り・過大ロット防止のための最小損失幅(例: 現在価格の0.05%程度)
        min_sl_distance = current_price * 0.0005 
        sl_distance = max(sl_distance, min_sl_distance)

        # 許容リスク額 (例: 現在の純資産の3%)
        risk_pct = 0.03
        allowable_loss_account = self.equity * risk_pct

        # 1ロットあたりの想定損失額を計算（為替レート・コントラクトサイズ考慮）
        loss_per_lot_quote = sl_distance * self.contract_size
        loss_per_lot_account = loss_per_lot_quote * self.exchange_rate

        # 損失額から逆算した最大許容ロット
        max_calculated_lot = allowable_loss_account / loss_per_lot_account

        # ========================================================
        # 2. AIの自信度(confidence)による目標ロット算出とシグナル判定
        # ========================================================
        confidence = abs(float(action[0]))
        current_sign = np.sign(self.current_position_size)
        action_sign = np.sign(float(action[0]))

        if self.current_position_size == 0:
            # 新規エントリーには強いシグナル（0.6以上）を要求
            if confidence < 0.6:
                raw_target_position = 0.0
            else:
                raw_target_position = float(action[0]) * max_calculated_lot
        else:
            # 既にポジションがある場合
            if action_sign == current_sign:
                if confidence < 0.3:
                    raw_target_position = 0.0
                else:
                    raw_target_position = float(action[0]) * max_calculated_lot
            else:
                if confidence > 0.6:
                    raw_target_position = float(action[0]) * max_calculated_lot
                else:
                    raw_target_position = 0.0

        # ========================================================
        # 3. 丸め処理（MT5仕様）と「最小ロット未満の取引見送り」
        # ========================================================
        step_lot = 0.01
        # 0.01単位に丸める
        target_position_size = round(raw_target_position / step_lot) * step_lot
        
        # 💡 追加ペナルティ用の変数初期化
        skip_penalty = 0.0

        # ロットが極端に小さくなった場合（0.01未満）は取引を行わない
        if abs(target_position_size) < 0.01:
            target_position_size = 0.0
            
            # 自信度は高かったのに「リスクが大きすぎて(SLが広すぎて)見送った」場合
            # AIに「相場環境が悪かった」ことを教えるための微小ペナルティ
            if confidence >= 0.6 and self.current_position_size == 0:
                skip_penalty = -0.05
        # ========================================================
        # 2 & 3. ATR連動型 TP/SL と リスクリワード(RR)強制連動
        # ========================================================
        #現在の執行足に対応するATRを取得
        base_tf = self.price_col.split("_")[0]
        atr_col = f"{base_tf}_ATR"
        current_atr = self.raw_data.iloc[self.current_step][atr_col]
        #TPの設定
        atr_multiplier = (float(action[1] + 1.0) / 2.0 * (3.0 - 0.5) + 0.5)
        tp_distance = current_atr * atr_multiplier
        tp_ratio = tp_distance / current_price

        #SLの設定   
        rr_multiplier = (float(action[2] + 1.0)/2.0 * (1.0 - 0.3) + 0.3)
        sl_ratio = tp_ratio * rr_multiplier


        tp_hit = False
        sl_hit = False

        if self.current_position_size > 0:
            if current_price >= self.entry_price * (1.0 + tp_ratio):
                tp_hit = True
            elif current_price <= self.entry_price * (1.0 - sl_ratio):
                sl_hit = True
        elif self.current_position_size < 0:
            if current_price <= self.entry_price * (1.0 - tp_ratio):
                tp_hit = True
            elif current_price >= self.entry_price * (1.0 + sl_ratio):
                sl_hit = True

        if tp_hit or sl_hit:
            target_position_size = 0.0
        else:
            target_position_size = raw_target_position

        position_change = target_position_size - self.current_position_size

        step_realized_pnl = 0.0

        if self.current_position_size != 0 and position_change != 0:
            if target_position_size == 0 or np.sign(target_position_size) != np.sign(self.current_position_size):
                realized_pnl_quote = (current_price - self.entry_price) * self.current_position_size * self.contract_size
                step_realized_pnl = realized_pnl_quote * self.exchange_rate
                self.balance += step_realized_pnl
            elif abs(target_position_size) < abs(self.current_position_size):
                closed_amount = self.current_position_size - target_position_size
                realized_pnl_quote = (current_price - self.entry_price) * closed_amount * self.contract_size
                step_realized_pnl = realized_pnl_quote * self.exchange_rate
                self.balance += step_realized_pnl
        
        if position_change != 0:
            # 💡 修正: 売買コストの通貨換算
            trade_cost_quote = abs(position_change) * self.transaction_cost * self.contract_size
            self.balance -= trade_cost_quote * self.exchange_rate
            
            if self.current_position_size == 0 or np.sign(target_position_size) != np.sign(self.current_position_size):
                self.entry_price = current_price
            elif abs(target_position_size) > abs(self.current_position_size):
                old_units = abs(self.current_position_size)
                new_units = abs(position_change)
                total_units = abs(target_position_size)
                self.entry_price = (self.entry_price * old_units + current_price * new_units) / total_units
                
            self.current_position_size = target_position_size

        self.current_step += 1
        self.current_episode_step += 1 
        
        next_obs = self._get_observation()
        while next_obs is None and self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            next_obs = self._get_observation()

        done = self.current_step >= len(self.dataset) - 1
        truncated = False

        next_price = self.raw_data.iloc[self.current_step][self.price_col] if not done else current_price
        unrealized_pnl_account = 0.0
        
        if self.current_position_size != 0:
            unrealized_pnl_quote = (next_price - self.entry_price) * self.current_position_size * self.contract_size
            unrealized_pnl_account = unrealized_pnl_quote * self.exchange_rate
            self.holding_steps += 1
        else:
            self.holding_steps = 0
        
        new_equity = self.balance + unrealized_pnl_account

        raw_reward = new_equity - self.equity
        reward = (raw_reward / self.initial_balance) * 100.0 
        self.equity = new_equity

        reward += skip_penalty
        if self.holding_steps > 0:
            reward -= self.holding_steps * 0.0005


        
        # 【追加】無駄撃ち・ドテン往復ビンタ防止ペナルティ
        # ロットサイズに比例させることで「むやみに大きなロットで方向を変える」ことを罰する
        if position_change != 0:
            reward -= abs(position_change) * 0.2
            self.trades_in_interval += 1

        if self.current_position_size == 0:
            decay_rate = 0.01  
            self.consecutive_losses = max(0.0, self.consecutive_losses - decay_rate)

        
        realized_pnl_pct = 0.0
        if step_realized_pnl != 0:
            realized_pnl_pct = (step_realized_pnl / self.initial_balance) * 100.0

        if tp_hit or step_realized_pnl > 0:
            self.consecutive_losses = 0.0  
            
            # 利益率に倍率（2.0倍）をかけて報酬化。利幅が大きいほどAIは喜ぶ。
            # 「遠くのTPに到達する」ことの旨味を教え込み、「利大」の行動を引き出す。
            reward += realized_pnl_pct * 2.0 
                
        elif sl_hit or step_realized_pnl < 0:
            self.consecutive_losses += 1.0
            
            # 損失率に倍率（3.0倍）をかけてペナルティ化。（※ realized_pnl_pct はマイナス値なのでそのまま足す）
            # SL幅を広くして巨大な損失を出した場合、AIにとって致命的なマイナス報酬（激痛）となるため、
            # 次第に「無理のない適切な幅のSL」を置くようになる。

            
            

            reward += realized_pnl_pct * 3.0 
            
            # 連敗メンタル崩壊ストッパー（これは生存のために維持）
            base_penalty = 0.5         
            penalty_multiplier = 1.5   
            consecutive_penalty = base_penalty * (penalty_multiplier ** (self.consecutive_losses - 1.0))
            consecutive_penalty = min(consecutive_penalty, 5.0)
            reward -= consecutive_penalty
        
        #報酬の計算
        safe_equity = max(1.0,self.equity)
        safe_prev_equity = max(1.0,self.prev_equity)

        log_return = np.log(safe_equity / safe_prev_equity)
        base_reward = log_return * 10.0

        # ========================================================
        # 2. 一定期間ごとの総合損益評価（クリッピングの「前」に行う）
        # ========================================================
        if self.current_episode_step % self.evaluation_interval == 0:
            period_profit = self.equity - self.last_evaluation_equity

            if self.trades_in_interval > 0:    
                if period_profit > 0:
                    reward += 2.0  
                elif period_profit < 0:
                    reward -= 2.0  
            else:
                reward -= 0.5
            self.last_evaluation_equity = self.equity
            self.trades_in_interval = 0

        # ========================================================
        # 3. 報酬の統合とクリッピング（保護）
        # ========================================================
        # 今まで加算した reward (ボーナスやペナルティ) と base_reward を合算
        total_unclipped_reward = base_reward + reward

        # AI保護のため、最終的な報酬を -1.0 〜 1.0 に制限
        reward = np.clip(total_unclipped_reward, -1.0, 1.0)

        # ========================================================
        # 4. 破産判定 (クリッピングの「後」に絶対的なペナルティとして適用)
        # ========================================================
        if self.equity <= self.bankruptcy_threshold:
            done = True
            reward = -1.0
        
        self.prev_equity = self.equity

        if self.current_episode_step >= self.max_steps:
            truncated = True

        if next_obs is None:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "position": self.current_position_size,
            "step": self.current_step,
            "bankruptcy":(self.equity <= self.bankruptcy_threshold)
        }

        return next_obs, reward, done, truncated, info