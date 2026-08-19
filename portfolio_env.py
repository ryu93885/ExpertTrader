import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

NEW_HIGH_BONUS_COEF = 50.0
DRAWDOWN_PENALTY_COEF = 40.0
REALIZED_PNL_WEIGHT = 1.0
UNREALIZED_PNL_WEIGHT = 0.2
MIN_RR_RATIO = 1.5
CONSECUTIVE_LOSS_PENALTY_COEF = 0.5
COUNTERFACTUAL_PENALTY_COEF = 1.0


class PortfolioFXEnv(gym.Env):
    """
    7銘柄同時並行トレード環境（Action Space: 21次元）
    1つの口座資金を共有し、相関関係を利用したヘッジトレードを学習する
    """
    def __init__(self, dataset,symbols, initial_balance=10000000):
        super(PortfolioFXEnv, self).__init__()
        logging.info("PortfolioFXEnv: 21-Action Multi-Asset Initialization")
        logging.info("portfolio_env.py ver208.2")
        
        self.dataset = dataset
        
        self.symbols = symbols
        self.num_symbols = len(symbols)
        
        self.initial_balance = initial_balance
        
        # 銘柄ごとの設定値（コントラクトサイズやポイントバリュー）
        # ※実運用時は各銘柄の正しい値を辞書で渡す設計に拡張してください
        self.symbol_configs = {
            sym: {"point_value": 0.001, "contract_size": 100000} if "USDJPY" in sym or "EUR" in sym else {"point_value": 0.01, "contract_size": 100} # GOLD等
            for sym in self.symbols
        }

        # アクション空間: 3アクション(方向, TP, SL) × 7銘柄 = 21次元
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_symbols * 3,), dtype=np.float32)
        
        # 観察空間の次元計算（適宜調整）:
        # モデル出力(4) + ポジション状態(2) = 6 × 7銘柄 = 42次元 + 口座残高比率(1) = 43次元
        obs_dim = (4 + 2) * self.num_symbols + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset_env_variables()

    def reset_env_variables(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.high_water_mark = self.initial_balance
        self.prev_unrealized_pnl = 0.0

        # 各銘柄のポジション状態を辞書で管理
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.entry_prices = {sym: 0.0 for sym in self.symbols}
        self.tp_prices = {sym:0.0 for sym in self.symbols}
        self.sl_prices = {sym:0.0 for sym in self.symbols}
        self.current_step = self.dataset.seq_length
        self.recent_losses = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env_variables()
        
        # ランダムな開始位置の決定
        safe_max_start = len(self.dataset) - 1000
        self.current_step = np.random.randint(self.dataset.seq_length, max(safe_max_start, self.dataset.seq_length + 1))
        
        obs = self._get_observation()
        while obs is None and self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            obs = self._get_observation()
            
        return obs, {"equity": self.equity}

    def _get_observation(self):
        # 1. 範囲外アクセスの防止
        if self.current_step >= len(self.dataset.df):
            return None

        # 2. 現在のステップの行データ（事前計算済みの予測結果を含む）を取得
        row = self.dataset.df.iloc[self.current_step]
        
        obs_list = []
        
        # 3. 口座残高の比率
        balance_ratio = self.balance / self.initial_balance
        obs_list.append(balance_ratio)
        
        # USDJPYレートの安全な取得（損益の円換算用）
        base_tf = "M5" if self.dataset.mode == "short" else "M15" if self.dataset.mode == "medium" else "H4"
        usdjpy_col = f"USDJPY_{base_tf}_close_raw"
        current_usdjpy_rate = row.get(usdjpy_col, 150.0)
        
        # 4. 各銘柄の情報を追加
        for sym in self.symbols:
            # 🚀 事前計算されたAIの予測結果をDataFrameから「直接」読み込む（推論をバイパス）
            prob_0 = row.get(f"{sym}_prob_0", 0.0)
            prob_1 = row.get(f"{sym}_prob_1", 0.0)
            prob_2 = row.get(f"{sym}_prob_2", 0.0)
            risk_val = row.get(f"{sym}_risk_val", 0.0)
            
            obs_list.extend([prob_0, prob_1, prob_2])
            obs_list.append(risk_val)
            
            # 現在のポジションサイズ
            pos_size = float(self.positions[sym])
            obs_list.append(pos_size)
            
            # 含み損益の計算と追加
            unrealized_pnl_ratio = 0.0
            if pos_size != 0:
                current_price = row.get(f"{sym}_{base_tf}_close_row", self.entry_prices[sym])
                raw_pnl = (current_price - self.entry_prices[sym]) * pos_size * self.symbol_configs[sym]["contract_size"]
                
                # 円換算
                if "JPY" in sym:
                    converted_pnl = raw_pnl
                elif "USD" in sym or "GOLD" in sym: 
                    converted_pnl = raw_pnl * current_usdjpy_rate
                else:
                    converted_pnl = raw_pnl
                    
                unrealized_pnl_ratio = converted_pnl / max(self.balance, 1e-8)
                
            obs_list.append(unrealized_pnl_ratio)
            
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        step_realized_pnl = 0.0
        unrealized_pnl_total = 0.0
        counterfactual_penalty = 0.0
        row = self.dataset.df.iloc[self.current_step]
        base_tf = "M5" if self.dataset.mode == "short" else "M15" if self.dataset.mode == "medium" else "H4"
        tf = "M15" if self.dataset.mode == "short" else "H4" if self.dataset.mode == "medium" else "D1"
        usdjpy_col = f"USDJPY_{base_tf}_close"
        if usdjpy_col in row.index:
            current_usdjpy_rate = row[usdjpy_col]
        else:
            current_usdjpy_rate = 150.0
        
        for i, sym in enumerate(self.symbols):
            act_direction = action[i*3]
            act_tp = action[i*3 + 1]
            act_sl = action[i*3 + 2]
            
            
            
            current_close = row[f"{sym}_{base_tf}_close_raw"]
            current_high = row[f"{sym}_{base_tf}_high_raw"]
            current_low = row[f"{sym}_{base_tf}_low_raw"]
            spread = row.get(f"{sym}_{base_tf}_spread_raw", 0.0002)
            atr = row.get(f"{sym}_{tf}_ATR_raw", current_close * 0.005)
            config = self.symbol_configs[sym]
            
            # 1. 既存ポジションのTP/SL判定
            if self.positions[sym] != 0:
                is_closed = False
                pos = self.positions[sym]
                entry_p = self.entry_prices[sym]
                raw_pnl = 0.0
                
                is_sl_closed = False
                if pos > 0:
                    if current_high >= self.tp_prices[sym]:
                        raw_pnl = (self.tp_prices[sym] - entry_p) * pos * config["contract_size"]
                        is_closed = True
                    elif current_low <= self.sl_prices[sym]:
                        raw_pnl = (self.sl_prices[sym] - entry_p) * pos * config["contract_size"]
                        is_closed = True
                        is_sl_closed = True
                elif pos < 0:
                    if current_low <= self.tp_prices[sym]:
                        raw_pnl = (entry_p - self.tp_prices[sym]) * abs(pos) * config["contract_size"]
                        is_closed = True
                    elif current_high >= self.sl_prices[sym]:
                        raw_pnl = (entry_p - self.sl_prices[sym])*abs(pos) * config["contract_size"]
                        is_closed = True
                        is_sl_closed = True

                if is_sl_closed:
                    self.recent_losses += 1.0
                    future_start = self.current_step+1
                    future_end = min(len(self.dataset.df),self.current_step + 21)
                    if future_start<future_end:
                        future_rows = self.dataset.df.iloc[future_start:future_end]
                        high_col = f"{sym}_{base_tf}_high_raw"
                        low_col = f"{sym}_{base_tf}_low_raw"
                        if pos > 0 and high_col in future_rows.columns:
                            max_future_high = future_rows[high_col].max()
                            if max_future_high >= entry_p +(self.tp_prices[sym] - entry_p) * 0.5:
                                counterfactual_penalty += COUNTERFACTUAL_PENALTY_COEF
                        elif pos < 0 and low_col in future_rows.columns:
                            min_future_low = future_rows[low_col].min()
                            if min_future_low <= entry_p -(entry_p - self.tp_prices[sym]) * 0.5:
                                counterfactual_penalty += COUNTERFACTUAL_PENALTY_COEF
                if is_closed:
                    if "JPY" in sym:
                        converted_pnl = raw_pnl
                    elif sym in ["EURUSD", "GBPUSD", "AUDUSD", "GOLD"]: 
                        converted_pnl = raw_pnl * current_usdjpy_rate
                    elif sym == "USDCAD":
                        current_usdcad_rate = row.get(f"USDCAD_{tf}_close", 1.35) # 該当時間軸の終値
                        converted_pnl = raw_pnl * (current_usdjpy_rate / current_usdcad_rate)
                        
                    step_realized_pnl += converted_pnl
                    self.positions[sym] = 0.0
            
            # 2. 新規エントリー判定
            threshold = 0.2
            if self.positions[sym] == 0 and abs(act_direction) > threshold:
                direction_sign = 1.0 if act_direction > 0 else -1.0
                risk_amount = max(self.equity, 0) * 0.02 * abs(act_direction)
                
                sl_multiplier = 1.0 + (act_sl + 1.0)
                sl_dist = atr * sl_multiplier
                
                if sl_dist > 0:
                    raw_lot = risk_amount / (sl_dist * config["contract_size"])
                    target_lot = round(max(0.01, min(raw_lot, 5.0)), 2)
                else:
                    target_lot = 0.01
                
                raw_spread_cost = target_lot * spread * config["contract_size"]
                if "JPY" in sym:
                    spread_cost = raw_spread_cost
                elif "USD" in sym or "GOLD" in sym:
                    spread_cost = raw_spread_cost * current_usdjpy_rate
                else:
                    spread_cost = raw_spread_cost
                    
                step_realized_pnl -= spread_cost
                self.positions[sym] = target_lot * direction_sign
                self.entry_prices[sym] = current_close
                
                tp_multiplier = 1.0 + (act_tp + 1.0) * 2
                tp_dist = atr * tp_multiplier

                min_tp_dist = sl_dist * MIN_RR_RATIO
                if tp_dist < min_tp_dist:
                    tp_dist = min_tp_dist
                
                if direction_sign > 0:
                    self.tp_prices[sym] = current_close + tp_dist
                    self.sl_prices[sym] = current_close - sl_dist
                else:
                    self.tp_prices[sym] = current_close - tp_dist
                    self.sl_prices[sym] = current_close + sl_dist

            # 3. 含み損益の更新
            if self.positions[sym] != 0:
                raw_unrealized = (current_close - self.entry_prices[sym]) * self.positions[sym] * config["contract_size"]
                if "JPY" in sym:
                    unrealized_pnl_total += raw_unrealized
                elif "USD" in sym or "GOLD" in sym:
                    unrealized_pnl_total += raw_unrealized * current_usdjpy_rate
                else:
                    unrealized_pnl_total += raw_unrealized

        self.balance += step_realized_pnl
        self.equity = self.balance + unrealized_pnl_total

        delta_unrealized = unrealized_pnl_total - self.prev_unrealized_pnl
        realized_reward = (step_realized_pnl / self.initial_balance) * 100.0 * REALIZED_PNL_WEIGHT
        unrealized_reward = (delta_unrealized / self.initial_balance) * 100.0 * UNREALIZED_PNL_WEIGHT
        base_reward = realized_reward + unrealized_reward

        self.recent_losses *= 0.9
        consecutive_loss_penalty = self.recent_losses * CONSECUTIVE_LOSS_PENALTY_COEF
        reward = base_reward - counterfactual_penalty - consecutive_loss_penalty

        self.prev_unrealized_pnl = unrealized_pnl_total

        
        if self.equity > self.high_water_mark:
            reward += (self.equity - self.high_water_mark)/self.initial_balance * 100 * NEW_HIGH_BONUS_COEF

        self.prev_equity = self.equity
        self.current_step += 1
        done = self.current_step >= len(self.dataset) - 1
        
        if self.equity <= self.initial_balance * 0.5:
            done = True
            reward -= 20.0

        drawdown_ratio = max(0.0,(self.high_water_mark - self.equity) / self.high_water_mark)
        reward -= (drawdown_ratio **2) * DRAWDOWN_PENALTY_COEF

        next_obs = self._get_observation()
        while next_obs is None and self.current_step < len(self.dataset) - 1:
            self.current_step += 1
            next_obs = self._get_observation()

        if next_obs is None:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {"equity": self.equity, "balance": self.balance}
        return next_obs, reward, done, False, info