import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import time
from datetime import datetime,timedelta
import os
import sys
import logging
import io
import traceback
import matplotlib.pyplot as plt
import mplfinance as mpf
import gymnasium as gym

from torchvision import transforms
from PIL import Image
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# カスタムモジュールのパス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datafetch_modeltrain.model import MultimodalFXmodel, SYMBOL_TO_ID
from trade.live_datafetch import get_live_features

# ディレクトリ設定
SAVE_MODEL_DIR = "../datafetch_modeltrain/saved_model"       
RL_MODEL_DIR = "../datafetch_modeltrain/saved_rl_models" 
SCALER_DIR = "../datafetch_modeltrain/saved_models"
LOG_DIR = "trading_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 資金管理・インターバル設定
INTERVAL_CONFIG = {"short": 5 * 60, "medium": 15 * 60, "long": 4 * 60 * 60}
MAGIC_NUMBER = 20260626
MIN_RR_RATIO = 1.5  # portfolio_env.py の学習時ロジックと一致させる最低リスクリワード比
# 💡 サーキットブレーカー: self.recent_losses[sym](銘柄別の直近連敗数、0.9倍/サイクルで減衰)が
# この値以上になったら、その銘柄への新規エントリーを一時停止する。3.0は概ね「短期間に3〜4回
# 連敗した状態」に相当し、実際のテストトレードで見られたGBPJPY等の高速な同方向再エントリー連敗
# (最大22連敗)を早期に遮断することを狙っている。値が下がれば(勝ちの有無に関わらず時間経過で
# 自然減衰する)自動的に再開する。
CIRCUIT_BREAKER_THRESHOLD = 3.0
MIN_RR_RATIO = 1.5  # portfolio_env.py の学習時ロジックと一致させる最低リスクリワード比
MAX_LOT_SIZE = 5.0  # portfolio_env.py の学習時ロジックと一致させるロット数の絶対上限

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DummyEnvForLoading(gym.Env):
    """💡 SB3のチェックをパスするためのポートフォリオ用ダミー環境"""
    def __init__(self, observation_shape, num_actions):
        super(DummyEnvForLoading, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        
    def reset(self, seed=None, options=None): 
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
    def step(self, action): 
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

class PortfolioFXTradingBot:
    def __init__(self, symbols, mode):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 観測・行動空間の次元
        # 💡 portfolio_env.py ver209.0 と合わせ、銘柄別の直近連敗数(1) を追加
        self.obs_shape = ((4 + 2 + 1) * self.num_symbols + 1,)
        self.num_actions = self.num_symbols * 3

        self.initial_balance = None
        self.vec_normalize = None

        # 💡 portfolio_env.py の self.recent_losses (銘柄別・直近連敗数) と同じ役割。
        # ライブでは学習時のようにその場でSL/TP判定できないため、
        # 「前回チェック時にあったポジションが今回無くなっている」ことを検知し、
        # MT5の約定履歴から損失決済だったかを確認して更新する(_update_recent_losses参照)。
        self.recent_losses = {sym: 0.0 for sym in self.symbols}
        self._last_known_position = {sym: 0.0 for sym in self.symbols}

        # スケーラーのロード
        self.scalers = {}
        scalerpath = os.path.join(SCALER_DIR, f"scaler_{mode}.pkl")
        if os.path.exists(scalerpath):
            loaded_scalers = joblib.load(scalerpath)
            for sym in self.symbols:
                if sym in loaded_scalers:
                    self.scalers[sym] = loaded_scalers[sym]
                else:
                    logger.warning(f"⚠️ {sym} のスケーラーが見つかりません。")
                    self.scalers[sym] = None
        
        self.analyst_model = self._load_analyst_model()
        self.rl_trader = self._load_rl_model()

    def _load_analyst_model(self):
        actual_features_count = 32
        for scaler in self.scalers.values():
            if scaler is not None:
                actual_features_count = len(scaler.feature_names_in_)
                break
                
        logger.info(f"✅ テーブルデータの特徴量数を取得: {actual_features_count}次元")
        model = MultimodalFXmodel(num_tabular_features=actual_features_count).to(self.device)
        
        model_path = os.path.join(SAVE_MODEL_DIR, f"final_best_model_{self.mode}.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(SAVE_MODEL_DIR, f"model_{self.mode}.pth")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
            logger.info(f"✅ アナリストモデルをロードしました: {model_path}")
            return model
        else:
            logger.error(f"❌ アナリストモデルが見つかりません: {model_path}")
            sys.exit()

    def _load_rl_model(self):
        rl_model_path = os.path.join(RL_MODEL_DIR, f"sac_portfolio_agent_{self.mode}.zip")
        vec_norm_path = os.path.join(RL_MODEL_DIR, f"vec_normalize.pkl") # train_portfolio_rl.pyの保存名に準拠
        
        if os.path.exists(rl_model_path):
            model = SAC.load(rl_model_path, device=self.device)
            logger.info(f"✨ 強化学習(ポートフォリオ)モデルをロードしました: {rl_model_path}")
        else:
            logger.error(f"❌ 強化学習モデルが見つかりません: {rl_model_path}")
            sys.exit()

        if os.path.exists(vec_norm_path):
            try:
                dummy_raw_env = DummyEnvForLoading(self.obs_shape, self.num_actions)
                dummy_venv = DummyVecEnv([lambda: dummy_raw_env])
                self.vec_normalize = VecNormalize.load(vec_norm_path, dummy_venv)
                self.vec_normalize.training = False 
                self.vec_normalize.norm_reward = False
                logger.info("📈 学習時の正規化統計情報を復元しました。")
            except Exception as e:
                logger.error(f"❌ VecNormalizeのロードエラー: {e}")
                self.vec_normalize = None

        return model

    def _get_current_position(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return 0.0
        
        total_lots = 0.0
        for p in positions:
            if p.magic == MAGIC_NUMBER:
                if p.type == mt5.POSITION_TYPE_BUY:
                    total_lots += p.volume
                elif p.type == mt5.POSITION_TYPE_SELL:
                    total_lots -= p.volume
        return total_lots

    def _get_unrealized_pnl_ratio(self, symbol, balance):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return 0.0
        total_profit = sum(p.profit for p in positions if p.magic == MAGIC_NUMBER)
        return total_profit / max(balance, 1e-8)

    def _update_recent_losses(self):
        """
        portfolio_env.py の self.recent_losses(銘柄別・直近連敗数)をライブ側でも再現する。
        学習時はステップ内でSL/TP判定と同時に連敗数を更新できるが、ライブではブローカーが
        自動でSL/TPを執行するため、「前回チェック時にあったポジションが今回無くなっている」
        ことをもって決済を検知し、MT5の約定履歴からその決済が損失だったかを確認する。
        呼び出しごと(=学習時の1ステップに相当)に、全銘柄を0.9倍で減衰させる。
        """
        now = datetime.now()
        interval = INTERVAL_CONFIG.get(self.mode, 300)
        lookback_start = now - timedelta(seconds=interval * 3 + 3600)

        for sym in self.symbols:
            current_pos = self._get_current_position(sym)
            prev_pos = self._last_known_position.get(sym, 0.0)

            if prev_pos != 0.0 and current_pos == 0.0:
                try:
                    deals = mt5.history_deals_get(lookback_start, now, group=sym)
                except Exception as e:
                    deals = None
                    logger.warning(f"⚠️ [{sym}] 約定履歴の取得に失敗しました: {e}")

                if deals:
                    closing_deals = [d for d in deals if d.magic == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_OUT]
                    if closing_deals:
                        latest_deal = max(closing_deals, key=lambda d: d.time)
                        if latest_deal.profit < 0:
                            self.recent_losses[sym] = self.recent_losses.get(sym, 0.0) + 1.0
                            logger.info(f"📉 [{sym}] 決済損失を検知。連敗カウント: {self.recent_losses[sym]:.2f}")

            self._last_known_position[sym] = current_pos

        for sym in self.symbols:
            self.recent_losses[sym] = self.recent_losses.get(sym, 0.0) * 0.9

    def _build_sac_observation(self):
        imgs_list = []
        tabs_list = []
        # 💡 B案: 銘柄埋め込み用のID。datafetch_modeltrain/model.py の SYMBOL_TO_ID を
        # 唯一の対応表として使い、学習側とライブ側でズレないようにする。
        symbol_id_list = [SYMBOL_TO_ID[sym] for sym in self.symbols]
        raw_dfs = {}
        
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for sym in self.symbols:
            live_data = get_live_features(sym, self.mode)
            if live_data is None:
                logger.warning(f"⚠️ {sym} のデータ取得失敗。観測生成をスキップします。")
                return None, None
                
            live_df, raw_df = live_data[0], live_data[1]
            raw_dfs[sym] = raw_df
            
            # テーブルデータ
            live_df_filled = live_df.ffill().bfill().fillna(1).tail(40)
            scaler = self.scalers.get(sym)
            try:
                scaled_data = scaler.transform(live_df_filled) if scaler else live_df_filled.values
            except:
                scaled_data = scaler.transform(live_df_filled.values) if scaler else live_df_filled.values
            tabs_list.append(torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0))

            # 画像データ生成（以前のボットのio.BytesIO方式を採用）
            try:
                buf = io.BytesIO()
                chart_df = raw_df.tail(60).copy() 
                
                def extract_tf_df(df, prefix):
                    tf_cols = [c for c in df.columns if c.startswith(prefix)]
                    sub_df = df[['time'] + tf_cols].copy()
                    rename_map = {}
                    for c in tf_cols:
                        c_lower = str(c).lower()
                        if c_lower.endswith('open'): rename_map[c] = 'open'
                        elif c_lower.endswith('high'): rename_map[c] = 'high'
                        elif c_lower.endswith('low'): rename_map[c] = 'low'
                        elif c_lower.endswith('close'): rename_map[c] = 'close'
                    sub_df.rename(columns=rename_map, inplace=True)
                    sub_df.set_index('time', inplace=True)
                    return sub_df

                if self.mode == "short":
                    df_base = extract_tf_df(chart_df, "M5_")
                    df_mid = extract_tf_df(chart_df, "M15_")
                    df_long = extract_tf_df(chart_df, "M30_")
                elif self.mode == "medium":
                    df_base = extract_tf_df(chart_df, "M15_")
                    df_mid = extract_tf_df(chart_df, "H1_")
                    df_long = extract_tf_df(chart_df, "H4_")
                else:
                    df_base = extract_tf_df(chart_df, "H4_")
                    df_mid = extract_tf_df(chart_df, "D1_")
                    df_long = extract_tf_df(chart_df, "W1_")

                custom_style = mpf.make_mpf_style(base_mpf_style="charles", gridstyle=" ", facecolor="white", figcolor="white")
                ap = [mpf.make_addplot(df_mid, type="candle", panel=1), mpf.make_addplot(df_long, type="candle", panel=2)]
                
                fig, axes = mpf.plot(
                    df_base, type="candle", style=custom_style, addplot=ap,
                    panel_ratios=(1, 1, 1), figsize=(9.33, 9.33), axisoff=True,
                    returnfig=True, tight_layout=True
                )

                fig.savefig(buf, format='png', bbox_inches="tight", pad_inches=0.1, dpi=72)
                plt.close(fig) 
                
                buf.seek(0)
                img_pil = Image.open(buf).convert("RGB")
                img_tensor = image_transform(img_pil).unsqueeze(0)
                
            except Exception as e:
                logger.error(f"画像生成エラー ({sym}): {e}")
                img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            
            imgs_list.append(img_tensor)

        imgs_tensor = torch.cat(imgs_list, dim=0).to(self.device)
        tabs_tensor = torch.cat(tabs_list, dim=0).to(self.device)
        symbol_id_tensor = torch.tensor(symbol_id_list, dtype=torch.long, device=self.device)

        with torch.no_grad():
            class_out, risk_out = self.analyst_model(imgs_tensor, tabs_tensor, symbol_id_tensor)
            probs = torch.softmax(class_out, dim=1).cpu().numpy()
            risk_vals = risk_out.cpu().numpy()

        account_info = mt5.account_info()
        if self.initial_balance is None: self.initial_balance = account_info.balance

        obs_list = [account_info.balance / self.initial_balance]
        for i, sym in enumerate(self.symbols):
            obs_list.extend(probs[i])
            obs_list.append(risk_vals[i][0])
            obs_list.append(float(self._get_current_position(sym)))
            obs_list.append(self._get_unrealized_pnl_ratio(sym, account_info.balance))
            # 💡 portfolio_env.py ver209.0 と同じ並び順で、銘柄別の直近連敗数を末尾に追加
            obs_list.append(float(self.recent_losses.get(sym, 0.0)))

        hybrid_obs_array = np.array([obs_list], dtype=np.float32)
        if self.vec_normalize is not None:
            normalized_obs = self.vec_normalize.normalize_obs(hybrid_obs_array)
            return normalized_obs, raw_dfs
        return hybrid_obs_array, raw_dfs

    def execute_trade_logic(self):

        # 💡 観測を作る前に、前回チェックからの決済(勝敗)を反映して連敗数を更新する
        self._update_recent_losses()

        logger.info("📡 ポートフォリオ全銘柄の相場データを取得・推論中...")
        obs, raw_dfs = self._build_sac_observation()
        if obs is None: return

        action, _ = self.rl_trader.predict(obs, deterministic=True)
        action = np.clip(np.array(action).flatten(), -1.0, 1.0)
        
        account = mt5.account_info()
        equity = account.equity if account else 1000.0

        for i, sym in enumerate(self.symbols):
            current_pos = self._get_current_position(sym)
            if current_pos != 0.0:
                logger.debug(f"[{sym}] 既存ポジション保持中。AIの新規判断をスキップし、TP/SLに任せます。")
                continue

            # 💡 サーキットブレーカー: 直近連敗数が閾値以上の銘柄は新規エントリーを見送る
            loss_streak = self.recent_losses.get(sym, 0.0)
            if loss_streak >= CIRCUIT_BREAKER_THRESHOLD:
                logger.warning(f"🛑 [{sym}] サーキットブレーカー作動中(直近連敗数: {loss_streak:.2f})。新規エントリーを見送ります。")
                continue

            act_direction = float(action[i*3])
            act_tp = float(action[i*3 + 1])
            act_sl = float(action[i*3 + 2])
            
            threshold = 0.2
            if abs(act_direction) > threshold:
                direction_sign = 1.0 if act_direction > 0 else -1.0
                
                base_tf = "M5" if self.mode == "short" else "M15" if self.mode == "medium" else "H4"
                tf = "M15" if self.mode == "short" else "H4" if self.mode == "medium" else "D1"
                current_close = raw_dfs[sym][f"{base_tf}_close"].iloc[-1]
                atr = raw_dfs[sym].get(f"{tf}_ATR", current_close * 0.005)
                if isinstance(atr, pd.Series): atr = atr.iloc[-1]

                # Envと完全に一致したTP/SL距離の計算
                sl_dist = atr * (1.0 + (act_sl + 1.0))
                tp_dist = atr * (1.0 + (act_tp + 1.0) * 2)

                # Envと同じ最低リスクリワード比(MIN_RR_RATIO倍)のクランプ
                min_tp_dist = sl_dist * MIN_RR_RATIO
                if tp_dist < min_tp_dist:
                    tp_dist = min_tp_dist

                # Envと完全に一致した動的ロット計算（リスク2%ベース）
                risk_amount = equity * 0.02 * abs(act_direction)
                
                symbol_info = mt5.symbol_info(sym)
                if symbol_info and symbol_info.trade_tick_size > 0:
                    loss_per_price_unit = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                    loss_for_this_trade = sl_dist * loss_per_price_unit
                    raw_lot = risk_amount / max(loss_for_this_trade, 1e-5)
                    raw_lot = min(raw_lot, MAX_LOT_SIZE) 
                    
                    step = symbol_info.volume_step
                    target_pos = round(raw_lot / step) * step
                    target_pos = max(symbol_info.volume_min, min(target_pos, symbol_info.volume_max))
                else:
                    target_pos = 0.01
                
                
                if direction_sign > 0:
                    self._send_order(sym, mt5.ORDER_TYPE_BUY, target_pos, tp_dist, sl_dist)
                else:
                    self._send_order(sym, mt5.ORDER_TYPE_SELL, target_pos, tp_dist, sl_dist)
    def _send_order(self, symbol, order_type, lots, tp_dist, sl_dist):
        if lots < 0.01: return
            
        
        account = mt5.account_info()
        if account is not None:
            equity = account.equity
            symbol_info_cap = mt5.symbol_info(symbol)

            # 💡 修正: 旧実装は「資産1000ドル/10万円あたり固定ロット」という、
            # SL距離(sl_dist)を一切考慮しない静的な上限だった。このため、GOLDのように
            # ATRに応じてSL距離が広がる場面で、実際のリスクが意図した2%を大きく
            # 超過していた(実測でリスク6.9%〜12.2%、最大約6倍)。
            # execute_trade_logic()の2%リスク計算と同じ sl_dist 連動のロジックで、
            # その2倍(4%)を「通常あり得ないはずの異常値」に対する最終防御ラインとする。
            MAX_RISK_PCT_HARD_CAP = 0.04
            if symbol_info_cap and symbol_info_cap.trade_tick_size > 0 and sl_dist > 0:
                loss_per_price_unit = symbol_info_cap.trade_tick_value / symbol_info_cap.trade_tick_size
                loss_for_this_trade_per_lot = sl_dist * loss_per_price_unit
                absolute_safety_cap = max(0.01, round((equity * MAX_RISK_PCT_HARD_CAP) / max(loss_for_this_trade_per_lot, 1e-5), 2))
            else:
                # symbol_info や sl_dist が取得できない場合の最終フォールバック(従来の静的上限)
                currency = getattr(account, 'currency', 'USD')
                base_unit = 1000.0 if currency == "USD" else 100000.0
                safety_multiplier = 0.02 if ("GOLD" in symbol or "XAU" in symbol) else 0.10
                absolute_safety_cap = max(0.01, round((equity / base_unit) * safety_multiplier, 2))

            if lots > absolute_safety_cap:
                logger.warning(f"⚠️ [{symbol}] 防御フィルター作動: AIの要求ロットを {absolute_safety_cap:.2f} Lot に制限。")
                lots = absolute_safety_cap

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        tp_price = price + tp_dist if order_type == mt5.ORDER_TYPE_BUY else price - tp_dist
        sl_price = price - sl_dist if order_type == mt5.ORDER_TYPE_BUY else price + sl_dist

        filling_type = self.get_filling_mode(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(float(lots), 2),
            "type": order_type,
            "price": price,
            "sl": sl_price, 
            "tp": tp_price, 
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Portfolio SAC",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ [{symbol}] 注文拒否。エラー: {result.retcode}")
        else:
            t_str = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            logger.info(f"✨ [{symbol}] 新規注文成功: {t_str} {lots:.2f} Lot")
    def get_filling_mode(self, symbol):
        """XM対応版：ブローカーがサポートしている注文実行モードを取得・補正"""
        symbol_info = mt5.symbol_info(symbol)
        
        # 情報が取得できない場合のデフォルト
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC
            
        filling_flags = symbol_info.filling_mode
        
        # 1. サーバーが明示的に許可しているモードを判定 (XMではIOCが通りやすい)
        if filling_flags & mt5.ORDER_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        elif filling_flags & mt5.ORDER_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
            
        # 2. フラグが0（設定なし）の場合の「XM向け」強力なフォールバック
        
        return mt5.ORDER_FILLING_IOC

def main():
    print("========================================")
    print(" 🌍 ポートフォリオ強化学習 自動売買システム")
    print("========================================")
    
    # 学習時と「完全一致」の順番
    TARGET_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]

    mode_input = input("実行モード（時間軸）を入力してください (short/medium/long) [デフォルト: short]: ").strip()
    mode = mode_input if mode_input else "short"

    print("\n--- 起動設定 ---")
    print(f"対象銘柄: {', '.join(TARGET_SYMBOLS)}")
    print(f"モード  : {mode}")
    print("----------------")
    
    if not mt5.initialize():
        logger.error("MT5の初期化に失敗しました。終了します。")
        sys.exit()

    # MT5の気配値に追加・待機処理（データ取得エラー防止）
    for sym in TARGET_SYMBOLS:
        if not mt5.symbol_select(sym, True):
            logger.error(f"❌ {sym} をMT5の気配値に追加できませんでした。MT5の銘柄名を確認してください。")
            sys.exit()

    logger.info("⏳ MT5サーバーから過去データを同期しています。5秒お待ちください...")
    time.sleep(5)
    for sym in TARGET_SYMBOLS:
        mt5.symbol_info_tick(sym)

    bot = PortfolioFXTradingBot(symbols=TARGET_SYMBOLS, mode=mode)
    interval = INTERVAL_CONFIG.get(mode, 300)

    logger.info(f"=== 稼働開始 (更新間隔: {interval}秒) ===")
    
    try:
        while True:
            if not mt5.terminal_info():
                if not mt5.initialize():
                    time.sleep(60)
                    continue

                # 1. まずトレードロジックを実行
            try:
                bot.execute_trade_logic()
            except Exception as e:
                logger.error(f"❌ トレードループエラー: {e}")
                import traceback
                traceback.print_exc()

            # 2. 次の「5分の倍数」の時刻を計算して、そこまで待機する
            now = datetime.now()
            # M5なら5分、M15なら15分単位で同期（modeに合わせてインターバルを分に変換）
            interval_minutes = interval // 60 
            
            # 次の実行時刻を算出 (例: 10:03 なら 10:05:00)
            minutes_to_add = interval_minutes - (now.minute % interval_minutes)
            next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            
            sleep_time = (next_run - now).total_seconds()
            logger.info(f"💤 次の確定足({next_run.strftime('%H:%M:00')})まで {sleep_time:.1f} 秒待機します...")
            
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("システムが停止されました。")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
