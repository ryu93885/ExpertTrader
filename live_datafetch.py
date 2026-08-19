import MetaTrader5 as mt5
import pandas as pd
import numpy as np

MT5_TIMEFRAMES = {
    "M5":mt5.TIMEFRAME_M5,
    "M15":mt5.TIMEFRAME_M15,
    "M30":mt5.TIMEFRAME_M30,
    "H1":mt5.TIMEFRAME_H1,
    "H4":mt5.TIMEFRAME_H4,
    "D1":mt5.TIMEFRAME_D1,
    "W1":mt5.TIMEFRAME_W1
}


def get_mtf_config(mode):
    if mode == "short":
        return ["M5","M15","M30"]
    elif mode == "medium":
        return ["M15","H1","H4"]
    elif mode == "long":
        return ["H4","D1","W1"]
    return ["M15","H1","H4"]


def fetch_recent_data(symbol,timeframe_str,num_candles = 100):
    tf = MT5_TIMEFRAMES.get(timeframe_str)
    if tf is None:
        raise ValueError(f"Invalid timeframe:{timeframe_str}")
    
    rates = mt5.copy_rates_from_pos(symbol,tf,0,num_candles)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch data for {symbol}{timeframe_str}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[["time","open","high","low","close","tick_volume","spread"]]
    rename_dict = {col: f"{timeframe_str}_{col}" for col in df.columns if col != 'time'}
    df = df.rename(columns = rename_dict)

    return df


def calculate_features(df, tf_str):
    close_col = f"{tf_str}_close"
    high_col = f"{tf_str}_high"
    low_col = f"{tf_str}_low"
    open_col = f"{tf_str}_open"

    # 1. Return
    df[f"{tf_str}_return"] = df[close_col].pct_change()

    # 2. Volatility (preprocessing.py とロジックを完全統一)
    df[f"{tf_str}_volatility"] = (df[high_col] - df[low_col]) / df[open_col]
    
    # 3. MACD
    ema12 = df[close_col].ewm(span=12, adjust=False).mean()
    ema26 = df[close_col].ewm(span=26, adjust=False).mean()
    df[f"{tf_str}_macd"] = ema12 - ema26
    df[f"{tf_str}_macd_signal"] = df[f"{tf_str}_macd"].ewm(span=9, adjust=False).mean()
    df[f"{tf_str}_macd_hist"] = df[f"{tf_str}_macd"] - df[f"{tf_str}_macd_signal"]

    # 4. Bollinger Bands (20, 2)
    rolling_mean = df[close_col].rolling(window=20).mean()
    rolling_std = df[close_col].rolling(window=20).std()
    df[f"{tf_str}_bb_upper"] = rolling_mean + (rolling_std * 2)
    df[f"{tf_str}_bb_lower"] = rolling_mean - (rolling_std * 2)
    df[f"{tf_str}_bb_width"] = (df[f"{tf_str}_bb_upper"] - df[f"{tf_str}_bb_lower"]) / rolling_mean

    # 5. EMA (20, 50, 200) と乖離率
    for period in [20, 50, 200]:
        ema = df[close_col].ewm(span=period, adjust=False).mean()
        df[f"{tf_str}_ema{period}"] = ema
        df[f"{tf_str}_ema{period}_diff"] = (df[close_col] - ema) / ema * 100

    # 6. その他の性質特徴量
    df[f"{tf_str}_high_low_spread"] = (df[high_col] - df[low_col]) / df[close_col] * 100
    df[f"{tf_str}_close_open_spread"] = (df[close_col] - df[open_col]) / df[close_col] * 100
    df[f"{tf_str}_volatility_rolling"] = df[close_col].rolling(window=20).std() / df[close_col] * 100
    return df
def add_technical_indicators(df, tf_str):
    """すべての時間軸に対して個別にRSI, ATR, ADXを計算し、時間軸名のプレフィックスを付与する"""
    close_col = f"{tf_str}_close"
    high_col = f"{tf_str}_high"
    low_col = f"{tf_str}_low"
    open_col = f"{tf_str}_open"
    # --- RSI ---
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df[f"{tf_str}_RSI"] = 100 - (100 / (1 + rs))

    # --- ATR ---
    high_low = df[high_col] - df[low_col]
    high_close = (df[high_col] - df[close_col].shift()).abs()
    low_close = (df[low_col] - df[close_col].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"{tf_str}_ATR"] = tr.rolling(window=14).mean()

    # --- ADX ---
    up_move = df[high_col].diff()
    down_move = df[low_col].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = df[f"{tf_str}_ATR"].replace(0, 1e-10)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10))
    df[f"{tf_str}_ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    df[f"{tf_str}_norm_atr_pct"] = (df[f"{tf_str}_ATR"] / df[close_col]) * 100

    # 2. ノイズ率 (Wick-to-Body Ratio)
    body = abs(df[close_col] - df[open_col])
    wick_total = (df[high_col] - df[low_col]) - body
    df[f"{tf_str}_noise_ratio"] = wick_total / (body + 1e-8) # 1e-8はゼロ除算防止

    # 3. 短期的な値幅の標準偏差
    df[f"{tf_str}_volatility_std"] = df[close_col].rolling(window=20).std() / df[close_col] * 100
    return df


def get_live_features(symbol, mode, seq_length=40):
    if not mt5.terminal_info():
        mt5.initialize()

    tfs = get_mtf_config(mode)
    
    # 内部ヘルパー関数: 各時間軸のデータを取得してインジケーターを計算
    def get_tf_data(tf_str):
        # 15本分取得するために、余裕を持って過去150本を取得
        df = fetch_recent_data(symbol, tf_str, num_candles=300)
        df = calculate_features(df, tf_str)
        df = add_technical_indicators(df, tf_str)
        return df.dropna().sort_values("time")

    # 1. 各時間軸のデータを個別に計算
    df_base = get_tf_data(tfs[0])
    df_mid  = get_tf_data(tfs[1])
    df_long = get_tf_data(tfs[2])

    # 2. ベース時間軸(例: M5)に合わせて、上位足の直近データを時間同期 (merge_asof)
    df_merged = pd.merge_asof(df_base, df_mid, on="time", direction="backward")
    df_merged = pd.merge_asof(df_merged, df_long, on="time", direction="backward")

    # 3. 欠損値の処理
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)
    df_merged.fillna(1, inplace=True)

    # 4. モデルが要求するシーケンス長（15本）だけを最新から切り取る
    if len(df_merged) < seq_length:
        raise ValueError(f"データが不足しています。最低 {seq_length} 本必要ですが、{len(df_merged)} 本しかありません。")
        
    live_df = df_merged.tail(seq_length).copy().reset_index(drop=True)

    
    live_df['sin_time'] = np.sin(2 * np.pi * live_df['time'].dt.hour / 24)
    live_df['cos_time'] = np.cos(2 * np.pi * live_df['time'].dt.hour / 24)

    
    live_df = live_df.drop(columns=['time'])

    return live_df,df_merged

#test
if __name__ == "__main__":
    mt5.initialize()
    print("Fetching live features for USDJPY")
    try:
        features = get_live_features("USDJPY","medium")
        print(features.T)
    except Exception as e:
        print(f"Error:{e}")

    mt5.shutdown()

