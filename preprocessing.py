import glob
import os
import logging
import pandas as pd
import argparse
import numpy as np

TRADING_MODE = {
    "short":["M5","M15","M30"],
    "medium":["M15","H1","H4"],
    "long":["H4","D1","W1"]
}

DEFAULT_SYMGOLS = ["USDJPY","EURUSD","GBPUSD","EURJPY","GBPJPY","AUDUSD","XAUUSD"]

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir,exist_ok=True)
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s [%(levelname)s] %(message)s')


def load_csv_data(symbol,tf_name):
    #load a saved CSV file
    filepath = f"data/{symbol}_{tf_name}.csv"
    if not os.path.exists(filepath):return pd.DataFrame()
    df = pd.read_csv(filepath)
    
    df["time"] = pd.to_datetime(df["time"])


    df = df[["time","open","high","low","close","tick_volume","spread"]].copy()
    df.rename(columns = {col:f"{tf_name}_{col}" for col in df.columns if col != "time"},inplace = True)

    return df


def calculate_returns(df,tf_name):
    """価格データを変化率やテクニカル指標に変換し、live_datafetch.pyと完全同期する"""
    close_col = f"{tf_name}_close"
    high_col = f"{tf_name}_high"
    low_col = f"{tf_name}_low"
    open_col = f"{tf_name}_open"
    
    if close_col not in df.columns:
        return df

    # 1. Return
    df[f"{tf_name}_return"] = df[close_col].pct_change()

    # 2. Volatility
    df[f"{tf_name}_volatility"] = (df[high_col] - df[low_col]) / df[open_col]
    
    # 3. MACD
    ema12 = df[close_col].ewm(span=12, adjust=False).mean()
    ema26 = df[close_col].ewm(span=26, adjust=False).mean()
    df[f"{tf_name}_macd"] = ema12 - ema26
    df[f"{tf_name}_macd_signal"] = df[f"{tf_name}_macd"].ewm(span=9, adjust=False).mean()
    df[f"{tf_name}_macd_hist"] = df[f"{tf_name}_macd"] - df[f"{tf_name}_macd_signal"]

    # 4. Bollinger Bands (20, 2)
    rolling_mean = df[close_col].rolling(window=20).mean()
    rolling_std = df[close_col].rolling(window=20).std()
    df[f"{tf_name}_bb_upper"] = rolling_mean + (rolling_std * 2)
    df[f"{tf_name}_bb_lower"] = rolling_mean - (rolling_std * 2)
    df[f"{tf_name}_bb_width"] = (df[f"{tf_name}_bb_upper"] - df[f"{tf_name}_bb_lower"]) / rolling_mean

    # 5. EMA (20, 50, 200) と乖離率
    for period in [20, 50, 200]:
        ema = df[close_col].ewm(span=period, adjust=False).mean()
        df[f"{tf_name}_ema{period}"] = ema
        df[f"{tf_name}_ema{period}_diff"] = (df[close_col] - ema) / ema * 100

    # 6. その他の性質特徴量
    df[f"{tf_name}_high_low_spread"] = (df[high_col] - df[low_col]) / df[close_col] * 100
    df[f"{tf_name}_close_open_spread"] = (df[close_col] - df[open_col]) / df[close_col] * 100
    df[f"{tf_name}_volatility_rolling"] = df[close_col].rolling(window=20).std() / df[close_col] * 100

    return df

def add_time_features(df):
    """共通の時間周期性特徴量を1回だけデータの末尾に追加する"""
    df['sin_time'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['cos_time'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    return df

def process_mtf_data(symbol,mode):
    #Combine the three time frames and perform preprocessing
    tfs = TRADING_MODE.get(mode)
    
    if not tfs:
        return
    
    base_tf,mid_tf,long_tf = tfs[0],tfs[1],tfs[2]


    df_base = load_csv_data(symbol,base_tf)
    df_mid = load_csv_data(symbol,mid_tf)
    df_long = load_csv_data(symbol,long_tf)

    if df_base.empty or df_mid.empty or df_long.empty:
        logging.error(f"{symbol} - Missing data for one or more timeframes. Skipping.")
        return
    df_base = calculate_returns(df_base, base_tf)
    df_base = add_technical_indicators(df_base, base_tf)

    df_mid = calculate_returns(df_mid, mid_tf)
    df_mid = add_technical_indicators(df_mid, mid_tf)

    df_long = calculate_returns(df_long, long_tf)
    df_long = add_technical_indicators(df_long, long_tf)

    # Integration to Prevent Future Information Leaks
    df_base = df_base.sort_values("time")
    df_mid = df_mid.sort_values("time")
    df_long = df_long.sort_values("time")

    df_merged = pd.merge_asof(df_base, df_mid, on="time", direction="backward")
    df_merged = pd.merge_asof(df_merged, df_long, on="time", direction="backward")

    # filling in missing value
    df_merged.ffill(inplace=True)
    df_merged = add_time_features(df_merged)
    #delete NaN bypct_change
    df_merged.dropna(inplace = True)

    #save
    out_dir = "processed_data"
    os.makedirs(out_dir,exist_ok=True)
    out_file = f"{out_dir}/{symbol}_{mode}_processed.csv"
    df_merged.to_csv(out_file,index = False)
    logging.info(f"Preprocessing complete · Saved:{out_file}(number of data:{len(df_merged)})")

def add_technical_indicators(df, tf_name):
    """すべての時間軸に対して個別にRSI, ATR, ADXを計算し、時間軸名のプレフィックスを付与する"""
    close_col = f"{tf_name}_close"
    high_col = f"{tf_name}_high"
    low_col = f"{tf_name}_low"
    open_col = f"{tf_name}_open"
    
    if close_col not in df.columns:
        return df

    # --- RSI ---
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df[f"{tf_name}_RSI"] = 100 - (100 / (1 + rs))

    # --- ATR ---
    high_low = df[high_col] - df[low_col]
    high_close = (df[high_col] - df[close_col].shift()).abs()
    low_close = (df[low_col] - df[close_col].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"{tf_name}_ATR"] = tr.rolling(window=14).mean()

    # --- ADX ---
    up_move = df[high_col].diff()
    down_move = df[low_col].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = df[f"{tf_name}_ATR"].replace(0, 1e-10)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10))
    df[f"{tf_name}_ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # 1. 正規化ボラティリティ
    df[f"{tf_name}_norm_atr_pct"] = (df[f"{tf_name}_ATR"] / df[close_col]) * 100

    # 2. ノイズ率 (Wick-to-Body Ratio)
    body = abs(df[close_col] - df[open_col])
    wick_total = (df[high_col] - df[low_col]) - body
    df[f"{tf_name}_noise_ratio"] = wick_total / (body + 1e-8)

    # 3. 短期的な値幅の標準偏差
    df[f"{tf_name}_volatility_std"] = df[close_col].rolling(window=20).std() / df[close_col] * 100
    
    return df

def main():
    setup_logger()
    logging.info("===Launch Data Processing System===")

    parser = argparse.ArgumentParser(description="Preprocess FX Data")
    parser.add_argument("--symbol", type=str, help="Trading symbol(s) separated by comma")
    parser.add_argument("--mode", type=str, choices=["short", "medium", "long"], help="Trading mode")
    args = parser.parse_args()

    # --- 銘柄 (Symbol) の決定 ---
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
    else:
        print(f"Default stocks: {','.join(DEFAULT_SYMGOLS)}")
        symbol_input = input("Please enter the stocks to be processed, separated by commas (press Enter for default): ").strip()
        symbols = [s.strip().upper() for s in symbol_input.split(',')] if symbol_input else DEFAULT_SYMGOLS

    if not symbols:
        logging.error("No stock has been specified.")
        return
    
    # --- モード (Mode) の決定 ---
    if args.mode:
        mode = args.mode.strip().lower()
    else:
        mode = input("Please enter the processing mode.(short/medium/long) [Default: short]: ").strip().lower() or "short"

    if mode not in TRADING_MODE:
        logging.error("invalid mode")
        return
    
    # --- データ処理の実行 ---
    for symbol in symbols:
        logging.info(f"Processing{symbol}data")
        process_mtf_data(symbol,mode)
    
    logging.info("=== all process done===")

if __name__ == "__main__":
    main()