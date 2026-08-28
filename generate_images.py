import pandas as pd
import mplfinance as mpf
import os
import logging
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import argparse

# --- setting ---
DEFAULT_SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD", "GOLD"]
WINDOW_SIZE = 60
MTF_IMAGE_SIZE = (9.33, 9.33)
DPI = 72

OVERWRITE = False #Being false,it generate only new period. else it generate all period entered by user.


TF_RESAMPLE_RULES = {
    "M5": "5T",   # 5分
    "M15": "15T", # 15分
    "M30": "30T", # 30分
    "H1": "1H",   # 1時間
    "H4": "4H",   # 4時間
    "D1": "1D",   # 1日
    "W1": "1W"    # 1週間
}

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_mtf_chart_image(df_base,df_mid,df_long,save_path):
    custom_style = mpf.make_mpf_style(
        base_mpf_style = "charles",
        gridstyle = " ",
        facecolor = "white",
        figcolor = "white"
        )
    ap = [
        mpf.make_addplot(df_mid,type = "candle",panel = 1),
        mpf.make_addplot(df_long,type = "candle",panel = 2)
    ]
    
    fig,axes = mpf.plot(
        df_base,
        type = "candle",
        style = custom_style,
        addplot = ap,
        panel_ratios = (1,1,1),
        figsize = MTF_IMAGE_SIZE,
        axisoff = True,
        returnfig = True,
        tight_layout = True
    )

    fig.savefig(save_path,bbox_inches = "tight",pad_inches = 0.1,dpi = DPI)
    plt.close(fig)

def process_single_mtf_image(args):
    window_base, window_mid, window_long, save_path = args
    try:
        create_mtf_chart_image(window_base, window_mid, window_long, save_path)
        plt.close('all')#メモリの開放
        return True
    except Exception as e:
        return False

def generate_mtf_images_for_symbol(symbol, mode,start_date = None,end_date = None):
    """
    1行（同じ時刻）に並んだM5, M15, M30のデータを切り出して
    1枚のMTF画像を生成するメインロジック
    """
    csv_path = f"processed_data/{symbol}_{mode}_processed.csv"

    if not os.path.exists(csv_path):
        logging.warning(f"Processed CSV file is not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # モードから3つの時間足プレフィックスを決定 (例: short -> M5, M15, M30)
    if mode == "short":
        tfs = ["M5", "M15", "M30"]
    elif mode == "medium":
        tfs = ["M15", "H1", "H4"]
    elif mode == "long":
        tfs = ["H4", "D1", "W1"]
    else:
        logging.error(f"Unknown mode: {mode}")
        return

    base_tf, mid_tf, long_tf = tfs[0], tfs[1], tfs[2]

    # データが存在するかチェック
    if f"{base_tf}_close" not in df.columns or f"{long_tf}_close" not in df.columns:
        logging.warning(f"Required TF columns not found in {csv_path}. Skipping.")
        return

    # 時間軸ごとにOHLCデータを分割し、mplfinance用にリネームするヘルパー関数
    def extract_ohlc(tf_prefix):
        df_tf = df[[f"{tf_prefix}_open", f"{tf_prefix}_high", f"{tf_prefix}_low", f"{tf_prefix}_close"]].copy()
        df_tf.rename(columns={
            f"{tf_prefix}_open": "open",
            f"{tf_prefix}_high": "high",
            f"{tf_prefix}_low": "low",
            f"{tf_prefix}_close": "close"
        }, inplace=True)
        return df_tf

    df_base = extract_ohlc(base_tf)
    df_mid = extract_ohlc(mid_tf)
    df_long = extract_ohlc(long_tf)

    # MTF画像専用の保存ディレクトリ
    img_dir = f"images/{symbol}/{mode}_MTF"
    os.makedirs(img_dir, exist_ok=True)
    
    if len(df) < WINDOW_SIZE:
        logging.warning(f"Not enough data for {symbol} ({mode}) to generate MTF images. Need {WINDOW_SIZE}. Skipping.")
        return

    logging.info(f"Generate {symbol} ({mode}) MTF images. Total data: {len(df)}")

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    tasks = []
    # 進行状況バーと画像生成ループ
    for i in range(WINDOW_SIZE, len(df)):
        current_time = df.index[i]

        if start_dt and current_time < start_dt:
            continue
        if end_dt and current_time > end_dt:
            break
        timestamp_str = df.index[i].strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(img_dir, f"{timestamp_str}.png")

        if not OVERWRITE and os.path.exists(save_path):
            continue

        # 過去 WINDOW_SIZE (60) 本分のデータを一括で切り出し
        # 💡 修正: 旧実装は iloc[i-60:i] で判断時点(行i)自体を含んでおらず、
        # (a) 同じ学習データのタブラー特徴量側(dataset.py)は iloc[start:target_idx+1] で
        #     行iを含んでいるため、画像とタブラーで基準時刻がズレていた
        # (b) ライブ側(portfolio_trading_bot.py)の画像は raw_df.tail(60) で現在足を
        #     含んでいるため、学習時とライブで見せている情報が1本分ズレていた
        # 判断時点(行i)を含む直近60本になるよう、窓を1本分だけ後ろにずらす。
        window_base = df_base.iloc[i - WINDOW_SIZE + 1 : i + 1]
        window_mid  = df_mid.iloc[i - WINDOW_SIZE + 1 : i + 1]
        window_long = df_long.iloc[i - WINDOW_SIZE + 1 : i + 1]

        # 1枚のMTF画像として描画・保存
        tasks.append((window_base,window_mid,window_long,save_path))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_mtf_image,tasks),total = len(tasks),desc=f"{symbol} {mode} MTF Images"))

def main():
    setup_logger()
    logging.info("=== AI Chart Image Generation System Launch ===")

    parser = argparse.ArgumentParser(description="Generate Chart Images")
    parser.add_argument("--symbol", type=str, help="Trading symbol(s) separated by comma")
    parser.add_argument("--mode", type=str, choices=["short", "medium", "long"], help="Trading mode")
    parser.add_argument("--tfs", type=str, help="Timeframes separated by comma (e.g., M5,M15,M30)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # --- 銘柄 (Symbol) の決定 ---
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]
    else:
        print(f"Default stocks: {','.join(DEFAULT_SYMBOLS)}")
        symbol_input = input("Enter the stocks to process, separated by commas (press Enter for default): ").strip()
        symbols = [s.strip().upper() for s in symbol_input.split(",")] if symbol_input else DEFAULT_SYMBOLS

    if not symbols:
        logging.error("No stock has been specified.")
        return

    # --- モード (Mode) の決定 ---
    target_mode = args.mode if args.mode else input("Enter mode (short/medium/long) [Default: short]: ").strip().lower() or "short"

    # --- 時間足 (Timeframes) の自動決定 ---
    if target_mode == "short":
        target_tfs = ["M5", "M15", "M30"]
    elif target_mode == "medium":
        target_tfs = ["M15", "H1", "H4"]
    elif target_mode == "long":
        target_tfs = ["H4", "D1", "W1"]
    else:
        target_tfs = ["M5", "M15", "M30"]

    logging.info(f"Target Symbols    : {', '.join(symbols)}")
    logging.info(f"Target Mode       : {target_mode}")
    logging.info(f"Target Timeframes : {', '.join(target_tfs)}")

    for symbol in symbols:
        logging.info(f"\n--- Starting generation for {symbol} | Mode: {target_mode} ---")
        generate_mtf_images_for_symbol(symbol,target_mode,args.start,args.end)

    logging.info("=== All processing complete ===")

if __name__ == "__main__":
    main()