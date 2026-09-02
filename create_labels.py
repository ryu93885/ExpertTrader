import pandas as pd
import os
import glob
import logging
import argparse

# 💡 修正ポイント1: pipsの閾値を、変化率（％）の閾値に変更
# ※USDJPY(150円)の場合、0.03%が約4.5pips、0.1%が約15pips、0.7%が約105pipsに相当
# 運用しながら最適な％に微調整
LABEL_SETTINGS = {
    "short": {
        "base_tf": "M5",
        "lookahead_bars": 12,
        "threshold_factor": 0.6  # ボラティリティ標準偏差の 0.6倍 を売買閾値にする
    },
    "medium": {
        "base_tf": "M15",
        "lookahead_bars": 16,
        "threshold_factor": 1.0  # ボラティリティ標準偏差の 1.0倍 を売買閾値にする
    },
    "long": {
        "base_tf": "H4",
        "lookahead_bars": 30,
        "threshold_factor": 1.5  # ボラティリティ標準偏差の 1.5倍 を売買閾値にする
    }
}

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def create_labels(df, tf_name, lookahead_bars, threshold_factor,symbol):
    """未来の価格変動から、AI学習用の正解ラベル（分類・回帰）を生成する"""
    close_col = f"{tf_name}_close"
    low_col = f"{tf_name}_low"
    high_col = f"{tf_name}_high"

    # --- 1. class label ---
    df['future_close'] = df[close_col].shift(-lookahead_bars)
    
    # 💡 修正ポイント3: 変化率(%) の計算に変更: (未来 - 現在) / 現在 * 100
    df['price_diff_pct'] = ((df['future_close'] - df[close_col]) / df[close_col]) * 100

    valid_diff = df["price_diff_pct"].dropna()
    if len(valid_diff) == 0:
        df["target_class"] = 0
        df["target_risk_pct"] = 0.0
        return df

    # 💡 修正: 旧実装は0を中心に対称な閾値(±std*factor)を使っていたため、
    # ラベル付け期間にその銘柄が一方向にトレンドしていた場合(例: GOLDの上昇局面)、
    # 「これから反転するか」ではなく「その期間たまたま上昇していたか」を学習して
    # しまい、BUY/SELLラベル数が偏っていた。分布の平均(price_diff_mean、その期間の
    # ドリフト成分)を差し引いた上で、標準偏差ベースの閾値と比較するように変更する。
    price_diff_mean = valid_diff.mean()
    price_diff_std = valid_diff.std()
    dynamic_threshold_pct = price_diff_std * threshold_factor
    logging.info(
        f"  📊 銘柄動的解析 [{symbol}]: 平均 = {price_diff_mean:.4f}% / 標準偏差 = {price_diff_std:.4f}% "
        f"➡️ 自動閾値 = 平均 ± {dynamic_threshold_pct:.4f}% (倍率: {threshold_factor}x)"
    )
    df['target_class'] = 0
    df.loc[df['price_diff_pct'] >= price_diff_mean + dynamic_threshold_pct, 'target_class'] = 1
    df.loc[df['price_diff_pct'] <= price_diff_mean - dynamic_threshold_pct, 'target_class'] = -1

    # --- 2. regression label ---
    future_min = df[low_col].rolling(window=lookahead_bars).min().shift(-lookahead_bars)
    future_max = df[high_col].rolling(window=lookahead_bars).max().shift(-lookahead_bars)

    df['target_risk_pct'] = ((future_max - future_min) / df[close_col]) * 100

    # delete NaN
    
    df.dropna(subset=['target_class', 'target_risk_pct'], inplace=True)
    total = len(df)
    if total > 0:
        p_sell = (df['target_class'] == -1).sum() / total * 100
        p_hold = (df['target_class'] == 0).sum() / total * 100
        p_buy  = (df['target_class'] == 1).sum() / total * 100
        logging.info(f"  📈 ラベル分布: SELL = {p_sell:.1f}% | HOLD = {p_hold:.1f}% | BUY = {p_buy:.1f}%")
    return df

def main():
    setup_logger()
    logging.info("===Launching the AI Training Label Generation System ===")

    parser = argparse.ArgumentParser(description="Generate AI Training Labels")
    parser.add_argument("--symbol", type=str, help="Trading symbol(s) separated by comma")
    parser.add_argument("--mode", type=str, choices=["short", "medium", "long"], help="Trading mode")
    args = parser.parse_args()

    # --- 銘柄 (Symbol) の決定 ---
    if args.symbol:
        target_symbols = [s.strip().upper() for s in args.symbol.split(',')]
    else:
        symbol_input = input("Enter the stocks to process, separated by commas (press Enter to process all available): ").strip()
        target_symbols = [s.strip().upper() for s in symbol_input.split(',')] if symbol_input else []

    # --- モード (Mode) の決定 ---
    if args.mode:
        target_mode = args.mode.strip().lower()
    else:
        target_mode = input("Enter the processing mode (short/medium/long) [press Enter to process all available]: ").strip().lower()

    files = glob.glob("processed_data/*_processed.csv")
    if not files:
        logging.error("The file to be processed cannot be found")
        return

    processed_count = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) < 3:
            logging.warning(f"The file format is unexpected.skip: {filename}")
            continue
            
        symbol = parts[0]
        mode = parts[1]

        if target_symbols and symbol not in target_symbols:
            continue
        if target_mode and mode != target_mode:
            continue

        if mode not in LABEL_SETTINGS:
            logging.warning(f"unexpected mode: '{mode}' ,skip: {filename}")
            continue

        logging.info(f"generating labels: {file_path} (mode: {mode})")
        
        settings = LABEL_SETTINGS[mode]
        base_tf = settings["base_tf"]
        lookahead_bars = settings["lookahead_bars"]
        threshold_factor = settings["threshold_factor"]

        df = pd.read_csv(file_path)
        
        df_labeled = create_labels(df,base_tf,lookahead_bars,threshold_factor,symbol)
        
        out_dir = "labeled_data"
        os.makedirs(out_dir, exist_ok=True)
        out_file = f"{out_dir}/{filename.replace('_processed', '_labeled')}"
        df_labeled.to_csv(out_file, index=False)
        logging.info(f"save complete: {out_file} (num of data: {len(df_labeled)})\n")
        
        processed_count += 1
        
    if processed_count == 0:
        logging.warning("No files matched the specified symbol/mode criteria.")
    else:
        logging.info(f"=== All processing complete. Processed {processed_count} files. ===")

if __name__ == "__main__":
    main()