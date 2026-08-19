import pandas as pd
import os
import glob
import logging

# ==========================================
# Label Settings (Same as Training Data)
# ==========================================
LABEL_SETTINGS = {
    "short": {
        "base_tf": "M5",
        "lookahead_bars": 12,
        "threshold_pips": 5.0
    },
    "medium": {
        "base_tf": "M15",
        "lookahead_bars": 16,
        "threshold_pips": 15.0
    },
    "long": {
        "base_tf": "H4",
        "lookahead_bars": 30,
        "threshold_pips": 100.0
    }
}

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def create_labels(df, tf_name, lookahead_bars, threshold_pips, pip_value):
    """Generate correct labels (classification and regression) for AI training from future price movements"""
    close_col = f"{tf_name}_close"
    low_col = f"{tf_name}_low"
    high_col = f"{tf_name}_high"

    # --- 1. class label ---
    df['future_close'] = df[close_col].shift(-lookahead_bars)
    df['price_diff_pips'] = (df['future_close'] - df[close_col]) / pip_value

    df['target_class'] = 0
    df.loc[df['price_diff_pips'] >= threshold_pips, 'target_class'] = 1
    df.loc[df['price_diff_pips'] <= -threshold_pips, 'target_class'] = -1

    # --- 2. regression label ---
    future_min = df[low_col].rolling(window=lookahead_bars).min().shift(-lookahead_bars)
    future_max = df[high_col].rolling(window=lookahead_bars).max().shift(-lookahead_bars)
    
    df['target_risk_pips'] = (future_max - future_min) / pip_value

    # delete NaN
    df.dropna(subset=['target_class', 'target_risk_pips'], inplace=True)
    
    return df

def main():
    setup_logger()
    logging.info("=== Launching the AI Validation Data Generation System ===")

    # 🟢 Explicitly target ONLY 'test' files to create validation data
    files = glob.glob("processed_data/*_test_*_processed.csv")
    
    if not files:
        logging.error("The validation files (test data) cannot be found in processed_data/")
        return

    for file_path in files:
        # ex: "processed_data/USDJPY_test_medium_processed.csv" -> ["USDJPY", "test", "medium", "processed.csv"]
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) < 3:
            logging.warning(f"The file format is unexpected. skip: {filename}")
            continue
            
        symbol = parts[0]
        mode = parts[2]

        if mode not in LABEL_SETTINGS:
            logging.warning(f"Unexpected mode: '{mode}', skip: {filename}")
            continue

        logging.info(f"Generating validation labels: {file_path} (mode: {mode})")
        
        # seek setting of mode
        settings = LABEL_SETTINGS[mode]
        base_tf = settings["base_tf"]
        lookahead_bars = settings["lookahead_bars"]
        threshold_pips = settings["threshold_pips"]
        
        pip_value = 0.01 if "JPY" in symbol else 0.0001

        df = pd.read_csv(file_path)
        df_labeled = create_labels(df, base_tf, lookahead_bars, threshold_pips, pip_value)
        
        out_dir = "labeled_data"
        os.makedirs(out_dir, exist_ok=True)
        out_file = f"{out_dir}/{filename.replace('_processed', '_labeled').replace("_test_","_val_")}"
        
        df_labeled.to_csv(out_file, index=False)
        logging.info(f"Save complete: {out_file} (num of data: {len(df_labeled)})\n")

if __name__ == "__main__":
    main()