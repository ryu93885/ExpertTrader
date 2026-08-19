import os
import glob
import logging
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch

# 既存のデータセットクラスを読み込む
from dataset import FXMultimodalDataset

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_target_tfs(mode):
    if mode == "short": return ["M1", "M5", "M15"]
    elif mode == "medium": return ["M15", "H1", "H4"]
    elif mode == "long": return ["H4", "D1", "W1"]
    return []

def filter_valid_data(csv_path, img_dir, temp_save_path, start_date, end_date):
    """train.pyと全く同じデータフィルタリング処理"""
    if not os.path.exists(csv_path):
        return False
        
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    if df.empty:
        return False
    
    valid_times = []
    for timestamp in df.index:
        img_path = os.path.join(img_dir, f"{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
        if os.path.exists(img_path):
            valid_times.append(timestamp)
            
    if not valid_times:
        return False
        
    df_valid = df.loc[valid_times].copy()
    df_valid.to_csv(temp_save_path)
    return True

def main():
    setup_logger()
    
    parser = argparse.ArgumentParser(description="Generate Risk Scalers (JSON) for Evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["short", "medium", "long"], 
                        help="Target mode (short, medium, long)")
    parser.add_argument("--start_date", type=str, default="2000-01-01")
    parser.add_argument("--end_date", type=str, default="2099-12-31")
    args = parser.parse_args()

    logging.info("==================================================")
    logging.info("=== Scaler Generation Tool for Local Evaluation ===")
    logging.info(f"Target Mode: {args.mode}")
    logging.info("==================================================")

    
    SCALER_DIR = "saved_models"
    os.makedirs(SCALER_DIR, exist_ok=True)

    train_csv_files = glob.glob(f"labeled_data/*_train_{args.mode}_scaled.csv")
    if not train_csv_files:
        logging.error(f"No training data found for mode '{args.mode}' in labeled_data/")
        return

    risk_scalers = {}

    logging.info("Scanning datasets and calculating precise risk statistics...")

    for file_path in train_csv_files:
        filename = os.path.basename(file_path)
        symbol = filename.split('_')[0]
        
        img_base_dir = f"images/{symbol}/train/"
        if not os.path.exists(img_base_dir): 
            continue
            
        available_tfs = [d for d in os.listdir(img_base_dir) if os.path.isdir(os.path.join(img_base_dir, d))]
        target_tfs = get_target_tfs(args.mode)
        
        for tf in target_tfs:
            if tf not in available_tfs: 
                continue

            IMG_DIR = f"images/{symbol}/train/{tf}"
            temp_train_csv = f"temp_scalergen_{symbol}_{args.mode}_{tf}.csv"

            # データの存在確認と一時CSV作成
            success = filter_valid_data(file_path, IMG_DIR, temp_train_csv, args.start_date, args.end_date)
            
            if success:
                ds_train = FXMultimodalDataset(csv_file=temp_train_csv, img_dir=IMG_DIR)
                if len(ds_train) == 0: 
                    continue

                # 🌟 train.pyと全く同じロジックで平均と標準偏差を計算する
                
                if symbol not in risk_scalers:
                    logging.info(f"Calculating risk scaler for {symbol} (using TF: {tf})...")
                    sample_size = min(1000, len(ds_train))
                    indices = random.sample(range(len(ds_train)), sample_size)
                    
                    # ds_train[i][3] が 'risk'（pips）の生データ
                    sampled_risks = [float(ds_train[i][3]) for i in indices]
                    mean_val = float(np.mean(sampled_risks))
                    std_val = float(np.std(sampled_risks))
                    
                    if std_val == 0 or np.isnan(std_val): 
                        std_val = 1e-6
                        
                    risk_scalers[symbol] = {'mean': mean_val, 'std': std_val}
                    logging.info(f" -> [Success] {symbol}: Mean={mean_val:.4f}, Std={std_val:.4f}")
                    
            # お掃除
            if os.path.exists(temp_train_csv):
                os.remove(temp_train_csv)

    if not risk_scalers:
        logging.error("Failed to generate any scalers. Please check data files.")
        return

    # JSONとして保存
    scaler_path = os.path.join(SCALER_DIR, f"risk_scalers_{args.mode}.json")
    with open(scaler_path, "w") as f:
        json.dump(risk_scalers, f, indent=4)
        
    logging.info("==================================================")
    logging.info(f"✅ Successfully saved exact risk scalers to: {scaler_path}")
    logging.info("You can now safely run evaluate.py on this machine.")

if __name__ == "__main__":
    main()