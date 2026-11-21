import subprocess
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

DEFAULT_SYMBOLS = "USDJPY,EURUSD,GBPUSD,EURJPY,GBPJPY,AUDUSD,XAUUSD"

def get_user_input():
    print("="*40)
    print("   AI FX Daily Prediction Pipeline")
    print("="*40)
    
    symbol = input(f"Enter symbol(s) [Default: {DEFAULT_SYMBOLS}]: ").strip().upper() or DEFAULT_SYMBOLS
    mode = input("Enter mode (short/medium/long) [Default: short]: ").strip().lower() or "short"
    
    today = datetime.now()
    default_end = today.strftime("%Y-%m-%d")
    
    print("\n--- Date Settings ---")
    print("The system will automatically detect existing data and skip processed dates.")
    print("For any missing periods, it will fetch up to 30 days of past data for indicators.")
    
    # 日々の運用では「いつまで予測するか（今日）」だけ指定すればOKです。
    predict_end = input(f"Predict end date (YYYY-MM-DD)   [Default: {default_end}]: ").strip() or default_end
    
    return symbol, mode, predict_end

def get_mtf_config(mode):
    if mode == "short": return ["M5", "M15", "M30"]
    elif mode == "medium": return ["M15", "H1", "H4"]
    elif mode == "long": return ["H4", "D1", "W1"]
    else: return ["M5", "M15", "M30"]

def run_script(script_name, args_list):
    cmd = ["python", script_name] + args_list
    cmd_str = " ".join(cmd)
    print(f"\n[>>>] Running: {cmd_str}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] Error occurred while running {script_name}. Pipeline stopped.")
        sys.exit(1)

def get_latest_processed_date(symbol, mode):
    """処理済みの最新日付を自動検知する"""
    csv_path = f"processed_data/{symbol}_test_{mode}_processed.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'time' in df.columns:
                last_time = pd.to_datetime(df['time'].iloc[-1])
                return last_time
        except Exception:
            pass
    return None

def main():
    symbols_str, mode, predict_end = get_user_input()
    symbols = [s.strip() for s in symbols_str.split(",")]
    tfs = get_mtf_config(mode)
    
    print("\n" + "="*40)
    print(f" Daily Prediction Summary:")
    print(f" Symbols:     {symbols_str}")
    print(f" Mode:        {mode}")
    print(f" Target End:  {predict_end}")
    print("="*40)
    
    end_dt = datetime.strptime(predict_end, "%Y-%m-%d")
    
    
    for symbol in symbols:
        print(f"\n" + "-"*40)
        print(f" Processing: {symbol}")
        print("-"*40)
        
        latest_date = get_latest_processed_date(symbol, mode)
        
        if latest_date:
            print(f"[*] Found existing data up to: {latest_date.strftime('%Y-%m-%d %H:%M')}")
            
            fetch_start_dt = latest_date - timedelta(days=30)
            fetch_start = fetch_start_dt.strftime("%Y-%m-%d")
            print(f"[*] Skipping previously processed dates.")
            print(f"[*] Fetching past 30 days from {fetch_start} to calculate indicators accurately.")
        else:
            print(f"[*] No existing data found for {symbol}.")
            fetch_start_dt = end_dt - timedelta(days=30)
            fetch_start = fetch_start_dt.strftime("%Y-%m-%d")
        
        # Step 1: Fetch Data 
        run_script("fetch_data.py", [
            "--symbol", symbol, 
            "--mode", mode, 
            "--train_start", "2023-01-01",  # Dummy
            "--train_end", "2023-01-02",    # Dummy
            "--test_start", fetch_start, 
            "--test_end", predict_end
        ])
        
        # Step 2: Preprocessing
        run_script("preprocessing.py", ["--symbol", symbol, "--mode", mode])
        
        # Step 3: Generate Images
        tfs_str = ",".join(tfs)
        run_script("generate_images.py", ["--symbol", symbol, "--mode", mode, "--tfs", tfs_str])
        
        # Step 4: Run Prediction
        run_script("predict.py", ["--symbol", symbol, "--mode", mode])
        
    print("\n[+] Daily Prediction Pipeline completed successfully!")

if __name__ == "__main__":
    main()