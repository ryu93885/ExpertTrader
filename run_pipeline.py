import subprocess
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

DEFAULT_SYMBOLS = "USDJPY,EURUSD,GBPUSD,EURJPY,GBPJPY,AUDUSD,GOLD"

# --- モード別の推奨最低データ ---
# AIが相場環境（レジーム）を学習するために最低限必要な行数の目安です
MIN_DATA_REQUIREMENTS = {
    "short": 10000,  # 短期(M5等): サンプル数が集まりやすいため多めに要求
    "medium": 3000,  # 中期(H1等): 数ヶ月〜数年分
    "long": 1000     # 長期(H4/D1等): 長期間のデータが必要
}

def get_user_input():
    print("="*40)
    print("   AI FX Prediction Pipeline Setup")
    print("="*40)
    
    print(f"Default Symbols: {DEFAULT_SYMBOLS}")
    symbol_str = input("Enter symbol(s) separated by comma [Press Enter for default]: ").strip().upper() or DEFAULT_SYMBOLS
    mode = input("Enter mode (short/medium/long) [Default: short]: ").strip().lower() or "short"
    
    print("\n--- Date Settings ---")
    print("Notice: The system will automatically skip periods that already have data.")
    # train / test の区別をなくし、全データ期間として統合
    start_date = input("Start date (YYYY-MM-DD) [Default: 2020-01-01]: ").strip() or "2020-01-01"
    end_date = input("End date (YYYY-MM-DD)   [Default: 2024-12-31]: ").strip() or "2024-12-31"
    
    return symbol_str, mode, start_date, end_date

def get_mtf_config(mode):
    """
    Returns the Multi-Timeframe list based on the mode.
    """
    if mode == "short":
        return ["M5", "M15", "M30"]
    elif mode == "medium":
        return ["M15", "H1", "H4"]
    elif mode == "long":
        return ["H4", "D1", "W1"]
    else:
        return ["M5", "M15", "M30"]

def run_script(script_name, args_list):
    """Executes a python script with given arguments"""
    cmd = ["python", script_name] + args_list
    cmd_str = " ".join(cmd)
    print(f"\n[>>>] Running: {cmd_str}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n[!] Error occurred while running {script_name}. Pipeline stopped.")
        sys.exit(1)

def get_latest_processed_date(symbol, mode):
    """処理済みの最新日付を自動検知する（periodの区別なし）"""
    csv_path = f"processed_data/{symbol}_{mode}_processed.csv"
    if os.path.exists(csv_path):
        try:
            # メモリ節約のため time 列のみ読み込む
            df = pd.read_csv(csv_path, usecols=['time'])
            if not df.empty:
                last_time = pd.to_datetime(df['time'].iloc[-1])
                return last_time
        except Exception:
            pass
    return None

def verify_data_volume(symbol, mode):
    """生成されたデータの件数を検証し、モードに対して適切かチェックする"""
    print(f"\n--- Data Verification for {symbol} ({mode} mode) ---")
    
    threshold = MIN_DATA_REQUIREMENTS.get(mode, 1000)
    
    # 統合されたデータファイルのパス（環境に合わせて processed_data か labeled_data か調整してください）
    file_path = f"labeled_data/{symbol}_{mode}_labeled.csv"
    if not os.path.exists(file_path):
        # 古いパス構造や別スクリプトの出力先にフォールバック
        file_path = f"processed_data/{symbol}_{mode}_labeled.csv"
    
    if os.path.exists(file_path):
        try:
            # 行数だけを高速にカウント
            df = pd.read_csv(file_path, usecols=['time'])
            count = len(df)
            
            print(f"[DATA] Found {count} records in {os.path.basename(file_path)}.")
            
            if count < threshold:
                print(f"  [!] Warning: データ数が {threshold} 件を下回っています。")
                print(f"      AIが {mode} モードの相場環境を学習するには期間が短すぎる可能性があります。")
            else:
                print(f"  [+] Data volume is sufficient for {mode} mode.")
        except Exception as e:
            print(f"  [!] Could not verify data: {e}")
    else:
        print(f"  [!] Verification skipped: labeled data file not found.")

def main():
    # 1. Get configuration
    symbols_str, mode, start_date, end_date = get_user_input()
    symbols = [s.strip() for s in symbols_str.split(",")]
    tfs = get_mtf_config(mode)
    
    print("\n" + "="*40)
    print(f" Configuration Summary:")
    print(f" Symbols:     {symbols_str}")
    print(f" Mode:        {mode}")
    print(f" MTF Sets:    {', '.join(tfs)}")
    print(f" Date Range:  {start_date} to {end_date}")
    print("="*40)
    
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 2. Run Pipeline Steps (銘柄ごとにループして独立処理)
    for symbol in symbols:
        print(f"\n" + "-"*40)
        print(f" Processing Pipeline for: {symbol}")
        print("-" * 40)

        # --- データのチェックと取得期間の調整 ---
        skip_fetch = False
        adj_start = start_date
        latest_date = get_latest_processed_date(symbol, mode)

        if latest_date:
            if latest_date >= end_dt:
                print(f"[*] Data for {symbol} is fully up-to-date (until {latest_date.date()}).")
                skip_fetch = True
            elif latest_date > datetime.strptime(start_date, "%Y-%m-%d"):
                # データが途中まである場合は、インジケータ計算の余白(約30日)を含めて再開地点を設定
                adj = latest_date - timedelta(days=30)
                adj_start = adj.strftime("%Y-%m-%d")
                print(f"[*] Data exists up to {latest_date.date()}. Fetching missing part from {adj_start}.")

        # --- データの取得と前処理の実行制御 ---
        if skip_fetch:
            print(f"[*] Data is fully up-to-date for {symbol}. Skipping Fetch & Preprocessing.")
        else:
            # Step 1: Fetch Data 
            fetch_args = [
                "--symbol", symbol,
                "--mode", mode,
                "--start", adj_start,  # 調整済みの開始日を渡す
                "--end", end_date
            ]
            run_script("fetch_data.py", fetch_args)
            
            # Step 2: Preprocessing
            run_script("preprocessing.py", ["--symbol", symbol, "--mode", mode])

        # Step 3: Generate Images
        tfs_str = ",".join(tfs)
        #run_script("generate_images.py", ["--symbol", symbol, "--mode", mode,"--start",start_date,"--end",end_date])
        
        # Step 4: Label Data
        run_script("create_labels.py", ["--symbol", symbol, "--mode", mode])
        
        run_script("apply_scaler.py", ["--mode", mode,"--symbol",symbol])
        # Step 5: Data Verification (Split Data はColabで行うため除外)
        verify_data_volume(symbol, mode)

    # Step 6: Apply Scaler to all processed data
    print(f"\n" + "-"*40)
    print(f" Applying Scaler to all {mode} mode data")
    print("-" * 40)
    
    run_script("unify_scalers.py",["--mode",mode])
    print("\n[+] All Pipeline tasks and verifications completed successfully!")

if __name__ == "__main__":
    main()