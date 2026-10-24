import os
import glob
import pandas as pd
import zipfile
import tempfile
import shutil
from datetime import datetime
import argparse


REQUIRED_SCRIPTS = ["train.py", "evaluate.py", "model.py", "dataset.py"]
# そのまま含めるディレクトリ
REQUIRED_DIRS = ["saved_model", "saved_models"]

def export_to_zip(start_date_str, end_date_str, output_zip="fx_data.zip"):
   
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # 時刻が指定されておらず「00:00:00」になっている場合、その日の終わりまでを含める
    if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
        end_date = end_date.replace(hour=23, minute=59, second=59)
        
    print(f"📦 Creating Dataset ZIP ({start_date} to {end_date})...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # 1. 必須スクリプトの追加
            print("--- Adding Scripts & Models ---")
            for script in REQUIRED_SCRIPTS:
                if os.path.exists(script):
                    zipf.write(script, script)
                    print(f"  ✅ Added script: {script}")
                else:
                    print(f"  ⚠️ Warning: {script} not found.")

            # 2. モデル・スケーラー用ディレクトリの追加
            for d in REQUIRED_DIRS:
                if os.path.exists(d):
                    for root, _, files in os.walk(d):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path)
                    print(f"  ✅ Added directory: {d}/")
            
            # 3. labeled_data の期間抽出と追加
            print("\n--- Processing CSV Data ---")
            
            csv_files = glob.glob("labeled_data/*.csv")
            valid_timestamps = set()
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    if 'time' not in df.columns:
                        print(f"  ⚠️ Warning: 'time' column missing in {csv_file}. Skipping.")
                        continue
                        
                    df['time_dt'] = pd.to_datetime(df['time'])
                    
                    # 期間フィルタリング（指定期間のデータ「のみ」を残す）
                    mask = (df['time_dt'] >= start_date) & (df['time_dt'] <= end_date)
                    filtered_df = df.loc[mask].copy()
                    
                    if filtered_df.empty:
                        print(f"  ⏭️ No data in range for {os.path.basename(csv_file)}. Skipped.")
                        continue
                    
                    # 画像抽出用に、抽出された時刻をセットに記録
                    valid_timestamps.update(filtered_df['time_dt'].dt.strftime('%Y%m%d_%H%M%S').tolist())
                    
                    # 一時ディレクトリに抽出済みCSVを保存 (time_dt列は削除して元の形に戻す)
                    filtered_df.drop(columns=['time_dt'], inplace=True)
                    temp_csv_path = os.path.join(temp_dir, os.path.basename(csv_file))
                    filtered_df.to_csv(temp_csv_path, index=False)
                    
                    # 一時CSVをZIP内に書き込む
                    zip_path = f"labeled_data/{os.path.basename(csv_file)}"
                    zipf.write(temp_csv_path, zip_path)
                    print(f"  ✅ Added CSV: {os.path.basename(csv_file)} ({len(filtered_df)} rows)")
                    
                except Exception as e:
                    print(f"  ⚠️ Error processing {csv_file}: {e}")

            # 4. images の期間抽出と追加
            print("\n--- Extracting Matching Images ---")
            img_count = 0
            for root, _, files in os.walk("images"):
                for file in files:
                    if file.endswith(".png"):
                        
                        timestamp_str = file.replace(".png", "")
                        
                        
                        if timestamp_str in valid_timestamps:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, file_path)
                            img_count += 1
            
            print(f"  ✅ Added {img_count} images matching the specified period.")

    print(f"\n🎉 Successfully created '{output_zip}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FX Dataset to ZIP by Date Range")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    export_to_zip(args.start, args.end)