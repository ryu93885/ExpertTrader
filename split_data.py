import os
import argparse
import pandas as pd
import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()
    logging.info("split_data.py ver208.2")
    parser = argparse.ArgumentParser(description="学習用(Train)とテスト用(Test)のデータを物理的に分割します")
    parser.add_argument("--symbol", type=str, default="USDJPY", help="銘柄 (例: USDJPY)")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="モード")
    parser.add_argument("--split_date", type=str, required=True, help="分割する境界の日付 (例: 2024-01-01)")
    parser.add_argument("--train_start_date",type = str,default = "2020-01-01",help = "学習データの開始日")
    args = parser.parse_args()

    DRIVE_BASE = "/content/datafetch_modeltrain"
    data_dir = os.path.join(DRIVE_BASE, "labeled_data")

    # 元データのパス
    scaled_csv = os.path.join(data_dir, f"{args.symbol}_{args.mode}_scaled.csv")
    raw_csv = os.path.join(data_dir, f"{args.symbol}_{args.mode}_labeled.csv")

    if not os.path.exists(scaled_csv) or not os.path.exists(raw_csv):
        logging.error("元のCSVファイルが見つかりません。パスを確認してください。")
        return

    logging.info(f"データをロード中... (学習開始日:{args.train_start_date},境界日: {args.split_date})")
    scaled_df = pd.read_csv(scaled_csv)
    raw_df = pd.read_csv(raw_csv)

    scaled_df["time"] = pd.to_datetime(scaled_df["time"])
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    split_date = pd.to_datetime(args.split_date)
    
    train_start = pd.to_datetime(args.train_start_date)
    
    
    # Trainデータ抽出 (split_dateより前)
    scaled_train = scaled_df[(scaled_df["time"] >= train_start) & (scaled_df["time"] < split_date)]
    raw_train = raw_df[(raw_df["time"] >= train_start) & (raw_df["time"] < split_date)]

    # Testデータ抽出 (split_date以降)
    scaled_test = scaled_df[scaled_df["time"] >= split_date]
    raw_test = raw_df[raw_df["time"] >= split_date]

    # 保存先パスの生成
    scaled_train_path = os.path.join(data_dir, f"{args.symbol}_{args.mode}_scaled_train.csv")
    raw_train_path = os.path.join(data_dir, f"{args.symbol}_{args.mode}_labeled_train.csv")
    scaled_test_path = os.path.join(data_dir, f"{args.symbol}_{args.mode}_scaled_test.csv")
    raw_test_path = os.path.join(data_dir, f"{args.symbol}_{args.mode}_labeled_test.csv")

    # CSV出力
    scaled_train.to_csv(scaled_train_path, index=False)
    raw_train.to_csv(raw_train_path, index=False)
    scaled_test.to_csv(scaled_test_path, index=False)
    raw_test.to_csv(raw_test_path, index=False)

    logging.info("✅ データの分割と保存が完了しました！")
    logging.info(f"📊 Trainデータ件数: {len(scaled_train)} 行 -> {scaled_train_path}")
    logging.info(f"📊 Testデータ件数:  {len(scaled_test)} 行 -> {scaled_test_path}")

if __name__ == "__main__":
    main()