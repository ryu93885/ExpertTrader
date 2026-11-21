import os
import argparse
import pandas as pd
import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()
    logging.info("split_data.py ver209.0")
    parser = argparse.ArgumentParser(description="学習用(Train)とテスト用(Test)のデータを物理的に分割します")
    parser.add_argument("--symbol", type=str, default="USDJPY", help="銘柄 (例: USDJPY)")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="モード")
    parser.add_argument("--split_date", type=str, default=None,
                         help="分割する境界の日付 (例: 2024-01-01)。省略時は train.py の内部分割"
                              "(時系列70/20/10、残り10%%がtest)と同じ比率から自動計算し、"
                              "train.pyの評価に使うtest期間と一致させる。")
    parser.add_argument("--train_start_date",type = str,default = "2020-01-01",help = "学習データの開始日")
    args = parser.parse_args()

    # 💡 修正: 他の全スクリプトと同じ相対パスに統一。
    # 旧実装は "/content/datafetch_modeltrain" というColab専用の絶対パスで、
    # Codespaces等、別の作業ディレクトリで実行すると必ず失敗していた。
    data_dir = "labeled_data"

    # 元データのパス
    scaled_csv = os.path.join(data_dir, f"{args.symbol}_{args.mode}_scaled.csv")
    raw_csv = os.path.join(data_dir, f"{args.symbol}_{args.mode}_labeled.csv")

    if not os.path.exists(scaled_csv) or not os.path.exists(raw_csv):
        logging.error("元のCSVファイルが見つかりません。パスを確認してください。")
        return

    scaled_df = pd.read_csv(scaled_csv)
    raw_df = pd.read_csv(raw_csv)

    scaled_df["time"] = pd.to_datetime(scaled_df["time"])
    raw_df["time"] = pd.to_datetime(raw_df["time"])

    if args.split_date:
        split_date = pd.to_datetime(args.split_date)
        logging.warning(
            "⚠️ --split_date が明示的に指定されています。train.py が学習時に内部で行う"
            "70/20/10の時系列分割の境界と一致しない場合、ポートフォリオRLの学習(train)"
            "データに、アナリストモデルの評価(test)に使われた期間が混入する恐れがあります。"
        )
    else:
        # 💡 修正: train.py の内部分割ロジック
        # (train_idx=int(n*0.70), val_idx=train_idx+int(n*0.20), 残りがtest)
        # と全く同じ比率・並び順で境界日を自動計算する。これにより、
        # merge_portfolio_data.py の "test" フェーズの期間が、train.py が評価
        # (evaluate.py / best_*_model選定)に使う test_data/ の期間と一致し、
        # ポートフォリオRLの学習データへのデータ漏洩を防ぐ。
        df_sorted = scaled_df.sort_values("time").reset_index(drop=True)
        n = len(df_sorted)
        val_idx = int(n * 0.70) + int(n * 0.20)
        if n == 0 or val_idx >= n:
            logging.error("データ件数が少なすぎて、境界日を自動計算できません。--split_date を明示的に指定してください。")
            return
        split_date = df_sorted.iloc[val_idx]["time"]
        logging.info(f"📐 --split_date 未指定のため、train.pyと同じ70/20/10比率から境界日を自動計算しました: {split_date}")

    train_start = pd.to_datetime(args.train_start_date)

    logging.info(f"データをロード中... (学習開始日:{args.train_start_date},境界日: {split_date})")

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