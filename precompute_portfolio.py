import argparse
import sys
import errno
import time
import torch
from torch.utils.data import DataLoader,dataloader
import pandas as pd
import numpy as np
import os
import logging
import shutil
from tqdm import tqdm

from portfolio_dataset import PortfolioFXDataset
from model import MultimodalFXmodel, SYMBOL_TO_ID

logging.basicConfig(level=logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")

# 💡 追加: merge_portfolio_data.py と同じDrive保存先(永続化用)。
DRIVE_PORTFOLIO_DATA_DIR = "/content/drive/MyDrive/FX_AI_Models/portfolio_data"

# 💡 追加(2026-10-30の調査で判明した問題への対応): Google Drive(FUSEマウント)への
# アデセスが集中すると、接続自体が切断されることがある(OSError: [Errno 107]
# Transport endpoint is not connected / [Errno 5] Input/output error)。
# 一時的な切断はリトリイで吸収する。
DRIVE_IO_MAX_RETRIES = 5
DRIVE_IO_RETRY_BACKOFF_BASE = 10  # 秒。リトリイ毎に倍増


def backup_to_drive(local_path):
    """Google Driveがマウントされていれば、生成したファイルをコピーして永続化する。"""
    if not os.path.exists("/content/drive/MyDrive"):
        return
    try:
        os.makedirs(DRIVE_PORTFOLIO_DATA_DIR, exist_ok=True)
        dest = os.path.join(DRIVE_PORTFOLIO_DATA_DIR, os.path.basename(local_path))
        shutil.copy2(local_path, dest)
        logging.info(f"☁️ Google Driveにバックアップしました: {dest}")
    except Exception as e:
        logging.error(f"⚠️ Google Driveへのバックアップに失敗しました ({local_path}): {e}")


def collate_fn_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return dataloader.default_collate(batch)


def _scandir_with_retry(path):
    """
    💡 追加(2026-10-30): os.scandir()をリトライ付きで実行する。
    Driveの接続が一時的に切れている場合(Errno 107/5)は待機して再試行し、
    それ以外のエラー(フォルダが存在しない等)は即座に呼び出し元に伝える。
    """
    last_err = None
    for attempt in range(DRIVE_IO_MAX_RETRIES):
        try:
            with os.scandir(path) as it:
                return {entry.name: entry for entry in it}
        except OSError as e:
            if e.errno not in (errno.ENOTCONN, errno.EIO):
                raise
            last_err = e
            wait = DRIVE_IO_RETRY_BACKOFF_BASE * (2 ** attempt)
            logging.warning(
                f"⚠️ Google Driveへの接続が一時的に切断されたようです({path}): {e}"
                f" -- {wait}秒後に再試行します({attempt + 1}/{DRIVE_IO_MAX_RETRIES})"
            )
            time.sleep(wait)
    raise last_err


def check_images_complete(dataset, symbols, mode, img_dir):
    """
    💡 追加(2026-10-23の検証で判明した重大な問題への対応):
    このあとの推論ループは画像を分析モデルに通し、その結果(prob_0/1/2・risk_val)を
    CSVに焼き付ける。PortfolioFXDataset側は画像が欠損・破損していても例外を出さず
    黒画像(ゼロテンソル)にフォールバックするため、画像展開(zip展開・rclone同期等)が
    不完全な状態でこのスクリプトを実行すると、無意味な推論結果がエラーなくCSVに
    書き込まれてしまう。しかもこのCSVは以降の学習・バックテストすべてで無条件に
    信用されるため、一度汚染されると気づかれないまま使われ続ける。
    そのため、実際に推論処理を始める前に、必要な画像がすべて揃っているかを検証する。

    💡 修正(2026-10-30の調査で判明した問題への対応): 以前は期待する全(時刻×銘柄)
    ペアごとにos.path.exists()+os.path.getsize()を個別に呼んでいたため、mediumモード
    の全期間では数十万回規模のDriveアクセスが発生していた。この規模の連続アクセスで
    Google Drive(FUSEマウント)の接続自体が切断され(OSError: [Errno 107] Transport
    endpoint is not connected)、このチェック自体がprecompute_portfolio.py全体を
    クラッシュさせる事例が発生した。銘柄ごとに1回だけフォルダを列挙する方式に変更し、
    個々の判定はメモリ上で行うことでDriveへのアクセス回数を大幅に削減する。
    あわせて、一時的な切断はリトライで吸収し、復旧不能な場合は明示的なエラーとして
    処理を中止する(無言のクラッシュにはしない)。
    """
    time_strs_to_check = dataset.df["time_str"].values[dataset.seq_length - 1:]
    missing_by_symbol = {sym: [] for sym in symbols}
    total_checked = 0

    for sym in symbols:
        folder = os.path.join(img_dir, sym, f"{mode}_MTF")
        try:
            entries = _scandir_with_retry(folder)
        except OSError as e:
            logging.error(f"❌ 画像フォルダにアクセスできません: {folder} ({e})")
            logging.error(
                "Google Driveの接続切断、またはフォルダ自体が未展開であることが考えられます。"
                "Googleドライブの再マウント・Colabランタイムの再起動、または"
                "画像展開(zip展開・rclone同期等)の完了状況を確認してから再実行してください。"
            )
            return False

        for time_str in time_strs_to_check:
            total_checked += 1
            filename = f"{time_str}.png"
            entry = entries.get(filename)
            is_present = False
            if entry is not None:
                try:
                    is_present = entry.stat().st_size > 0
                except OSError:
                    is_present = False
            if not is_present:
                missing_by_symbol[sym].append(os.path.join(folder, filename))

    total_missing = sum(len(v) for v in missing_by_symbol.values())
    if total_missing == 0:
        logging.info(f"✅ 画像の完全性チェックOK({total_checked}件すべて存在)。事前推論を開始します。")
        return True

    logging.error(f"❌ 画像が{total_missing}件(/{total_checked}件)欠損・破損しています。事前推論を中止します。")
    for sym, paths in missing_by_symbol.items():
        if paths:
            logging.error(f"  [{sym}] {len(paths)}件欠損。例:")
            for p in paths[:5]:
                logging.error(f"    - {p}")
    logging.error(
        "画像展開(zip展開・rclone同期等)が完全に終わっているか確認してから再実行してください。"
        "欠損を許容してでも続行したい場合は --allow-missing-images を付けてください"
        "(その場合、該当箇所の精度は保証されません)。"
    )
    return False


def main():
    logging.info("precompute_portfolio.py ver211.0")
    # 💡 修正: 以前は "train" フェーズ専用に固定されており、test_portfolio.py が
    # 必要とする "portfolio_merged_data_test_with_preds.csv" を作る手段が存在しなかった。
    # (merge_portfolio_data.py は train/test 両方を出力するのに、precompute側が
    #  trainしか処理できず、test側は prob_0/1/2・risk_val が永遠に追加されないままだった)
    parser = argparse.ArgumentParser(description="ポートフォリオ用の事前推論(prob/risk列の追加)")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"], help="対象フェーズ")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="対象モード")
    parser.add_argument("--allow-missing-images", action="store_true",
                         help="画像の欠損を検出しても処理を中止せず続行する(欠損箇所は黒画像で代替され、精度は保証されない)")
    args = parser.parse_args()

    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    mode = args.mode
    merged_csv = f"portfolio_merged_data_{args.phase}.csv"
    img_dir = "images"
    model_path = f"saved_model/final_best_model_{mode}.pth"
    output_csv = f"portfolio_merged_data_{args.phase}_with_preds.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"デバイス:{device}")

    #1.preparing dataset and dataloader
    dataset = PortfolioFXDataset(merged_csv_path=merged_csv,base_img_dir = img_dir,symbols = symbols,mode = mode)

    # 💡 追加: 推論前に画像の完全性を検証する(詳細はcheck_images_complete参照)。
    images_ok = check_images_complete(dataset, symbols, mode, img_dir)
    if not images_ok and not args.allow_missing_images:
        logging.error("処理を中止しました。画像を揃えてから再実行してください。")
        # 💡 修正(2026-10-30): 以前はここが return のみだったため終了コードが0(正常終了)に
        # なり、run_colab_pipeline.py の run_command() がこの中止を検出できず「✅ 完了」と
        # 表示してそのまま後続処理に進んでしまっていた(結果、後続のステップが存在しない
        # ファイルを参照して別のエラーで落ち、原因がわかりにくくなっていた)。
        # sys.exit(1)で明示的に異常終了を伝える。
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_fn_drop_none)

    #2.loading model
    input_dim = len(dataset.feature_cols[symbols[0]])
    model = MultimodalFXmodel(num_tabular_features=input_dim).to(device)
    model.load_state_dict(torch.load(model_path,map_location = device,weights_only=False)["model_state_dict"])
    model.eval()

    #3.listing result of prediction
    all_probs = {sym:[] for sym in symbols}
    all_risks = {sym:[] for sym in symbols}

    logging.info("全データの事前推論を開始(バッチ処理)")

    for _ in range(dataset.seq_length - 1):
        for sym in symbols:
            all_probs[sym].append([0.0,0.0,0.0])
            all_risks[sym].append(0.0)


    with torch.no_grad():
        for batch in tqdm(dataloader,desc = "Pre-computing"):
            if batch is None:continue
            imgs_batch,tabs_batch = batch

            for i,sym in enumerate(symbols):
                sym_imgs = imgs_batch[:,i,:,:,:].to(device)
                sym_tabs = tabs_batch[:,i,:,:].to(device)
                # 💡 B案: 銘柄埋め込み用のID。バッチ内の全サンプルがこの銘柄なので、
                # 同じIDをバッチサイズ分並べる(学習側dataset.pyと同じSYMBOL_TO_IDを使用)。
                sym_ids = torch.full((sym_imgs.shape[0],), SYMBOL_TO_ID[sym], dtype=torch.long, device=device)

                class_out,risk_out = model(sym_imgs,sym_tabs,sym_ids)
                probs = torch.softmax(class_out,dim = 1).cpu().numpy()
                risks = risk_out.cpu().numpy()

                all_probs[sym].extend(probs.tolist())
                all_risks[sym].extend(risks.flatten().tolist())

    #4.saving CSV
    df = pd.read_csv(merged_csv)
    for sym in symbols:
        df[f"{sym}_prob_0"] = [p[0] for p in all_probs[sym]]
        df[f"{sym}_prob_1"] = [p[1] for p in all_probs[sym]]
        df[f"{sym}_prob_2"] = [p[2] for p in all_probs[sym]]
        df[f"{sym}_risk_val"] = all_risks[sym]
    df.to_csv(output_csv, index=False)
    logging.info(f"✨ 事前推論完了！ {output_csv} に保存しました。")
    backup_to_drive(output_csv)

if __name__ == "__main__":
    main()
