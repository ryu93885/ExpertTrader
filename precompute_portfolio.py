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


def _with_drive_retry(func, *args, **kwargs):
    """
    💡 追加(2026-10-30の調査で判明した問題への対応): Google Drive(FUSEマウント)上の
    ファイルを読み書きする呼び出し(pd.read_csv・torch.load・os.scandir・to_csv等)を、
    一時的な接続切断(OSError: [Errno 107] Transport endpoint is not connected /
    [Errno 5] Input/output error)に対してリトリイする汎用リーパー。
    実際に、この種の切断が原因でCSV読み込み中にprecompute_portfolio.py全体が
    クラッシュする事例が発生した(pandasが例外処理中にさらに同じ例外を出し、
    二重例外として生のまま落ちていた)。それ以外のエラーは即座に呼び出し元に伝える。
    """
    last_err = None
    for attempt in range(DRIVE_IO_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno not in (errno.ENOTCONN, errno.EIO):
                raise
            last_err = e
            wait = DRIVE_IO_RETRY_BACKOFF_BASE * (2 ** attempt)
            logging.warning(
                f"⚠️ Google Driveへの接続が一時的に切断されたようです: {e}"
                f" -- {wait}秒後に再試行します({attempt + 1}/{DRIVE_IO_MAX_RETRIES})"
            )
            time.sleep(wait)
    raise last_err


def check_images_complete(dataset, symbols, max_missing_pct, allow_missing_images):
    """
    💡 追加(2026-10-23の検証で判明した重大な問題への対応):
    このあとの推論ループは画像を分析モデルに通し、その結果(prob_0/1/2・risk_val)を
    CSVに焼き付ける。画像が欠損・破損した状態で推論すると、気づかれないまま汚染された
    CSVが生成され、以降の学習・バックテストすべてで無条件に信用されてしまう。そのため、
    実際に推論を始める前に、画像の欠損状況を検証し、閾値を超える場合は明示的に中止する。

    💡 修正(2026-10-31の調査で判明した問題への対応): 画像の列挙・欠損判定自体は
    PortfolioFXDataset.__init__ 側(_resolve_image_paths)に一本化した(Driveへの
    スキャンをここで重複させないため)。ここではその結果(dataset.missing_image_info)を
    読んで、銘柄ごとの欠損率が閾値(max_missing_pct)を超えるかどうかだけを判断する。
    閾値未満の欠損は、データセット側で直前/直後の実在画像に自動的に解決されている
    (銘柄ごとの取引カレンダーの違い等による、ごく少数の避けられない欠損を想定)ため、
    警告のみで続行する。閾値以上の欠損は、--allow-missing-images が明示されない限り
    中止する(原因調査には diagnose_missing_images.py を使う)。
    """
    any_blocking = False
    for sym in symbols:
        info = dataset.missing_image_info[sym]
        total = info["total"]
        missing = info["missing_count"]
        pct = missing / total * 100 if total else 0.0

        if missing == 0:
            logging.info(f"✅ [{sym}] 画像の完全性チェックOK({total}件すべて存在)。")
            continue

        if pct < max_missing_pct:
            logging.warning(
                f"⚠️ [{sym}] 画像が{missing}件(/{total}件, {pct:.3f}%)欠損していますが、"
                f"閾値({max_missing_pct}%)未満のため、直前/直後の実在画像を使い回して続行します。"
            )
            for p in info["examples"]:
                logging.warning(f"    - {p}")
            continue

        any_blocking = True
        logging.error(
            f"❌ [{sym}] 画像が{missing}件(/{total}件, {pct:.3f}%)欠損しており、"
            f"閾値({max_missing_pct}%)を超えています。"
        )
        for p in info["examples"]:
            logging.error(f"    - {p}")

    if not any_blocking:
        logging.info("事前推論を開始します。")
        return True

    if allow_missing_images:
        logging.warning(
            "⚠️ 閾値を超える欠損がありますが、--allow-missing-images が指定されているため、"
            "直前/直後の実在画像を使い回して続行します(該当箇所の精度は保証されません)。"
        )
        return True

    logging.error(
        "画像展開(zip展開・rclone同期等)が完全に終わっているか確認するか、"
        "diagnose_missing_images.py で原因(特定期間への集中か、ランダムな欠落か等)を"
        "調査してから再実行してください。欠損を許容してでも続行したい場合は"
        "--allow-missing-images を付けてください(その場合、該当箇所の精度は保証されません)。"
    )
    return False


def main():
    logging.info("precompute_portfolio.py ver213.0")
    # 💡 修正: 以前は "train" フェーズ専用に固定されており、test_portfolio.py が
    # 必要とする "portfolio_merged_data_test_with_preds.csv" を作る手段が存在しなかった。
    # (merge_portfolio_data.py は train/test 両方を出力するのに、precompute側が
    #  trainしか処理できず、test側は prob_0/1/2・risk_val が永遠に追加されないままだった)
    parser = argparse.ArgumentParser(description="ポートフォリオ用の事前推論(prob/risk列の追加)")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"], help="対象フェーズ")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="対象モード")
    parser.add_argument("--allow-missing-images", action="store_true",
                         help="--max-missing-pctを超える画像欠損を検出しても処理を中止せず続行する"
                              "(欠損箇所は直前/直後の実在画像で代替され、精度は保証されない)")
    parser.add_argument("--max-missing-pct", type=float, default=0.1,
                         help="銘柄ごとの画像欠損率がこの値(%%)未満であれば、自動的に直前/直後の"
                              "実在画像で解決して続行する(デフォルト0.1%%。銘柄間の取引カレンダーの"
                              "違い等による、ごく少数の避けられない欠損を許容するための閾値)")
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
    images_ok = check_images_complete(dataset, symbols, args.max_missing_pct, args.allow_missing_images)
    if not images_ok:
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
    # 💡 修正(2026-10-30): モデル重みもDrive上のファイルであり、同じErrno 107/5の
    # リスクがあるため、_with_drive_retryで読み込む。
    checkpoint = _with_drive_retry(torch.load, model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    # 💡 修正(2026-10-30): mergeされたCSVの再読み込み・結果の書き出しも同じDrive上の
    # 操作であり、Errno 107/5のリスクがあるため、_with_drive_retryで実行する。
    df = _with_drive_retry(pd.read_csv, merged_csv)
    for sym in symbols:
        df[f"{sym}_prob_0"] = [p[0] for p in all_probs[sym]]
        df[f"{sym}_prob_1"] = [p[1] for p in all_probs[sym]]
        df[f"{sym}_prob_2"] = [p[2] for p in all_probs[sym]]
        df[f"{sym}_risk_val"] = all_risks[sym]
    _with_drive_retry(df.to_csv, output_csv, index=False)
    logging.info(f"✨ 事前推論完了！ {output_csv} に保存しました。")
    backup_to_drive(output_csv)

if __name__ == "__main__":
    main()
