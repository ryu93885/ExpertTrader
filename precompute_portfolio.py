import argparse
import sys
import errno
import time
import pickle
import torch
from torch.utils.data import DataLoader,dataloader,Subset
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

# 💡 追加(2026-11-20の検証で判明した問題への対応): evaluate.pyが選定した最新モデルは
# このDrive直下のパスにのみ保存される(evaluate.py側のdrive_model_dir参照)。
DRIVE_MODEL_DIR = "/content/drive/MyDrive/FX_AI_Models"

# 💡 追加(2026-10-30の調査で判明した問題への対応): Google Drive(FUSEマウント)への
# アデセスが集中すると、接続自体が切断されることがある(OSError: [Errno 107]
# Transport endpoint is not connected / [Errno 5] Input/output error)。
# 一時的な切断はリトリイで吸収する。
DRIVE_IO_MAX_RETRIES = 5
DRIVE_IO_RETRY_BACKOFF_BASE = 10  # 秒。リトリイ毎に倍増

# 💡 追加(2026-11-03の調査で判明した問題への対応): 以前はzip展開でColabローカル
# ディスクに置いていた画像を、現在はGoogle Drive上に直接展開した構造から読んでおり、
# 個別ファイル読み込みのレイテンシがそのまま積み重なって本体の推論ループが
# 大幅に遅くなっている(実測: 約0.37秒/枚 × 約64万枚 ≒ 65時間、観測されたETAと一致)。
# この速度自体はnum_workersの増加(main()側)で対処するが、それでも長時間(数時間〜)
# かかることに変わりはないため、途中経過を定期的に保存し、ランタイム切断等で中断しても
# 最初からやり直しにならないようにする。
CHECKPOINT_DIR = "precompute_checkpoints"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/FX_AI_Models/precompute_checkpoints"
CHECKPOINT_EVERY_N_BATCHES = 20


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


def sync_best_model_from_drive(mode):
    """
    💡 追加(2026-11-20の検証で判明した問題への対応): evaluate.pyが選定した最新モデルは
    Google Drive側(DRIVE_MODEL_DIR/final_best_model_{mode}.pth)にのみ保存され、
    本スクリプトが読むローカルパス(saved_model/final_best_model_{mode}.pth)への反映は
    従来Notebook側の手動セル(shutil.copy)に依存していた。コピーを忘れた場合や、
    同じColabランタイム内に以前の学習結果のローカルコピーが残っている場合、古いモデルが
    無言で使われ続ける危険があった(run_colab_pipeline.py経由の自動実行には、そもそも
    このコピー手順自体が存在しなかった)。ここで毎回Drive側の最新版を強制的にローカルへ
    反映することで、「ローカルに既にあるから」という理由で古いファイルが使われることを防ぐ。
    """
    drive_path = os.path.join(DRIVE_MODEL_DIR, f"final_best_model_{mode}.pth")
    local_path = f"saved_model/final_best_model_{mode}.pth"

    if not os.path.exists(drive_path):
        if os.path.exists(local_path):
            logging.warning(
                f"⚠️ Google Drive上に{drive_path}が見つからないため、ローカルの既存ファイルを"
                f"そのまま使用します(evaluate.pyの最新の実行結果かどうかは保証されません): {local_path}"
            )
            return
        logging.error(f"❌ モデルファイルがGoogle Drive・ローカルのどちらにも見つかりません: {drive_path}")
        sys.exit(1)

    os.makedirs("saved_model", exist_ok=True)
    if os.path.exists(local_path) and os.path.getmtime(local_path) == os.path.getmtime(drive_path):
        logging.info(f"✅ ローカルのモデルは既にGoogle Driveの最新版と同じです: {local_path}")
        return

    shutil.copy2(drive_path, local_path)
    logging.info(f"☁️ Google Driveの最新モデルをローカルに反映しました: {drive_path} -> {local_path}")


def get_checkpoint_num_symbols(checkpoint):
    """
    💡 追加(2026-11-20): evaluate.pyと同じロジック。チェックポイントの
    symbol_embedding.weightの形状から、実際に学習時に使われていた銘柄数を読み取る。
    --exclude-symbolsで学習されたモデル(埋め込み数が7未満)を、外部設定に頼らず
    安全に扱えるようにするため。
    """
    try:
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        weight = state_dict.get("symbol_embedding.weight")
        return int(weight.shape[0]) if weight is not None else None
    except Exception:
        return None


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


def _checkpoint_path(phase, mode):
    return os.path.join(CHECKPOINT_DIR, f"precompute_checkpoint_{phase}_{mode}.pkl")


def _drive_checkpoint_path(phase, mode):
    return os.path.join(DRIVE_CHECKPOINT_DIR, f"precompute_checkpoint_{phase}_{mode}.pkl")


def save_checkpoint(phase, mode, all_probs, all_risks, rows_done, total_rows):
    """
    💡 追加(2026-11-03の調査で判明した問題への対応): 行ごとの推論結果(途中経過)を
    ローカルとGoogle Drive(両方)に保存する。推論ループが長時間(数時間〜)かかる状況で、
    ランタイム切断等により中断してもここまでの計算を失わないようにするため。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _checkpoint_path(phase, mode)
    data = {
        "all_probs": all_probs,
        "all_risks": all_risks,
        "rows_done": rows_done,
        "total_rows": total_rows,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    if os.path.exists("/content/drive/MyDrive"):
        try:
            os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
            shutil.copy2(path, _drive_checkpoint_path(phase, mode))
        except Exception as e:
            logging.warning(f"⚠️ チェックポイントのGoogle Driveへのバックアップに失敗しました: {e}")


def load_checkpoint(phase, mode, total_rows):
    """
    💡 追加(2026-11-03の調査で判明した問題への対応): 保存済みのチェックポイントがあれば
    読み込み、そこから再開できるようにする。ローカルに無い場合はGoogle Driveからの
    復元も試みる(ランタイム再起動でローカルディスクが失われた場合に対応)。
    dataset側の行数が保存時と一致しない場合(merged_csvの内容が変わった等)は、
    不整合な再開を避けるため使用しない。
    """
    path = _checkpoint_path(phase, mode)
    if not os.path.exists(path):
        d_path = _drive_checkpoint_path(phase, mode)
        if os.path.exists(d_path):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            shutil.copy2(d_path, path)
            logging.info(f"☁️ Google Driveからチェックポイントを復元しました: {d_path}")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logging.warning(f"⚠️ チェックポイントの読み込みに失敗しました。最初から開始します: {e}")
        return None

    if data.get("total_rows") != total_rows:
        logging.warning(
            "⚠️ チェックポイントの行数が現在のデータと一致しないため、使用せず最初から"
            f"開始します(チェックポイント: {data.get('total_rows')}行, 現在: {total_rows}行)。"
        )
        return None

    return data


def _delete_checkpoint(phase, mode):
    for p in (_checkpoint_path(phase, mode), _drive_checkpoint_path(phase, mode)):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                logging.warning(f"⚠️ チェックポイントの削除に失敗しました ({p}): {e}")


def main():
    logging.info("precompute_portfolio.py ver215.0")
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
    # 💡 追加(2026-11-03の調査で判明した問題への対応): Google Drive上の個別ファイル
    # 読み込みはレイテンシが支配的で、num_workers=2では待ち時間が直列に積み重なり
    # 極端に遅くなっていた(実測: 約65時間)。I/Oバウンドな処理のため、CPUコア数より
    # 多いワーカー数を指定しても問題なく、並列化によって総時間を大きく短縮できる。
    parser.add_argument("--num-workers", type=int, default=16,
                         help="画像読み込みに使うDataLoaderのワーカー数(デフォルト16)")
    parser.add_argument("--exclude-symbols", type=str, default="",
                         help="推論対象から除外する銘柄(train.pyの--exclude-symbolsで学習したモデルを"
                              "使う場合に指定する。カンマ区切り、末尾の銘柄のみ指定可能)")
    args = parser.parse_args()

    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    exclude_symbols = {s.strip() for s in args.exclude_symbols.split(",") if s.strip()}
    if exclude_symbols:
        n = len(exclude_symbols)
        if set(symbols[-n:]) != exclude_symbols:
            logging.error(
                f"❌ --exclude-symbolsは銘柄埋め込みIDの安全性のため、末尾の銘柄のみ除外可能です"
                f"(順序={symbols}, 指定された除外={sorted(exclude_symbols)})"
            )
            sys.exit(1)
        logging.info(f"🚫 以下の銘柄は推論対象から除外します: {sorted(exclude_symbols)}")
        symbols = [s for s in symbols if s not in exclude_symbols]
    mode = args.mode
    merged_csv = f"portfolio_merged_data_{args.phase}.csv"
    img_dir = "images"
    model_path = f"saved_model/final_best_model_{mode}.pth"
    output_csv = f"portfolio_merged_data_{args.phase}_with_preds.csv"

    # 💡 追加(2026-11-03の調査で判明した問題への対応): 指定された(phase, mode)の
    # 事前推論が既に完了している(ローカルまたはGoogle Driveのバックアップ先に完成済みの
    # 出力ファイルが存在する)場合は、計算全体をスキップする。run_colab_pipeline.py側にも
    # 同様のチェック(data_already_prepared)があるが、precompute_portfolio.pyを単体で
    # 実行する場合にも効くよう、ここにも実装する。
    drive_output_path = os.path.join(DRIVE_PORTFOLIO_DATA_DIR, os.path.basename(output_csv))
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        logging.info(f"✅ {output_csv} は既に存在するため、事前推論をスキップします。")
        return
    if os.path.exists(drive_output_path) and os.path.getsize(drive_output_path) > 0:
        logging.info(
            f"☁️ Google Driveに完成済みの{os.path.basename(output_csv)}が見つかったため、"
            f"復元してスキップします: {drive_output_path}"
        )
        shutil.copy2(drive_output_path, output_csv)
        return

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

    #2.loading model
    # 💡 追加(2026-11-20): ローカルのモデルファイルがGoogle Drive上の最新(evaluate.pyの
    # 最新選定結果)と一致しているかを確認・反映する(詳細はsync_best_model_from_drive参照)。
    sync_best_model_from_drive(mode)

    input_dim = len(dataset.feature_cols[symbols[0]])
    # 💡 修正(2026-10-30): モデル重みもDrive上のファイルであり、同じErrno 107/5の
    # リスクがあるため、_with_drive_retryで読み込む。
    checkpoint_model = _with_drive_retry(torch.load, model_path, map_location=device, weights_only=False)

    # 💡 追加(2026-11-20): --exclude-symbolsで学習されたモデル(埋め込み数が7未満)を
    # 安全に扱うため、チェックポイントの実際の埋め込み数を読み取り、このスクリプトが
    # 推論しようとしている銘柄数と一致するか確認する(evaluate.pyと同じ考え方)。
    expected_num_symbols = get_checkpoint_num_symbols(checkpoint_model) or len(symbols)
    if expected_num_symbols != len(symbols):
        logging.error(
            f"❌ モデルの銘柄埋め込み数({expected_num_symbols})と、推論しようとしている銘柄数"
            f"({len(symbols)})が一致しません。--exclude-symbolsの指定を確認してください"
            f"(symbols={symbols})。"
        )
        sys.exit(1)

    model = MultimodalFXmodel(num_tabular_features=input_dim, num_symbols=expected_num_symbols).to(device)
    model.load_state_dict(checkpoint_model["model_state_dict"])
    model.eval()

    #3.listing result of prediction
    # 💡 修正(2026-11-03の調査で判明した問題への対応): チェックポイントがあれば
    # そこから再開する(詳細はload_checkpoint参照)。
    total_rows = len(dataset)
    resumed = load_checkpoint(args.phase, mode, total_rows)
    if resumed is not None:
        all_probs = resumed["all_probs"]
        all_risks = resumed["all_risks"]
        rows_done = resumed["rows_done"]
        logging.info(f"🔄 チェックポイントを検出しました。{rows_done}/{total_rows}行目から再開します。")
    else:
        all_probs = {sym:[] for sym in symbols}
        all_risks = {sym:[] for sym in symbols}
        for _ in range(dataset.seq_length - 1):
            for sym in symbols:
                all_probs[sym].append([0.0,0.0,0.0])
                all_risks[sym].append(0.0)
        rows_done = dataset.seq_length - 1

    logging.info(f"全データの事前推論を開始(バッチ処理、{rows_done}/{total_rows}行目から)")

    if rows_done < total_rows:
        # 💡 修正(2026-11-03): 残り行のみを対象とするSubsetを作ることで、既に完了した
        # 行の画像読み込み(最も時間のかかる処理)を再実行しない。num_workersも
        # args.num_workersに変更(詳細はCLI引数のヘルプ参照)。
        remaining_dataset = Subset(dataset, range(rows_done, total_rows))
        dataloader = DataLoader(
            remaining_dataset, batch_size=64, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn_drop_none,
        )

        with torch.no_grad():
            for batch_i, batch in enumerate(tqdm(dataloader, desc="Pre-computing")):
                if batch is None:
                    continue
                imgs_batch,tabs_batch = batch
                batch_rows = imgs_batch.shape[0]

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

                rows_done += batch_rows

                if (batch_i + 1) % CHECKPOINT_EVERY_N_BATCHES == 0:
                    save_checkpoint(args.phase, mode, all_probs, all_risks, rows_done, total_rows)
                    logging.info(f"💾 チェックポイント保存: {rows_done}/{total_rows}行")

        save_checkpoint(args.phase, mode, all_probs, all_risks, rows_done, total_rows)

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

    # 💡 追加(2026-11-03): 完了したのでチェックポイントは不要。残しておくと、次回
    # merged_csvが変わらないまま誤って古い途中経過から再開してしまうリスキがあるため削除する。
    _delete_checkpoint(args.phase, mode)

if __name__ == "__main__":
    main()
