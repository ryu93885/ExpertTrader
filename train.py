import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import os
import glob
import logging
from tqdm import tqdm
import argparse
import json
from PIL import Image
import random
import shutil
import sys
import time
import socket
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data._utils.collate import default_collate

from collections import Counter
from model import MultimodalFXmodel
from dataset import FXMultimodalDataset


try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.colab import auth
    from google.auth import default
    DRIVE_API_AVAILABLE = True
except ImportError:
    DRIVE_API_AVAILABLE = False

SEQ_LENGTH = 40

# 💡 追加(2026-11-07): データ準備(画像検証)のキャッシュ用ディレクトリ・設定。
DATASET_CACHE_DIR = "dataset_cache"
DRIVE_FOLDER_NAME = "FX_AI_Models"  # 既存のモデル保存先と同じDriveフォルダを再利用する
IMAGE_VALIDATION_LOG_INTERVAL = 1000  # この件数ごとに画像検証の進捗ログを出す
DRIVE_UPLOAD_TIMEOUT_SECONDS = 180  # Driveアップロードがこの秒数を超えたらタイムアウトとして扱う
# 💡 追加(2026-11-07の改修): 画像検証(os.path.exists/getsize + PNGのopen+verify)は
# CPUではなくファイルI/O待ち(特にGoogle Driveマウント環境)が支配的なため、
# スレッド並列化で大幅に高速化できる(GILはI/O待ち中に解放されるため)。
VALIDATION_MAX_WORKERS = 16
# この件数ごとに画像検証台帳をディスク(+Drive)へ保存する。1銘柄十万枚規模だと
# 検証だけで数時間かかり得るため、Colabのセッション切断等が起きても、このチャンク
# 単位でしか手戻りが発生しないようにする。
LEDGER_CHECKPOINT_CHUNK_SIZE = 5000

def calculate_auto_class_weights(csv_paths, method="sqrt", device="cpu", sample_ratio=0.2):
    """
    銘柄ごとに独立してクラスウェイトを計算し、それらを単純平均して最終的な重みを決定する。

    💡 修正: 旧実装は全銘柄のラベルを1つのプールに混ぜてから重みを計算していたため、
    行数の多い銘柄(データ期間が長い/取得頻度が高い銘柄)の分布が結果を支配し、
    行数の少ない銘柄固有の偏り(例: GBPJPYのHOLD過多)が薄まって補正されなかった。
    銘柄ごとに重みを個別計算してから単純平均することで、データ量によらず各銘柄が
    等しく重みの決定に反映されるようにする。
    """
    num_classes = 3
    per_symbol_weights = []

    for path in csv_paths:
        filename = os.path.basename(path).replace(".csv", "")
        parts = filename.split("_")
        # tmp_train_csv は "temp_train_{symbol}_{mode}.csv" の形式
        symbol = parts[2] if len(parts) >= 3 else filename

        df = pd.read_csv(path)
        # 1. 各銘柄のラベルを取得 (0: SELL, 1: HOLD, 2: BUY に補正)
        labels = (df["target_class"] + 1).tolist()

        # 2. 全体を読み込むのではなく、時系列の偏りを防ぐために等間隔でサンプリングする
        # 例: sample_ratio=0.2 なら、5行に1行のペースで抽出（時系列の傾向は維持される）
        step = max(1, int(1 / sample_ratio))
        sampled_labels = labels[::step]

        counts = Counter(sampled_labels)
        total_samples = sum(counts.values())
        if total_samples == 0:
            continue

        weights = []
        for i in range(num_classes):
            # ゼロ除算を防ぐため、最低でも1とする
            w = total_samples / (num_classes * counts.get(i, 1))
            weights.append(w)

        if method == "sqrt":
            weights = np.sqrt(weights).tolist()

        # HOLDの重みを1.0として正規化
        hold_weight = weights[1] if weights[1] > 0 else 1.0
        normalized = [w / hold_weight for w in weights]

        per_symbol_weights.append(normalized)
        print(
            f"  [{symbol}] SELL={counts.get(0, 0)}, HOLD={counts.get(1, 0)}, BUY={counts.get(2, 0)} "
            f"-> weight SELL={normalized[0]:.4f} HOLD={normalized[1]:.4f} BUY={normalized[2]:.4f}"
        )

    if not per_symbol_weights:
        logging.warning("クラスウェイトを計算できるデータがありませんでした。重み1.0を使用します。")
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)

    # 銘柄ごとの重みベクトルを単純平均(データ量の多い銘柄に支配されないようにする)
    avg_weights = np.mean(per_symbol_weights, axis=0).tolist()

    print(f"\n--- 自動重み計算 ({method} mode, 銘柄別に計算して平均, サンプリング率: {sample_ratio*100}%) ---")
    print(f"適用される正規化ウェイト(銘柄平均): SELL={avg_weights[0]:.4f}, HOLD={avg_weights[1]:.4f}, BUY={avg_weights[2]:.4f}\n")

    return torch.tensor(avg_weights, dtype=torch.float32).to(device)


def setup_logger():
    # 💡 修正(2026-11-07): 従来はコンソール出力のみで、Colab接続が切れた場合や
    # 長時間ログを遡って見たい場合に、どこで停滞していたかを後から確認できなかった。
    # fetch_data.py等、他のスクリプトと同じくファイルにも残すようにする。
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime(f"{log_dir}/train_%Y%m%d_%H%M%S.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_filename

def get_drive_service():
    if not DRIVE_API_AVAILABLE:
        logging.warning(f"Google API is not able to use.")
        return None
    try:
        
        creds,_ = default()
        return build("drive","v3",credentials = creds)
    except Exception as e:
        logging.error(f"Google Drive 認証エラー:{e}")
        return None
    
def safe_collate_fn(batch):
    """
    バッチの中から None（読み込み失敗データ）を安全に取り除くフィルター
    """
    # batch の中身は (image, tab, cls, risk) のタプルのリスト。None を除外する
    batch = list(filter(lambda x: x is not None, batch))
    
    # もしバッチ内のデータが全て None だった場合の安全対策
    if len(batch) == 0:
        return None
        
    # 生き残った正常なデータだけで、PyTorch標準のバッチ化を行う
    return default_collate(batch)

def upload_to_drive(service,local_path,folder_name = "FX_AI_Models"):
    if service is None or not os.path.exists(local_path):
        return
    
    file_name = os.path.basename(local_path)

    try:
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q = query,fields = "files(id)").execute()
        folders = results.get("files",[])

        if not folders:
            folder_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logging.info(f"Driveにフォルダを作成しました: {folder_name} (ID: {folder_id})")
        else:
            folder_id = folders[0]['id']

        query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])

        media = MediaFileUpload(local_path, resumable=True)
        
        if files:
            
            file_id = files[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
            logging.info(f"Drive上のモデルを更新しました: {file_name}")
        else:
            
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logging.info(f"Driveにモデルを新規保存しました: {file_name}")

    except Exception as e:
        logging.error(f"Driveアップロード失敗 ({file_name}): {e}")


def upload_to_drive_with_timeout(service, local_path, timeout_sec=DRIVE_UPLOAD_TIMEOUT_SECONDS):
    """
    💡 追加(2026-11-07の改修): upload_to_drive()本体(Google Drive API呼び出し)には
    明示的なタイムアウトが無く、ネットワーク不調等で応答が返らない場合、ログに何も
    残らないまま無期限に停滞しているように見えていた(「Driveにファイルを移植して
    タイムアウトが発生するが、どこで止まっているか分からない」という問題の原因の
    一つ)。開始・終了を明示的にログへ残し、socketの既定タイムアウトを一時的に
    設定することでtimeout_sec秒を超えたら例外として検出できるようにする
    (upload_to_drive内のtry/exceptで捕捉されログに残り、処理自体は継続する)。
    """
    if not os.path.exists(local_path):
        return
    file_name = os.path.basename(local_path)
    file_size_mb = os.path.getsize(local_path) / 1e6
    logging.info(f"☁️ Driveへのアップロード開始: {file_name} ({file_size_mb:.1f}MB, タイムアウト設定: {timeout_sec}秒)")
    start_time = time.time()
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_sec)
    try:
        upload_to_drive(service, local_path, folder_name=DRIVE_FOLDER_NAME)
    finally:
        socket.setdefaulttimeout(previous_timeout)
        elapsed = time.time() - start_time
        logging.info(f"☁️ Driveへのアップロード処理終了: {file_name} ({elapsed:.1f}秒経過)")


def get_ledger_paths(symbol, mode):
    """
    💡 追加(2026-11-07の改修): 銘柄ごとの画像検証結果(time, is_valid)を軽量な
    台帳としてキャッシュするためのパス。特徴量そのものはキャッシュしない
    (apply_scaler.py再実行によるスケーラー更新を正しく反映させるため、特徴量は
    毎回最新のlabeled_data/*_scaled.csvから読み直す)。
    """
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    filename = f"image_validity_ledger_{symbol}_{mode}.csv"
    local_path = os.path.join(DATASET_CACHE_DIR, filename)
    drive_path = f"/content/drive/MyDrive/{DRIVE_FOLDER_NAME}/{filename}"
    return local_path, drive_path


def restore_ledger_from_drive_if_missing(local_path, drive_path, symbol):
    if not os.path.exists(local_path) and os.path.exists(drive_path):
        shutil.copy(drive_path, local_path)
        logging.info(f"📥 [{symbol}] Driveから画像検証キャッシュを復元しました: {drive_path}")


def validate_images_with_progress(times, img_dir, symbol, log_interval=IMAGE_VALIDATION_LOG_INTERVAL,
                                   max_workers=VALIDATION_MAX_WORKERS):
    """
    💡 追加(2026-11-07の改修): 1件ずつ逐次でファイルI/O(os.path.exists/getsize +
    PNGのopen+verify)を行っていたため、件数が多いと無言のまま長時間止まっている
    ように見えていた上、実測で1000件あたり約90秒(Driveマウント環境でのI/O待ちが
    支配的)かかっていた。チェックする内容自体(存在確認・サイズ確認・PNGとしての
    デコード検証)は変えずに、以下の2点で高速化する:
      1. ディレクトリの中身を os.scandir() で一度だけ一括取得し、画像ごとの
         個別exists/getsize呼び出し(往復)を無くす。
      2. 残るImage.open+verify()はCPUよりI/O待ちが支配的なため、
         ThreadPoolExecutorで並列化する(GILはI/O待ち中に解放される)。
    log_interval件ごとに進捗(処理速度含む)をログへ残す。
    """
    times_list = list(times)
    n = len(times_list)

    try:
        dir_entries = {e.name: e.stat().st_size for e in os.scandir(img_dir)}
    except FileNotFoundError:
        dir_entries = {}

    def _check_one(t):
        filename = f"{t.strftime('%Y%m%d_%H%M%S')}.png"
        size = dir_entries.get(filename, 0)
        if size <= 0:
            return False
        img_path = os.path.join(img_dir, filename)
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    valid_mask = [False] * n
    start_time = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_check_one, t): i for i, t in enumerate(times_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            valid_mask[idx] = future.result()
            completed += 1
            if completed % log_interval == 0 or completed == n:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                logging.info(f"  [{symbol}] 画像検証中: {completed}/{n}件 ({elapsed:.1f}秒経過, {rate:.1f}件/秒)")
    return valid_mask


def get_target_tfs(mode):
    if mode == "short": return ["M1", "M5", "M15"]
    elif mode == "medium": return ["M15", "H1", "H4"]
    elif mode == "long": return ["H4", "D1", "W1"]
    return []

class ScaledFXDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, mean, std):
        self.dataset = original_dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, tab, cls, risk, symbol_id = self.dataset[idx]
        if isinstance(risk, torch.Tensor):
            scaled_risk = (risk - self.mean) / self.std
        else:
            scaled_risk = torch.tensor((risk - self.mean) / self.std, dtype=torch.float32)
        return img, tab, cls, scaled_risk, symbol_id

def main():
    log_file = setup_logger()
    logging.info(f"train.py ver2.0 (log: {log_file})")
    parser = argparse.ArgumentParser(description="FX Multiple training with date range")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--start", type=str, help="start date(xxxx-xx-xx)")
    parser.add_argument("--end", type=str, help="end date(xxxx-xx-xx)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 💡 修正(2026-11-07の改修): 従来はエポックのループ内で毎回get_drive_service()を
    # 呼び直しており、不要な認証処理が繰り返されていた(Drive関連の遅延・停滞要因の
    # 一つ)。データ準備時のキャッシュアップロードでも使うため、ここで一度だけ取得する。
    drive_service = get_drive_service()
    if drive_service:
        logging.info("☁️ Google Drive連携: 有効")
    else:
        logging.info("☁️ Google Drive連携: 無効(ローカル保存のみ)")

    start_dt = pd.to_datetime(args.start) if args.start else None
    end_dt = pd.to_datetime(args.end) if args.end else None
    if end_dt and end_dt.hour == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    
    csv_files = glob.glob(f"labeled_data/*_{args.mode}_scaled.csv")
    if not csv_files:
        logging.error(f"No scaled data found for mode: {args.mode}")
        return

    # 💡 追加(2026-10-23の検証で判明した問題への対応): 以前はglobで1件でも
    # 見つかれば(7銘柄中いくつ欠けていても)無言で学習を続行してしまっていた。
    # ポートフォリオ側(precompute_portfolio.py等)と同じ7銘柄を期待値として明示し、
    # 欠けている銘柄があればここで気づけるようにする。
    EXPECTED_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    found_symbols = sorted({os.path.basename(p).split("_")[0] for p in csv_files})
    missing_symbols = [s for s in EXPECTED_SYMBOLS if s not in found_symbols]
    if missing_symbols:
        logging.error(
            f"❌ 以下の銘柄のデータが見つからないため、学習を中止します: {missing_symbols}"
            f"(見つかった銘柄: {found_symbols})"
        )
        # 💡 修正(2026-10-30): return のみだと終了コードが0になり、run_colab_pipeline.py
        # がこの中止を検出できず後続処理に進んでしまうため、sys.exit(1)で明示的に
        # 異常終了を伝える。
        sys.exit(1)

    train_datasets = []
    val_datasets = []
    scaler_info = {}
    tmp_train_csv_list = []
    img_dir = "images"
    
    
    os.makedirs("test_data", exist_ok=True)

    # DataFrameのカラム数を取得するために、最初の有効なデータセットから input_dim を計算する用
    first_train_df = None

    # 💡 追加(2026-10-23の検証で判明した問題への対応): CSVは見つかったが、画像フォルダが
    # 無い等の理由で結局その銘柄のデータが1件も使えなかった場合を追跡する。
    # 従来は "train_datasets が空(=全銘柄が失敗)" でなければ気づけなかった。
    loaded_symbols = []

    for csv_path in csv_files:
        symbol = os.path.basename(csv_path).split("_")[0]
        test_csv_path = f"test_data/{symbol}_{args.mode}_test.csv"
        tmp_train_csv = f"temp_train_{symbol}_{args.mode}.csv"
        tmp_val_csv = f"temp_val_{symbol}_{args.mode}.csv"
        current_img_dir = os.path.join("images", symbol, f"{args.mode}_MTF")

        # 💡 修正(2026-11-07の改修): 従来は「前回の分割済みCSVがローカルに残っていれば
        # 画像検証・分割を丸ごとスキップ、無ければ全行を毎回検証」という二択だった。
        # しかしColab等のセッションは使い捨てで、実行終了時にtemp_*.csvも削除していた
        # ため、この高速パスは実質一度も使われず「実行するたびにデータセットを用意」
        # する状態になっていた。特徴量そのものではなく「画像検証結果(time, is_valid)」
        # だけを軽量な台帳としてDriveにも永続化し、次回以降は未検証の新規行だけを
        # 検証するように変更する(特徴量は毎回最新のscaled.csvから読むため、
        # apply_scaler.py再実行によるスケーラー更新も正しく反映される)。
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])

        if start_dt:
            df = df[df["time"] >= start_dt]
        if end_dt:
            df = df[df["time"] <= end_dt]
        if len(df) < 10:
            logging.warning(f"Too little data for {symbol} in the given range. skipping")
            continue
        df = df.sort_values("time").reset_index(drop=True)

        ledger_local_path, ledger_drive_path = get_ledger_paths(symbol, args.mode)
        restore_ledger_from_drive_if_missing(ledger_local_path, ledger_drive_path, symbol)

        if os.path.exists(ledger_local_path):
            ledger_df = pd.read_csv(ledger_local_path)
            ledger_df["time"] = pd.to_datetime(ledger_df["time"])
        else:
            ledger_df = pd.DataFrame(columns=["time", "is_valid"])

        known_times = set(ledger_df["time"])
        new_rows_df = df[~df["time"].isin(known_times)]

        if len(new_rows_df) > 0:
            logging.info(
                f"📦 [{symbol}] 画像検証キャッシュ: 既知{len(ledger_df)}件 / 新規{len(new_rows_df)}件を検証します"
                f"({LEDGER_CHECKPOINT_CHUNK_SIZE}件ごとに台帳を保存し、途中で止まっても手戻りを抑えます)"
            )
            # 💡 追加(2026-11-07の改修): 大量件数(例: 1銘柄10万件規模)の初回検証は
            # 時間がかかるため、LEDGER_CHECKPOINT_CHUNK_SIZE件ごとに台帳をディスク+
            # Driveへ保存するようにする。Colabのセッション切断等が途中で起きても、
            # 直近のチャンクまでの検証結果は失われない。
            new_times_list = new_rows_df["time"].tolist()
            for chunk_start in range(0, len(new_times_list), LEDGER_CHECKPOINT_CHUNK_SIZE):
                chunk_times = new_times_list[chunk_start:chunk_start + LEDGER_CHECKPOINT_CHUNK_SIZE]
                chunk_valid_mask = validate_images_with_progress(chunk_times, current_img_dir, symbol)
                chunk_ledger_rows = pd.DataFrame({"time": chunk_times, "is_valid": chunk_valid_mask})
                ledger_df = pd.concat([ledger_df, chunk_ledger_rows], ignore_index=True)
                ledger_df = ledger_df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
                ledger_df.to_csv(ledger_local_path, index=False)
                if drive_service:
                    upload_to_drive_with_timeout(drive_service, ledger_local_path)
                logging.info(
                    f"💾 [{symbol}] 台帳チェックポイント保存: "
                    f"{min(chunk_start + LEDGER_CHECKPOINT_CHUNK_SIZE, len(new_times_list))}/{len(new_times_list)}件まで完了"
                )
        else:
            logging.info(f"📦 [{symbol}] 新規行なし。画像検証キャッシュ{len(ledger_df)}件をそのまま使用します")

        merged = df.merge(ledger_df, on="time", how="left")
        missing_count = int((~merged["is_valid"].fillna(False)).sum())
        if missing_count > 0:
            logging.warning(f"⚠️ {symbol} にて {missing_count} 件の画像欠損または破損を検出。")
        else:
            logging.info(f"✅ {symbol} の画像はすべて揃っており、正常です。")
        df = merged[merged["is_valid"].fillna(False)].drop(columns=["is_valid"]).copy()

        if first_train_df is None:
            first_train_df = df

        n = len(df)
        train_idx = int(n * 0.70)
        val_idx = train_idx + int(n * 0.20)

        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()

        risk_mean = train_df["target_risk_pct"].mean()
        risk_std = train_df["target_risk_pct"].std() + 1e-8
        scaler_info[symbol] = {"mean": risk_mean, "std": risk_std}

        test_df.to_csv(test_csv_path, index=False)
        logging.info(f"Saved test data for {symbol}: {len(test_df)} rows -> {test_csv_path}")

        tmp_train_csv_list.append(tmp_train_csv)
        train_df.to_csv(tmp_train_csv, index=False)
        val_df.to_csv(tmp_val_csv, index=False)

        tf_map = {"short":"M5","medium":"M15","long":"H4"}
        base_tf = tf_map.get(args.mode,"M15")
        current_img_dir = os.path.join("images",symbol,f"{args.mode}_MTF")
        try:
            
            raw_tr_ds = FXMultimodalDataset(tmp_train_csv, current_img_dir, symbol=symbol, seq_length=SEQ_LENGTH)
            raw_val_ds = FXMultimodalDataset(tmp_val_csv, current_img_dir, symbol=symbol, seq_length=SEQ_LENGTH)

            if len(raw_tr_ds) > 0:
                train_datasets.append(ScaledFXDataset(raw_tr_ds, risk_mean, risk_std))
                loaded_symbols.append(symbol)
            if len(raw_val_ds) > 0:
                val_datasets.append(ScaledFXDataset(raw_val_ds, risk_mean, risk_std))

            logging.info(f"Loaded {symbol}: Train={len(raw_tr_ds)}, Val={len(raw_val_ds)}, Test={len(test_df)}")
        except Exception as e:
            logging.error(f"Error loading {symbol}: {e}")

    if not train_datasets:
        logging.error("No valid data loaded. Check date range of image paths.")
        return

    # 💡 追加(2026-10-23の検証で判明した問題への対応): CSVはあったが画像の問題等で
    # 結局使えなかった銘柄が無いかを確認する。1銘柄でも欠けると、その銘柄を全く
    # 見ないまま学習が完走してしまうため、黙って続行せず明示的に中止する。
    missing_after_load = [s for s in EXPECTED_SYMBOLS if s not in loaded_symbols]
    if missing_after_load:
        logging.error(
            f"❌ 以下の銘柄はCSVはあったものの、有効な学習データを1件も読み込めなかったため、"
            f"学習を中止します: {missing_after_load}(画像フォルダの有無・内容を確認してください)"
        )
        # 💡 修正(2026-10-30): 同上の理由でsys.exit(1)に変更。
        sys.exit(1)
    
    
    with open(f"saved_models/risk_scalers_{args.mode}.json", "w") as f:
        json.dump(scaler_info, f, indent=4)
    
    full_train_ds = ConcatDataset(train_datasets)
    full_val_ds = ConcatDataset(val_datasets)

    train_loader = DataLoader(
                full_train_ds, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4,
                collate_fn=safe_collate_fn
            )
    val_loader = DataLoader(
        full_val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=safe_collate_fn
    )
    # モデル・学習設定
    # first_train_df を使って input_dim を設定
    input_dim = first_train_df.shape[1] - 5
    model = MultimodalFXmodel(num_tabular_features=input_dim).to(device)
    log_var_c = torch.zeros(1, requires_grad=True, device=device)
    log_var_r = torch.zeros(1, requires_grad=True, device=device)
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{args.mode}.pth") # .pth に統一
    optimizer = optim.Adam(list(model.parameters()) + [log_var_c, log_var_r], lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch = 0
    checkpoint = None
    renewed = 0

    best_val_total_loss = float('inf')
    best_val_class_loss = float("inf")
    best_val_risk_loss = float("inf")

    drive_model_dir = "/content/drive/MyDrive/FX_AI_Models"
    drive_model_path = os.path.join(drive_model_dir,f"model_{args.mode}.pth")

    if not os.path.exists(model_path) and os.path.exists(drive_model_path):
        logging.info(f"Model not found at local drive. Load from Google Drive;{drive_model_path}")
        os.makedirs(model_dir,exist_ok = True)
        shutil.copy(drive_model_path,model_path)
    if os.path.exists(model_path):
        logging.info(f"[*] model_{args.mode} found")
        try:
            checkpoint = torch.load(model_path,map_location=device,weights_only = False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'log_var_c' in checkpoint and 'log_var_r' in checkpoint:
                    with torch.no_grad():
                        log_var_c.copy_(checkpoint['log_var_c'].to(device))
                        log_var_r.copy_(checkpoint['log_var_r'].to(device))
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_total_loss = checkpoint.get('best_val_total_loss', float('inf'))
                best_val_class_loss = checkpoint.get("best_val_class_loss",float("inf"))
                best_val_risk_loss = checkpoint.get("best_val_risk_loss",float("inf"))
                renewed = checkpoint.get("renewed",0)
                logging.info(f"✅ オプティマイザとスケジューラを復元しました。エポック {start_epoch} から再開します。")
            else:
                # 従来の重みのみのファイルだった場合
                model.load_state_dict(checkpoint)
                logging.info("ℹ️ 重みデータのみを読み込みました（オプティマイザ等は初期化されます）")
        except Exception as e:
            logging.info(f"Errror occured for loading moddel{e}")
    else:
        logging.info(f"model_{args.mode} not found")
        logging.info("create new model")
        
    auto_weight = calculate_auto_class_weights(tmp_train_csv_list,method = "sqrt",device = device)
    criterion_class = nn.CrossEntropyLoss(weight = auto_weight)
    criterion_risk = nn.SmoothL1Loss()
    #criterion_risk = nn.MSELoss()
    

    
    logging.info(f"Start Training: {args.mode} | Start: {args.start} | End: {args.end}")
    
    

    

    patience = 5
    
    # 💡 追加(2026-11-07の改修): train_loader・val_loaderそれぞれのバッチ数をログに
    # 残しておく。「今何バッチ目か」の進捗ログ(下記BATCH_LOG_INTERVAL)と合わせて見ることで、
    # 現在どの程度進んでいるかが分かるようにする。
    BATCH_LOG_INTERVAL = 50
    logging.info(f"📊 1エポックあたりのバッチ数: Train={len(train_loader)}, Val={len(val_loader)}")

    for epoch in range(start_epoch,args.epochs):
        logging.info(f"▶️ エポック{epoch+1}/{args.epochs}: 学習ループを開始します")
        model.train()
        train_loss = 0
        train_class_loss = 0
        train_risk_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(train_loader,desc = f"Epoch {epoch+1}")):
            if batch_data is None:
                continue
            imgs,tabs,cls,risks,symbol_ids = batch_data
            imgs, tabs, symbol_ids = imgs.to(device), tabs.to(device), symbol_ids.to(device)
            cls = (cls + 1).to(device).long() # -1,0,1 -> 0,1,2(SELL,HOLD,BUY)
            risks = risks.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            out_class, out_risk = model(imgs, tabs, symbol_ids)

            loss_c = criterion_class(out_class, cls)
            loss_r = criterion_risk(out_risk, risks)
            loss = 0.5 * torch.exp(-log_var_c) * loss_c + 0.5 * log_var_c + \
       0.5 * torch.exp(-log_var_r) * loss_r + 0.5 * log_var_r
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_class_loss += loss_c.item()
            train_risk_loss += loss_r.item()

            # 💡 追加: tqdmの進捗バーは標準出力のみでログファイルには残らないため、
            # ログファイル側にも一定間隔でバッチ進捗を残す(コスト無視できる頻度)。
            if (batch_idx + 1) % BATCH_LOG_INTERVAL == 0:
                logging.info(f"  [Train] バッチ {batch_idx+1}/{len(train_loader)} (loss={loss.item():.6f})")

        avg_train_total = train_loss / len(train_loader)
        avg_train_class = train_class_loss / len(train_loader)
        avg_train_risk = train_risk_loss / len(train_loader)

        # バリデーション
        logging.info(f"▶️ エポック{epoch+1}/{args.epochs}: 検証(validation)ループを開始します")
        model.eval()
        val_total_loss = 0
        val_class_loss = 0
        val_risk_loss = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                if batch_data is None:
                    continue
                imgs,tabs,cls,risks,symbol_ids = batch_data
                imgs, tabs, symbol_ids = imgs.to(device), tabs.to(device), symbol_ids.to(device)
                cls = (cls + 1).to(device).long()
                risks = risks.to(device).float().view(-1, 1)
                oc, ork = model(imgs, tabs, symbol_ids)

                loss_c = criterion_class(oc, cls)
                loss_r = criterion_risk(ork, risks)
                loss = 0.5 * torch.exp(-log_var_c) * loss_c + 0.5 * log_var_c + \
       0.5 * torch.exp(-log_var_r) * loss_r + 0.5 * log_var_r

                val_total_loss += loss.item()
                val_class_loss += loss_c.item()
                val_risk_loss += loss_r.item()

                if (batch_idx + 1) % BATCH_LOG_INTERVAL == 0:
                    logging.info(f"  [Val] バッチ {batch_idx+1}/{len(val_loader)}")
        
        avg_val_total = val_total_loss / len(val_loader)
        avg_val_class = val_class_loss / len(val_loader)
        avg_val_risk = val_risk_loss / len(val_loader)
        
        current_weight_c = torch.exp(-log_var_c).item()
        current_weight_r = torch.exp(-log_var_r).item()
        logging.info(f"🔄 [タスク間重み比率] 分類(Class): {current_weight_c:.4f} | 回帰(Risk): {current_weight_r:.4f}")
        logging.info(f"Epoch {epoch+1}")
        logging.info(f"  [Train] Total: {avg_train_total:.6f} (Class: {avg_train_class:.6f}, Risk: {avg_train_risk:.6f})")
        logging.info(f"  [Val]   Total: {avg_val_total:.6f} (Class: {avg_val_class:.6f}, Risk: {avg_val_risk:.6f})")
        
        scheduler.step(avg_val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"[LR] Current Learning Rate:{current_lr}")
        
        checkpoint_data = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'log_var_c': log_var_c.detach().cpu(), 
            'log_var_r': log_var_r.detach().cpu(),
            'best_val_total_loss': best_val_total_loss,
            'best_val_class_loss':best_val_class_loss,
            'best_val_risk_loss':best_val_risk_loss,
            'renewed':renewed,
            'args': args

        }
        renewed += 1

        logging.info(f"💾 エポック{epoch+1}/{args.epochs}: チェックポイント保存処理を開始します")

        if avg_val_total < best_val_total_loss:
            best_val_total_loss = avg_val_total
            renewed = 0
            path = os.path.join(model_dir, f"best_total_model_{args.mode}.pth")
            torch.save(checkpoint_data, path)
            logging.info(f"✨ Saved Best TOTAL model: {path}")


        if avg_val_class < best_val_class_loss:
            best_val_class_loss = avg_val_class
            renewed = 0
            path = os.path.join(model_dir, f"best_class_model_{args.mode}.pth")
            torch.save(checkpoint_data,path)
            logging.info(f"✨ Saved Best CLASS model: {path}")


        if avg_val_risk < best_val_risk_loss:
            best_val_risk_loss = avg_val_risk
            renewed = 0
            path = os.path.join(model_dir, f"best_risk_model_{args.mode}.pth")
            torch.save(checkpoint_data, path)
            logging.info(f"✨ Saved Best RISK model: {path}")

        # 💡 修正(2026-11-07の改修): 毎エポックget_drive_service()を呼び直すのをやめ、
        # main()冒頭で一度だけ取得したdrive_serviceを再利用する(不要な認証処理の
        # 削減)。アップロードもupload_to_drive_with_timeout()経由にし、開始・終了の
        # ログとタイムアウトを明示的に付与する(「Driveへの転送でタイムアウトが
        # 発生するが、どこで止まっているか分からない」問題への対応)。
        if drive_service:
            logging.info(f"☁️ エポック{epoch+1}/{args.epochs}: Driveへのモデルアップロードを開始します(4ファイル)")
            models_to_upload = [
                os.path.join(model_dir, f"best_total_model_{args.mode}.pth"),
                os.path.join(model_dir, f"best_class_model_{args.mode}.pth"),
                os.path.join(model_dir, f"best_risk_model_{args.mode}.pth"),
                os.path.join(model_dir, f"model_{args.mode}.pth")
            ]
            for m_path in models_to_upload:
                upload_to_drive_with_timeout(drive_service, m_path)
            logging.info(f"☁️ エポック{epoch+1}/{args.epochs}: Driveへのモデルアップロードが完了しました")

        if renewed == patience:
            logging.info(f"Early stop triggered")
            break

        latest_path = os.path.join(model_dir, f"model_{args.mode}.pth")
        torch.save(checkpoint_data, latest_path)
        logging.info(f"💾 エポック {epoch + 1} の状態を継続用として保存しました: {latest_path}")



    # クリーンアップ
    for f in glob.glob("temp_*_*.csv"):
        os.remove(f)

if __name__ == "__main__":
    main()
