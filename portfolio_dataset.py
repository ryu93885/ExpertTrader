import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import errno
import time
from PIL import Image
import torchvision.transforms as transforms
import logging

# 💡 追加(2026-10-30の調査で判明した問題への対応): Google Drive(FUSEマウント)への
# アデセスが集中・長時間化すると、接続自体が切断されることがある(OSError: [Errno 107]
# Transport endpoint is not connected / [Errno 5] Input/output error)。
# 一時的な切断はリトリイで吸収する。
DRIVE_IO_MAX_RETRIES = 5
DRIVE_IO_RETRY_BACKOFF_BASE = 10  # 秒。リトリイ毎に倍増


def _with_drive_retry(func, *args, **kwargs):
    """
    💡 追加(2026-10-30/31の調査で判明した問題への対応): Google Drive(FUSEマウント)上の
    ファイルを読み書きする呼び出し(pd.read_csv・os.scandir等)を、一時的な接続切断
    (OSError: [Errno 107] Transport endpoint is not connected / [Errno 5] Input/output
    error)に対してリトリイする汎用リーパー。それ以外のエラーは即座に呼び出し元に伝える。
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


def _list_dir_names(path):
    """
    💡 修正(2026-11-03の調査で判明した問題への対応): 以前は本関数がos.scandir()で
    DirEntryを返し、呼び出し元が各ファイルごとにentry.stat()を呼んでサイズ(>0)まで
    検証していた。しかしmediumモードの全期間では90,763ステップ×7銘柄=約64万回の
    stat呼び出しが発生し、このチェック自体(約37分)がGoogle Drive側のレート制限を
    悪化させ、直後に始まる本体の推論ループ(同じDriveマウント上で約64万回の画像読み込みを
    行う)まで連鎖的に遅くなる事例が発生した(「数時間」で終わっていた処理が「60時間」に
    悪化)。ファイル名の一覧取得(os.listdir、ディレクトリ1回の読み取りのみ)だけに留め、
    個々のファイルへのサイズ確認は行わない。ゼロバイト等の破損ファイルの検出は失うが、
    実際に読み込む際(__getitem__のImage.open)で例外として捕捉され黒画像にフォールバック
    するため、安全性は維持される。
    """
    return set(os.listdir(path))


class PortfolioFXDataset(Dataset):
    def __init__(self,merged_csv_path,base_img_dir,symbols,mode = "short",seq_length = 40):
        logging.info("portfolio_dataset.py ver212.0")
        self.df = _with_drive_retry(pd.read_csv, merged_csv_path)
        self.base_img_dir = base_img_dir
        self.symbols = symbols
        self.mode = mode
        self.seq_length = seq_length

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        # 銘柄ごとの特徴量カラムを抽出（timeやtargetなどの不要な列を除外）
        self.feature_cols = {}
        self.feature_cols = {}
        for sym in self.symbols:
            exclude_exact = ["time", "future_close", "price_diff_pct", "target_class", "target_risk_pct"]
            # merge_portfolio_data.py が付与する生値列（"_raw"）と、precompute_portfolio.py が
            # 付与する事前推論列（prob_0/1/2, risk_val）は、モデル学習時の特徴量には含まれていない
            # ため、_scaled.csv 由来の特徴量列とスケールが混在しないよう除外する。
            exclude_suffixes_exact = ["prob_0", "prob_1", "prob_2", "risk_val"]
            cols = []
            for c in self.df.columns:
                if c.startswith(f"{sym}_"):
                    # プレフィックス(例: "USDJPY_")を除去して完全一致判定を行う
                    base_col_name = c[len(f"{sym}_"):]
                    if base_col_name in exclude_exact:
                        continue
                    if base_col_name.endswith("_raw"):
                        continue
                    if base_col_name in exclude_suffixes_exact:
                        continue
                    cols.append(c)
            self.feature_cols[sym] = cols
            
        # 画像読み込み用に、time列を文字列フォーマットに変換しておく
        self.df["time_str"] = pd.to_datetime(self.df["time"]).dt.strftime("%Y%m%d_%H%M%S")
        logging.info(f"Portfolio Dataset 初期化完了: {len(self.symbols)}銘柄, 総ステップ数: {len(self.df)}")

        # 💡 追加(2026-10-31の調査で判明した問題への対応): 銘柄ごとの取引カレンダーの違い
        # 等により、一部の時刻でその銘柄の画像が存在しない行が生じる(merge_portfolio_data.py
        # のマスタータイムラインは全銘柄の時刻の和集合のため)。従来は黒画像(ゼロテンソル)で
        # 代替していたが、これは情報をゼロにしてしまう。表データ側は同じ種類の欠損を既に
        # ffill(直前値で埋める)で扱っているため、画像側も同じ考え方に揃え、直前(無ければ
        # 直後)の実在する画像を使い回す。__init__時点で銘柄ごとに1回だけフォルダを列挙し、
        # 全行分の解決済みパスを事前計算する(precompute_portfolio.py側の完全性チェックとも
        # この結果を共有し、Driveへの再スキャンを避ける)。
        self._resolve_image_paths()

    def _resolve_image_paths(self):
        self.resolved_img_paths = {}
        self.missing_image_info = {}

        time_strs = self.df["time_str"].values

        for sym in self.symbols:
            folder = os.path.join(self.base_img_dir, sym, f"{self.mode}_MTF")
            try:
                existing_names = _with_drive_retry(_list_dir_names, folder)
            except OSError as e:
                logging.error(f"❌ 画像フォルダにアクセスできません: {folder} ({e})")
                existing_names = set()

            paths = [None] * len(time_strs)
            was_real = [False] * len(time_strs)
            missing_idx = []

            for i, time_str in enumerate(time_strs):
                filename = f"{time_str}.png"
                if filename in existing_names:
                    paths[i] = os.path.join(folder, filename)
                    was_real[i] = True
                else:
                    missing_idx.append(i)

            # 前方補完(ffill): 直前の「実在する」画像のパスで埋める
            last_valid = None
            for i in range(len(paths)):
                if was_real[i]:
                    last_valid = paths[i]
                elif last_valid is not None:
                    paths[i] = last_valid

            # 後方補完(bfill): 先頭近くで前方補完すら効かなかった行を、直後の
            # 「実在する」画像のパスで埋める(was_realは元の実在判定のまま固定なので、
            # ffillで書き込まれた値を誤ってアンカーにすることはない)
            next_valid = None
            for i in range(len(paths) - 1, -1, -1):
                if was_real[i]:
                    next_valid = paths[i]
                elif paths[i] is None and next_valid is not None:
                    paths[i] = next_valid

            self.resolved_img_paths[sym] = paths
            self.missing_image_info[sym] = {
                "missing_count": len(missing_idx),
                "total": len(paths),
                "examples": [
                    os.path.join(folder, f"{time_strs[i]}.png") for i in missing_idx[:5]
                ],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # シーケンス長に満たない初期ステップは None を返す（環境側でスキップ処理する）
        if idx < self.seq_length - 1:
            return None
        
        imgs_list = []
        tabs_list = []
        
        for sym in self.symbols:
            # 1. 画像データの読み込み(__init__で解決済みのパスを使う。詳細は
            # _resolve_image_paths参照。直前/直後の実在画像で既に補完されているため、
            # 通常はこのimg_pathがNoneになることはない)
            img_path = self.resolved_img_paths[sym][idx]
            try:
                if img_path is None:
                    raise FileNotFoundError("この銘柄には有効な画像が1件も存在しません")
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception as e:
                # 💡 ここに到達するのは、銘柄に実在する画像が1件も無い(極端な異常事態)
                # か、解決済みのはずのパスへの読み込みがDrive側の一時的な問題等で
            # 失敗した場合のみ。本来の欠損補完は_resolve_image_pathsの前方/後方
            # 補完で既に完了しているため、ここは最後の防御線として黒画像にする。
                logging.warning(f"⚠️ 画像の読み込みに失敗したため黒画像で代替します: {img_path} ({e})")
                img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
            # 2. テーブル（時系列）データの読み込み
            cols = self.feature_cols[sym]
            seq_df = self.df.iloc[idx - self.seq_length + 1 : idx + 1][cols]
            tab_tensor = torch.tensor(seq_df.values, dtype=torch.float32)
            
            imgs_list.append(img_tensor)
            tabs_list.append(tab_tensor)
        
        # 3. 7銘柄分をスタックして、バッチサイズ7のテンソルを作成
        # imgs_tensor shape: (7, 3, 224, 224)
        # tabs_tensor shape: (7, seq_length, num_features)
        imgs_tensor = torch.stack(imgs_list)
        tabs_tensor = torch.stack(tabs_list)
        
        return imgs_tensor, tabs_tensor
