import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import logging

from model import SYMBOL_TO_ID

# 💡 修正(2026-11-13の調査で判明した問題への対応): train.pyが
# `input_dim = first_train_df.shape[1] - 5` という固定値で表形式特徴量の次元数を
# 算出しており、この「5」は本来ここ(EXCLUDE_COLSの件数)と連動しているはずの
# 暗黙の前提だった。ラベル生成側の列数が変わった場合に連動して直すのを忘れると、
# 気づきにくい形でモデルの入力次元がズレる。インスタンス属性ではなくモジュール
# レベルの定数として公開し、train.py側がlen(EXCLUDE_COLS)を直接参照できるように
# することで、「5」という決め打ちの値そのものを無くす。
EXCLUDE_COLS = ["time", "future_close", "price_diff_pct", "target_class", "target_risk_pct"]

# 💡 追加(2026-11-18の調査で判明した問題への対応): 「HOLD行は直近何本以内に
# 売買シグナルがあれば学習対象に残す」という窓が16本の固定値だったため、銘柄に
# よって売買シグナルの時間的な密度が異なる(例: GOLDはシグナルが偏って発生し、
# 間隔が広い)場合、同じ16本という基準では銘柄ごとに実質的な意味が変わって
# しまっていた(GOLDだけHOLD行の採用率が他銘柄の6〜7割程度まで下がり、有効
# サンプル数が大きく減っていた)。create_labels.pyが分類閾値を銘柄自身の標準
# 偏差で自動調整しているのと同じ考え方で、固定の本数ではなく、その銘柄自身の
# HOLD行の「次のシグナルまでの距離」の分布から動的に窓を算出する。
# HOLD_RETENTION_PERCENTILEは「HOLD行の何%を学習対象として残すか」を決め、
# 具体的な本数は銘柄ごとのデータから自動的に決まる。
HOLD_RETENTION_PERCENTILE = 50


def compute_next_action_idx(target_classes):
    """
    target_classes(target_class列のnumpy配列)から、各行について「直後に来る
    売買シグナル(HOLD以外)のインデックス」を計算する(無ければ-1)。
    FXMultimodalDataset.__init__と検証スクリプト(verify_training_data.py)の
    両方で使う共通ロジック。
    """
    n = len(target_classes)
    next_action_idx = [-1] * n
    last_action = -1
    for i in range(n - 1, -1, -1):
        val = target_classes[i]
        if not pd.isna(val) and val != 0:
            last_action = i
        next_action_idx[i] = last_action
    return next_action_idx


def compute_dynamic_window(target_classes, next_action_idx, percentile=HOLD_RETENTION_PERCENTILE):
    """
    HOLD_RETENTION_PERCENTILE参照。HOLD行の「次のシグナルまでの距離」の分布
    から、銘柄ごとに動的な窓(本数)を算出する。戻り値: (window, HOLD行の件数)。
    """
    n = len(target_classes)
    hold_gaps = [
        next_action_idx[i] - i
        for i in range(n)
        if target_classes[i] == 0 and next_action_idx[i] != -1
    ]
    if hold_gaps:
        return int(np.percentile(hold_gaps, percentile)), len(hold_gaps)
    return 16, 0  # HOLD行に未来のシグナルが1件も無い(通常あり得ない)場合のフォールバック


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 💡 修正(2026-11-13の調査で判明した問題への対応): 以前はここで
# Google Drive(FUSEマウント)の一時的な接続切断(OSError: [Errno 107] Transport
# endpoint is not connected / [Errno 5] Input/output error)に対し、2秒→4秒の
# 間隔でリトリイしていた。しかし実機でDrive接続が広範囲に不安定になった際、
# リトリイの明示的な待ち時間より、読み込み自体が失敗と判定されるまでの時間
# (FUSE側の内部タイムアウト)が支配的で、リトリイするたびにこの長い待ちを
# 重ねてしまい、1エポックが大幅に遅延する事態になった。この読み込みは
# __getitem__のexceptで失敗時にそのサンプルを1エポックぶんスキップするだけ
# (検証台帳のように恒久的な判定として残るわけではなく、次エポックでまた
# 読み直しの機会がある)なので、粘らず即座に諦める方が実害が小さい。

class FXMultimodalDataset(Dataset):

    def __init__(self, csv_file, img_dir, symbol, transform=None, seq_length=40, symbol_to_id=None):
        """
        Custom dataset for multimodal AI (Time-Series Safe & Missing Data Tolerant Version)

        Args:
            symbol (str): このデータセットが対象とする銘柄名(例: "USDJPY")。
                model.SYMBOL_TO_ID を通じて銘柄埋め込み用のIDに変換される。
                学習側・ライブ側で対応がズレないよう、必ずこの唯一の対応表を使う。
            symbol_to_id (dict, optional): 省略時はmodel.SYMBOL_TO_ID(全7銘柄)を
                使う。特定の銘柄を除いて学習・評価する場合(model.get_active_symbol_map
                参照)は、呼び出し元がここに絞り込んだ対応表を渡す。
        """
        # 1. データの読み込み（⚠️ 行は絶対に削除しない。時系列の連続性を保つため）
        self.data = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.seq_length = seq_length
        self.symbol = symbol
        self.symbol_id = (symbol_to_id or SYMBOL_TO_ID)[symbol]

        if not os.path.exists(img_dir):
            raise FileNotFoundError(
                f"❌ [致命的エラー] 画像ディレクトリ自体が存在しません: {img_dir}"
                f"👉 先に `generate_images.py` を実行してMTF画像を生成してください。"
                )
        
        self.exclude_cols = EXCLUDE_COLS
        self.features_cols = [col for col in self.data.columns if col not in self.exclude_cols]

        self.data[self.features_cols] = self.data[self.features_cols].ffill().bfill().fillna(1)
       

        # 画像の前処理設定
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        self.valid_target_indices = []
        target_classes = self.data["target_class"].values # ilocを避けてnumpy配列で高速化
        n = len(self.data)

        # 後ろから順に「最後に見たHold(1)以外のアクションのインデックス」を記録
        next_action_idx = compute_next_action_idx(target_classes)

        self.dynamic_window, n_hold_gaps = compute_dynamic_window(target_classes, next_action_idx)
        logging.info(
            f"📏 [{symbol}] 動的ウィンドウ: HOLD行の{HOLD_RETENTION_PERCENTILE}パーセンタイル = "
            f"{self.dynamic_window}本(HOLD行{n_hold_gaps}件から算出)"
        )

        # 前から1回だけループして条件を判定
        for i in range(n):
            if i < self.seq_length - 1:
                continue
            if pd.isna(target_classes[i]):
                continue

            target_idx = next_action_idx[i]
            if target_idx != -1 and (target_idx - i) <= self.dynamic_window:
                self.valid_target_indices.append(i)

    def __len__(self):
        # データフレームの行数ではなく、見つかった「有効なターゲット数」を返す
        return len(self.valid_target_indices)
    
    def __getitem__(self, idx):
        # 1. 有効なリストから、ターゲットにする行番号を取得
        target_idx = self.valid_target_indices[idx]
        
        # 2. 過去15件分を切り出し
        start_idx = target_idx - self.seq_length + 1
        seq_data = self.data.iloc[start_idx : target_idx + 1]
        
        tabular_data = seq_data[self.features_cols].values.astype("float32")
        tabular_tensor = torch.tensor(tabular_data)

        # 3. 画像データの処理
        target_row = self.data.iloc[target_idx]
        time_str = pd.to_datetime(target_row["time"]).strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.img_dir, f"{time_str}.png")
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)

            # 4. ラベルデータのテンソル化
            class_label = torch.tensor(target_row["target_class"], dtype=torch.long)
            risk_label = torch.tensor(target_row["target_risk_pct"], dtype=torch.float32)

            # 5. 銘柄IDのテンソル化(B案: 銘柄埋め込み用)
            symbol_id_tensor = torch.tensor(self.symbol_id, dtype=torch.long)

            return image_tensor, tabular_tensor, class_label, risk_label, symbol_id_tensor
        except Exception as e:
            logging.warning(f"error:{e}")
            return None

if __name__ == "__main__":
    setup_logger()
    
    SAMPLE_CSV = "labeled_data/USDJPY_short_scaled.csv"
    SAMPLE_IMG_DIR = "images/USDJPY/short_MTF"

    if os.path.exists(SAMPLE_CSV) and os.path.exists(SAMPLE_IMG_DIR):
        dataset = FXMultimodalDataset(csv_file=SAMPLE_CSV, img_dir=SAMPLE_IMG_DIR, symbol="USDJPY", seq_length=15)
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
        images, tabulars, classes, risks, symbol_ids = next(iter(dataLoader))

        logging.info("=== data loader test successful ===")
        logging.info(f"image batch size: {images.shape}")
        logging.info(f"numerical value's size: {tabulars.shape}")
        logging.info(f"classifier label size: {classes.shape}")
        logging.info(f"regressor label size: {risks.shape}")
        logging.info(f"symbol id size: {symbol_ids.shape}")
    else:
        logging.warning("The specified files could not be found. Test skipped.")