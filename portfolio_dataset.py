import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import logging


class PortfolioFXDataset(Dataset):
    def __init__(self,merged_csv_path,base_img_dir,symbols,mode = "short",seq_length = 40):
        logging.info("portfolio_dataset.py ver209.0")
        self.df = pd.read_csv(merged_csv_path)
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # シーケンス長に満たない初期ステップは None を返す（環境側でスキップ処理する）
        if idx < self.seq_length - 1:
            return None
        
        imgs_list = []
        tabs_list = []
        
        time_str = self.df.iloc[idx]["time_str"]
        
        for sym in self.symbols:
            # 1. 画像データの読み込み
            img_path = os.path.join(self.base_img_dir, sym, f"{self.mode}_MTF", f"{time_str}.png")
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception as e:
                # 💡 修正: 黒画像(ゼロテンソル)へのフォールバック自体は維持する(エラーで
                # 処理全体を止めないため)が、無言のまま進むと precompute_portfolio.py の
                # 出力に気づかれない汚染を生む(2026-10-23の検証で判明)。最低限ログに残す。
                # 本来の防御線は precompute_portfolio.py 側の事前完全性チェック
                # (check_images_complete)であり、ここに到達すること自体が想定外。
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

        
