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
        logging.info("portfolio_dataset.py ver208.2")
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
            cols = []
            for c in self.df.columns:
                if c.startswith(f"{sym}_"):
                    # プレフィックス(例: "USDJPY_")を除去して完全一致判定を行う
                    base_col_name = c[len(f"{sym}_"):]
                    if base_col_name not in exclude_exact:
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
            except Exception:
                # 【重要】特定の銘柄だけ休場等で画像がない場合、エラーで止めず黒画像（ゼロテンソル）を返す
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

        