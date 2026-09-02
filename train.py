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
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

def filter_valid_data(csv_path, img_dir):
    """
    画像ファイルが実際に存在するかを全件チェックし、
    画像がない行（欠損データ）を安全に除外した新しいCSVを生成する関数
    """
    df = pd.read_csv(csv_path)
    
    logging.info(f"⏳ 画像の欠損チェック中: {csv_path} ...")
    
    # 時刻から画像ファイル名をシミュレーションして、実際の存在をチェック
    time_strs = pd.to_datetime(df['time']).dt.strftime("%Y%m%d_%H%M%S")
    valid_mask = [os.path.exists(os.path.join(img_dir, f"{t}.png")) for t in time_strs]
    
    valid_count = sum(valid_mask)
    missing_count = len(df) - valid_count
    
    # 欠損が1枚もなければ、元のCSVパスをそのまま返す
    if missing_count == 0:
        logging.info("✅ 欠損画像なし。完全なデータセットです。")
        return csv_path
        
    # 欠損がある場合は、画像が存在する行（Trueの行）だけを抽出する
    logging.warning(f"⚠️ {missing_count}件の画像欠損を検出！数値とのズレを防ぐため、該当行をスキップした安全なCSVを生成します。")
    filtered_df = df[valid_mask]
    
    # 「_valid」と名前をつけた新しいCSVとして保存する
    valid_csv_path = csv_path.replace(".csv", "_valid.csv")
    filtered_df.to_csv(valid_csv_path, index=False)
    
    return valid_csv_path

def main():
    setup_logger()
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

    start_dt = pd.to_datetime(args.start) if args.start else None
    end_dt = pd.to_datetime(args.end) if args.end else None
    if end_dt and end_dt.hour == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    
    csv_files = glob.glob(f"labeled_data/*_{args.mode}_scaled.csv")
    if not csv_files:
        logging.error(f"No scaled data found for mode: {args.mode}")
        return
    
    train_datasets = []
    val_datasets = []
    scaler_info = {}
    tmp_train_csv_list = []
    img_dir = "images"
    
    
    os.makedirs("test_data", exist_ok=True)

    # DataFrameのカラム数を取得するために、最初の有効なデータセットから input_dim を計算する用
    first_train_df = None

    for csv_path in csv_files:
        symbol = os.path.basename(csv_path).split("_")[0]
        test_csv_path = f"test_data/{symbol}_{args.mode}_test.csv"
        tmp_train_csv = f"temp_train_{symbol}_{args.mode}.csv"
        tmp_val_csv = f"temp_val_{symbol}_{args.mode}.csv"
        current_img_dir = os.path.join("images", symbol, f"{args.mode}_MTF")
        
        if os.path.exists(tmp_train_csv) and os.path.exists(tmp_val_csv) and os.path.exists(test_csv_path):
            logging.info(f"✨ 既存の分割済みデータを読み込みます（画像整合性チェック・分割をスキップ）: {symbol}")
            train_df = pd.read_csv(tmp_train_csv)
            test_df = pd.read_csv(test_csv_path)
            if first_train_df is None:
                first_train_df = train_df
            risk_mean = float(train_df["target_risk_pct"].mean())
            risk_std = float(train_df["target_risk_pct"].std() + 1e-8)
            scaler_info[symbol] = {"mean": risk_mean, "std": risk_std}
            tmp_train_csv_list.append(tmp_train_csv)
        
        else:
            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"])

            if start_dt:
                df = df[df["time"] >= start_dt]
            if end_dt:
                df = df[df["time"] <= end_dt]
            if len(df) < 10:
                logging.warning(f"Too little data for {symbol} in the given range. skipping")
                continue
                
            valid_mask = []
            for t in df["time"]:
                img_path = os.path.join(current_img_dir, f"{t.strftime('%Y%m%d_%H%M%S')}.png")
                is_valid = False
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    try:
                        with Image.open(img_path) as img:
                            img.verify() 
                        is_valid = True
                    except Exception:
                        pass
                valid_mask.append(is_valid)
                
            missing_count = len(df) - sum(valid_mask)
            if missing_count > 0:
                logging.warning(f"⚠️ {symbol} にて {missing_count} 件の画像欠損または破損を検出。")
                df = df[valid_mask].copy() 
            else:
                logging.info(f"✅ {symbol} の画像はすべて揃っており、正常です。")
                
            df = df.sort_values("time")
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
            if len(raw_val_ds) > 0:
                val_datasets.append(ScaledFXDataset(raw_val_ds, risk_mean, risk_std))

            logging.info(f"Loaded {symbol}: Train={len(raw_tr_ds)}, Val={len(raw_val_ds)}, Test={len(test_df)}")
        except Exception as e:
            logging.error(f"Error loading {symbol}: {e}")
            
    if not train_datasets:
        logging.error("No valid data loaded. Check date range of image paths.")
        return
    
    
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
    
    for epoch in range(start_epoch,args.epochs):
        model.train()
        train_loss = 0
        train_class_loss = 0
        train_risk_loss = 0
        for batch_data in tqdm(train_loader,desc = f"Epoch {epoch+1}"):
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
        
        avg_train_total = train_loss / len(train_loader)
        avg_train_class = train_class_loss / len(train_loader)
        avg_train_risk = train_risk_loss / len(train_loader)

        # バリデーション
        model.eval()
        val_total_loss = 0
        val_class_loss = 0
        val_risk_loss = 0

        with torch.no_grad():
            for batch_data in val_loader:
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
        
        drive_service = get_drive_service()
        if drive_service:
            models_to_upload = [
                os.path.join(model_dir, f"best_total_model_{args.mode}.pth"),
                os.path.join(model_dir, f"best_class_model_{args.mode}.pth"),
                os.path.join(model_dir, f"best_risk_model_{args.mode}.pth"),
                os.path.join(model_dir, f"model_{args.mode}.pth")
            ]
            for m_path in models_to_upload:
                upload_to_drive(drive_service, m_path)
                logging.info(f"m_path:{m_path},drive_service:{drive_service}")

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