import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import glob
import logging
import argparse
import json
import shutil
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score
from torch.utils.data._utils.collate import default_collate

import torch.nn.functional as F
from model import MultimodalFXmodel
from dataset import FXMultimodalDataset

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ==========================================
# 🌟 破損・欠損データをバッチから安全に除外するフィルター
# ==========================================
def safe_collate_fn(batch):
    """
    バッチの中から None（読み込み失敗データ）を安全に取り除くフィルター
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)

def evaluate_single_model(model, loaders, device, risk_scalers):
    """
    特定のモデル重みでデータセットを走査し、評価指標を返す補助関数
    """
    all_cls_true, all_cls_pred = [], []
    all_risk_true, all_risk_pred = [], []
    
    model.eval()
    with torch.no_grad():
        for loader, pair_name in loaders:
            # 万が一スケーラー情報がない場合は安全のため平均0, 標準偏差1とする
            s_mean = risk_scalers.get(pair_name, {"mean": 0.0})["mean"]
            s_std = risk_scalers.get(pair_name, {"std": 1.0})["std"]
            
            for batch_data in loader:
                if batch_data is None:
                    continue  # 空のバッチ（全滅ケース）は安全にスキップ
                
                # 正常なデータのみ展開
                imgs, tabs, cls, risks = batch_data
                
                imgs, tabs = imgs.to(device), tabs.to(device)
                oc, ork = model(imgs, tabs)
                
                # 分類評価
                probs = F.softmax(oc, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # 評価時は元の target_class の値 (-1, 0, 1) に戻す
                all_cls_true.extend((cls).cpu().numpy().tolist())
                all_cls_pred.extend((preds - 1).cpu().numpy().tolist())
                
                # リスク評価 (スケーリングを元に戻す：逆変換)
                pred_risk_scaled = ork.cpu().numpy().flatten()
                pred_risk_original = pred_risk_scaled * s_std + s_mean
                
                true_risk_original = risks.cpu().numpy().flatten()
                
                all_risk_true.extend(true_risk_original.tolist())
                all_risk_pred.extend(pred_risk_original.tolist())
                
    if len(all_cls_true) == 0:
        return 0.0, 999.0, ([], [], [], [])
        
    acc = accuracy_score(all_cls_true, all_cls_pred)
    mae = mean_absolute_error(all_risk_true, all_risk_pred)
    
    return acc, mae, (all_cls_true, all_cls_pred, all_risk_true, all_risk_pred)

def main():
    parser = argparse.ArgumentParser(description="訓練済みマルチモーダルFXモデルの評価プログラム")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"], help="評価対象のモード")
    args = parser.parse_args()

    setup_logger()
    logging.info(f"--- 📊 マルチモーダルFXモデルの評価開始 (モード: {args.mode}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. 🌟 train.py の仕様に合わせてリスクスケーラー情報ファイル名を修正
    risk_scaler_path = f"saved_models/risk_scalers_{args.mode}.json"
    if not os.path.exists(risk_scaler_path):
        logging.error(f"❌ リスクスケーラー情報が見つかりません: {risk_scaler_path}")
        return
        
    with open(risk_scaler_path, "r", encoding="utf-8") as f:
        risk_scalers = json.load(f)

    # 2. 🌟 評価データの準備（test_dataフォルダ配下の専用テストCSVを使用）
    test_files = glob.glob(f"test_data/*_{args.mode}_test.csv")
    if not test_files:
        logging.error(f"❌ {args.mode} モードの評価用テストデータ(test_data/*_test.csv)が見つかりません。")
        return

    test_loaders = []
    for csv_file in test_files:
        filename = os.path.basename(csv_file)
        pair_name = filename.split("_")[0] # 例: USDJPY
        
        img_dir = f"images/{pair_name}/{args.mode}_MTF"
        
        try:
            # データセットの作成
            dataset = FXMultimodalDataset(csv_file=csv_file, img_dir=img_dir, seq_length=40)
            
            loader = DataLoader(
                dataset, 
                batch_size=64, 
                shuffle=False, 
                num_workers=4,
                collate_fn=safe_collate_fn
            )
            test_loaders.append((loader, pair_name))
            logging.info(f"✅ テストデータ登録完了: {pair_name} (データ行数: {len(dataset)})")
        except FileNotFoundError as e:
            logging.warning(f"⚠️ スキップ: {pair_name} の評価データを登録できませんでした。原因: {e}")
            continue

    if not test_loaders:
        logging.error("❌ 評価可能なデータローダーが1つも存在しません。終了します。")
        return

    # 3. モデルの構造定義と各種重みの評価
    first_dataset = test_loaders[0][0].dataset
    input_dim = len(first_dataset.features_cols)
    logging.info(f"📊 検出された数値データの入力次元数: {input_dim}")

    model = MultimodalFXmodel(num_tabular_features=input_dim).to(device)
    model_dir = "saved_model"
    
    model_variants = {
        "Best TOTAL Model (Class + Risk balanced)": f"best_total_model_{args.mode}.pth",
        "Best CLASS Model (Accuracy optimized)": f"best_class_model_{args.mode}.pth",
        "Best RISK Model (MAE optimized)": f"best_risk_model_{args.mode}.pth",
        "Final Epoch Model": f"model_{args.mode}.pth"
    }

    best_overall_acc = -1.0
    final_data = None
    best_variant_name = ""

    for name, filename in model_variants.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            logging.warning(f"⚠️ モデル重みファイルが見つかりません。スキップします: {path}")
            continue
            
        logging.info(f"\n🔍 評価中... [{name}]")
        checkpoint = torch.load(path, map_location=device,weights_only = False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        acc, mae, data_tuple = evaluate_single_model(model, test_loaders, device, risk_scalers)
        
        logging.info(f" ➡️ 結果 - Accuracy: {acc:.4f} | Risk MAE: {mae:.6f}%")
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            final_data = data_tuple
            best_variant_name = name

    if final_data is None:
        logging.error("❌ 評価を実行できるモデルの重みが1つも見つかりませんでした。")
        return

    # 4. 最も成績が良かったモデルを確定し、Google Driveへバックアップ
    logging.info("\n" + "🏆" * 30)
    logging.info(f" 最終選定モデル: {best_variant_name} (最高 Accuracy: {best_overall_acc:.4f})")
    
    best_model_src = os.path.join(model_dir, model_variants[best_variant_name])
    drive_model_dir = "/content/drive/MyDrive/FX_AI_Models"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(drive_model_dir, exist_ok=True)
        drive_best_model_dest = os.path.join(drive_model_dir, f"final_best_model_{args.mode}.pth")
        shutil.copy(best_model_src, drive_best_model_dest)
        logging.info(f"Final best model is saved to Google Drive: {drive_best_model_dest}")
    logging.info("🏆" * 30)

    # 最終的な詳細レポート出力
    t_cls, p_cls, t_risk, p_risk = final_data
    target_names = ['SELL (-1)', 'HOLD (0)', 'BUY (1)']
    
    logging.info("\n--- Final Classification Report ---")
    print(classification_report(t_cls, p_cls, labels=[-1, 0, 1], target_names=target_names, zero_division=0))

    logging.info("--- Final Confusion Matrix ---")
    cm = confusion_matrix(t_cls, p_cls, labels=[-1, 0, 1])
    print(f"      Pred: SELL  HOLD  BUY")
    for i, true_label in enumerate(['SELL', 'HOLD', 'BUY']):
        row_str = " ".join([f"{val:<5}" for val in cm[i]])
        print(f"True: {true_label:<4}  {row_str}")
    logging.info(f"--- Final Risk Prediction MAE: {mean_absolute_error(t_risk, p_risk):.6f}% ---")

if __name__ == "__main__":
    main()