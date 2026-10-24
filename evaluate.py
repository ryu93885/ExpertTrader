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


def get_checkpoint_input_dim(checkpoint):
    """
    チェックポイントのGRU重み（gru.weight_ih_l0）の形状から、
    そのモデルが実際に学習された入力次元数を直接読み取る。
    test_data/*.csv 側の列数を鵜呑みにすると、train.py のキャッシュ機構により
    古い分割データが残っていた場合に気づけないため、モデル本体を正とする。
    """
    try:
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        weight = state_dict.get("gru.weight_ih_l0")
        return int(weight.shape[1]) if weight is not None else None
    except Exception:
        return None


def evaluate_single_model(model, loaders, device, risk_scalers):
    """
    特定のモデル重みでデータセットを走査し、銘柄別・全体(プール)の評価データを返す。
    """
    per_symbol = {}  # pair_name -> {"cls_true":[...], "cls_pred":[...], "risk_true":[...], "risk_pred":[...]}
    all_cls_true, all_cls_pred = [], []
    all_risk_true, all_risk_pred = [], []

    model.eval()
    with torch.no_grad():
        for loader, pair_name in loaders:
            # 万が一スケーラー情報がない場合は安全のため平均0, 標準偏差1とする
            s_mean = risk_scalers.get(pair_name, {"mean": 0.0})["mean"]
            s_std = risk_scalers.get(pair_name, {"std": 1.0})["std"]

            sym_data = per_symbol.setdefault(
                pair_name, {"cls_true": [], "cls_pred": [], "risk_true": [], "risk_pred": []}
            )

            for batch_data in loader:
                if batch_data is None:
                    continue  # 空のバッチ（全滅ケース）は安全にスキップ

                imgs, tabs, cls, risks, symbol_ids = batch_data
                imgs, tabs, symbol_ids = imgs.to(device), tabs.to(device), symbol_ids.to(device)
                oc, ork = model(imgs, tabs, symbol_ids)

                # 分類評価
                probs = F.softmax(oc, dim=1)
                preds = torch.argmax(probs, dim=1)

                # 評価時は元の target_class の値 (-1, 0, 1) に戻す
                cls_true_batch = cls.cpu().numpy().tolist()
                cls_pred_batch = (preds - 1).cpu().numpy().tolist()

                # リスク評価 (スケーリングを元に戻す：逆変換)
                # 🌟 apply_scaler.py は target_risk_pct を除外して保存しているため、
                #    test_data/*.csv 内の値はすでに「元のパーセント単位」。
                #    モデルの生出力(train.py独自スケール)のみ逆変換すれば単位が揃う。
                pred_risk_scaled = ork.cpu().numpy().flatten()
                pred_risk_original = (pred_risk_scaled * s_std + s_mean).tolist()
                true_risk_original = risks.cpu().numpy().flatten().tolist()

                sym_data["cls_true"].extend(cls_true_batch)
                sym_data["cls_pred"].extend(cls_pred_batch)
                sym_data["risk_true"].extend(true_risk_original)
                sym_data["risk_pred"].extend(pred_risk_original)

                all_cls_true.extend(cls_true_batch)
                all_cls_pred.extend(cls_pred_batch)
                all_risk_true.extend(true_risk_original)
                all_risk_pred.extend(pred_risk_original)

    if len(all_cls_true) == 0:
        return 0.0, 999.0, per_symbol, (all_cls_true, all_cls_pred, all_risk_true, all_risk_pred)

    acc = accuracy_score(all_cls_true, all_cls_pred)
    mae = mean_absolute_error(all_risk_true, all_risk_pred)

    return acc, mae, per_symbol, (all_cls_true, all_cls_pred, all_risk_true, all_risk_pred)


def print_symbol_report(pair_name, cls_true, cls_pred, risk_true, risk_pred):
    """
    銘柄単体の分類レポート・混同行列・予測クラスの偏りを表示する。
    GOLD/GBPJPYのように「特定方向にしか予測しない」銘柄を一目で見つけるのが目的。
    """
    target_names = ['SELL (-1)', 'HOLD (0)', 'BUY (1)']
    n = len(cls_true)
    if n == 0:
        logging.warning(f"  [{pair_name}] 評価可能なサンプルが0件のためスキップします。")
        return

    acc = accuracy_score(cls_true, cls_pred)
    mae = mean_absolute_error(risk_true, risk_pred) if len(risk_true) > 0 else float("nan")

    pred_counts = {c: cls_pred.count(c) for c in [-1, 0, 1]}
    true_counts = {c: cls_true.count(c) for c in [-1, 0, 1]}
    pred_ratio = {c: pred_counts[c] / n for c in [-1, 0, 1]}

    dominant_class = max(pred_ratio, key=pred_ratio.get)
    bias_flag = ""
    if pred_ratio[dominant_class] >= 0.70:
        label_str = {-1: "SELL", 0: "HOLD", 1: "BUY"}[dominant_class]
        bias_flag = f"  ⚠️ 予測の{pred_ratio[dominant_class]:.0%}が{label_str}に偏っています（方向バイアスの疑いあり）"

    # 🌟 target_risk_pct 自体が元々小さいスケールの値のため、MAE単体では
    #    「性能が良い/バグがある」を判断できない。基準値として、
    #    対象の平均・標準偏差、および「平均値を毎回予測した場合のMAE(ナイーブ基準)」を併記する。
    risk_note = ""
    if len(risk_true) > 0:
        risk_arr = np.array(risk_true, dtype=np.float64)
        risk_mean_val = float(risk_arr.mean())
        risk_std_val = float(risk_arr.std())
        naive_mae = float(np.mean(np.abs(risk_arr - risk_mean_val)))
        improvement = (1 - mae / naive_mae) * 100 if naive_mae > 1e-8 else float("nan")
        risk_bias_flag = ""
        if naive_mae > 1e-8 and mae >= naive_mae * 0.9:
            risk_bias_flag = "  ⚠️ 平均値をただ予測した場合とほぼ同じ精度＝リスク回帰が機能していない可能性"
        risk_note = (
            f"\n  [Risk基準値] 対象の平均: {risk_mean_val:.6f}% / 標準偏差: {risk_std_val:.6f}% / "
            f"平均値で固定予測した場合のMAE: {naive_mae:.6f}% (モデルはこれより {improvement:.1f}% 改善){risk_bias_flag}"
        )

    print(f"\n=== [{pair_name}]  (評価サンプル数: {n}){bias_flag} ===")
    print(f"  Accuracy: {acc:.4f} | Risk MAE: {mae:.6f}%{risk_note}")
    print(
        f"  正解の内訳: SELL {true_counts[-1]:>4} ({true_counts[-1]/n:>6.1%})  "
        f"HOLD {true_counts[0]:>4} ({true_counts[0]/n:>6.1%})  "
        f"BUY {true_counts[1]:>4} ({true_counts[1]/n:>6.1%})"
    )
    print(
        f"  予測の内訳: SELL {pred_counts[-1]:>4} ({pred_ratio[-1]:>6.1%})  "
        f"HOLD {pred_counts[0]:>4} ({pred_ratio[0]:>6.1%})  "
        f"BUY {pred_counts[1]:>4} ({pred_ratio[1]:>6.1%})"
    )

    cm = confusion_matrix(cls_true, cls_pred, labels=[-1, 0, 1])
    print("        Pred: SELL  HOLD  BUY")
    for i, true_label in enumerate(['SELL', 'HOLD', 'BUY']):
        row_str = " ".join([f"{val:<5}" for val in cm[i]])
        print(f"  True: {true_label:<4}  {row_str}")

    print(classification_report(cls_true, cls_pred, labels=[-1, 0, 1], target_names=target_names, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="訓練済みマルチモーダルFXモデルの評価プログラム（銘柄別診断対応版）")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"], help="評価対象のモード")
    args = parser.parse_args()

    setup_logger()
    logging.info(f"--- 📊 マルチモーダルFXモデルの評価開始 (モード: {args.mode}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. train.py の仕様に合わせてリスクスケーラー情報ファイル名を参照
    risk_scaler_path = f"saved_models/risk_scalers_{args.mode}.json"
    if not os.path.exists(risk_scaler_path):
        logging.error(f"❌ リスクスケーラー情報が見つかりません: {risk_scaler_path}")
        return

    with open(risk_scaler_path, "r", encoding="utf-8") as f:
        risk_scalers = json.load(f)

    # 2. 評価データの準備（test_dataフォルダ配下の専用テストCSVを使用。train.pyが分割・保存したもの）
    test_files = glob.glob(f"test_data/*_{args.mode}_test.csv")
    if not test_files:
        logging.error(f"❌ {args.mode} モードの評価用テストデータ(test_data/*_test.csv)が見つかりません。")
        return

    test_loaders = []  # (loader, pair_name, n_features)
    for csv_file in sorted(test_files):
        filename = os.path.basename(csv_file)
        pair_name = filename.split("_")[0]  # 例: USDJPY

        img_dir = f"images/{pair_name}/{args.mode}_MTF"

        try:
            dataset = FXMultimodalDataset(csv_file=csv_file, img_dir=img_dir, symbol=pair_name, seq_length=40)
        except FileNotFoundError as e:
            logging.warning(f"⚠️ スキップ: {pair_name} の評価データを登録できませんでした。原因: {e}")
            continue

        if len(dataset) == 0:
            logging.warning(f"⚠️ スキップ: {pair_name} は有効なターゲットが0件のため評価できません。")
            continue

        # 🌟 test_data/*.csv がどの期間のものかを表示する。
        #    train.py はこのファイルが既に存在すると再分割せず使い回すため、
        #    古いキャッシュを気づかずに評価してしまう事故を防ぐための可視化。
        try:
            header_cols = pd.read_csv(csv_file, nrows=0).columns
            raw_df = pd.read_csv(csv_file, usecols=["time"]) if "time" in header_cols else None
        except Exception:
            raw_df = None

        if raw_df is not None and len(raw_df) > 0:
            logging.info(
                f"✅ テストデータ登録: {pair_name} "
                f"(行数:{len(raw_df)} / 有効サンプル:{len(dataset)} / 期間: {raw_df['time'].min()} 〜 {raw_df['time'].max()})"
            )
        else:
            logging.info(f"✅ テストデータ登録: {pair_name} (有効サンプル:{len(dataset)})")

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            collate_fn=safe_collate_fn
        )
        test_loaders.append((loader, pair_name, len(dataset.features_cols)))

    if not test_loaders:
        logging.error("❌ 評価可能なデータローダーが1つも存在しません。終了します。")
        return

    # 3. モデルの構造定義と各種重みの評価
    model_dir = "saved_model"
    model_variants = {
        "Best TOTAL Model (Class + Risk balanced)": f"best_total_model_{args.mode}.pth",
        "Best CLASS Model (Accuracy optimized)": f"best_class_model_{args.mode}.pth",
        "Best RISK Model (MAE optimized)": f"best_risk_model_{args.mode}.pth",
        "Final Epoch Model": f"model_{args.mode}.pth"
    }

    best_overall_acc = -1.0
    final_per_symbol = None
    final_pooled = None
    best_variant_name = ""
    best_model_src = None

    for name, filename in model_variants.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            logging.warning(f"⚠️ モデル重みファイルが見つかりません。スキップします: {path}")
            continue

        logging.info(f"\n🔍 評価中... [{name}]")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # 🌟 このチェックポイントが実際に学習された入力次元数を、重み自体から確認する
        expected_dim = get_checkpoint_input_dim(checkpoint)
        if expected_dim is None:
            logging.warning(f"⚠️ {path} からモデルの入力次元数を読み取れませんでした。スキップします。")
            continue

        model = MultimodalFXmodel(num_tabular_features=expected_dim).to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"❌ {path} の読み込みに失敗しました（モデル構造が一致しない可能性）: {e}")
            continue

        # 🌟 各銘柄のテストCSVの特徴量数が、このモデルの入力次元と一致するかを事前チェック。
        #    一致しない銘柄は「古いtest_data/キャッシュ」等の疑いがあるため、この変種の評価から除外する。
        usable_loaders = []
        for loader, pair_name, n_features in test_loaders:
            if n_features != expected_dim:
                logging.warning(
                    f"⚠️ [{pair_name}] のテストデータ特徴量数({n_features})がモデルの入力次元({expected_dim})と不一致。"
                    f"test_data/ や temp_train_*.csv が古いキャッシュのままの可能性があります。"
                    f"この銘柄はこのモデルの評価から除外します。"
                )
                continue
            usable_loaders.append((loader, pair_name))

        if not usable_loaders:
            logging.error(f"❌ [{name}] は入力次元が一致する評価データが1つもないため、評価をスキップします。")
            continue

        acc, mae, per_symbol, pooled = evaluate_single_model(model, usable_loaders, device, risk_scalers)

        logging.info(f" ➡️ 結果(全銘柄プール) - Accuracy: {acc:.4f} | Risk MAE: {mae:.6f}%")

        if acc > best_overall_acc:
            best_overall_acc = acc
            final_per_symbol = per_symbol
            final_pooled = pooled
            best_variant_name = name
            best_model_src = path

    if final_pooled is None:
        logging.error("❌ 評価を実行できるモデルの重みが1つも見つかりませんでした。")
        return

    # 4. 最も成績が良かったモデルを確定し、Google Driveへバックアップ
    logging.info("\n" + "🏆" * 30)
    logging.info(f" 最終選定モデル: {best_variant_name} (全銘柄プール Accuracy: {best_overall_acc:.4f})")

    drive_model_dir = "/content/drive/MyDrive/FX_AI_Models"
    if os.path.exists("/content/drive/MyDrive") and best_model_src:
        os.makedirs(drive_model_dir, exist_ok=True)
        drive_best_model_dest = os.path.join(drive_model_dir, f"final_best_model_{args.mode}.pth")
        shutil.copy(best_model_src, drive_best_model_dest)
        logging.info(f"Final best model is saved to Google Drive: {drive_best_model_dest}")
    logging.info("🏆" * 30)

    # ------------------------------------------------------------
    # 全体(プール)レポート ※従来通り、全銘柄を混ぜた総合の健全性チェック用
    # ------------------------------------------------------------
    t_cls, p_cls, t_risk, p_risk = final_pooled
    target_names = ['SELL (-1)', 'HOLD (0)', 'BUY (1)']

    logging.info("\n--- Overall (全銘柄プール) Classification Report ---")
    print(classification_report(t_cls, p_cls, labels=[-1, 0, 1], target_names=target_names, zero_division=0))

    logging.info("--- Overall (全銘柄プール) Confusion Matrix ---")
    cm = confusion_matrix(t_cls, p_cls, labels=[-1, 0, 1])
    print("      Pred: SELL  HOLD  BUY")
    for i, true_label in enumerate(['SELL', 'HOLD', 'BUY']):
        row_str = " ".join([f"{val:<5}" for val in cm[i]])
        print(f"True: {true_label:<4}  {row_str}")
    overall_mae = mean_absolute_error(t_risk, p_risk)
    # 🌟 全銘柄プールでのナイーブ基準(平均値を毎回予測した場合)。
    #    銘柄ごとにtarget_risk_pctのスケールが異なるため参考値だが、
    #    「MAEが小さいのは対象自体のスケールが小さいだけでは」という疑問への簡易チェックになる。
    t_risk_arr = np.array(t_risk, dtype=np.float64)
    overall_naive_mae = float(np.mean(np.abs(t_risk_arr - t_risk_arr.mean()))) if len(t_risk_arr) > 0 else float("nan")
    logging.info(
        f"--- Overall Risk Prediction MAE: {overall_mae:.6f}% "
        f"(参考: 対象の平均値で固定予測した場合のMAE = {overall_naive_mae:.6f}%。銘柄別の詳細は下記参照) ---"
    )

    # ------------------------------------------------------------
    # 🌟 銘柄別レポート：GOLD/GBPJPYのような特定銘柄の方向バイアスを見つけるための本題
    # ------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info(" 銘柄別 詳細レポート")
    logging.info("=" * 60)

    for pair_name in sorted(final_per_symbol.keys()):
        d = final_per_symbol[pair_name]
        print_symbol_report(pair_name, d["cls_true"], d["cls_pred"], d["risk_true"], d["risk_pred"])


if __name__ == "__main__":
    main()
