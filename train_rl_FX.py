import torch
import os
import logging
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 既存のPyTorchモデルとデータセットをインポート
from model import MultimodalFXmodel
from dataset import FXMultimodalDataset
from fx_env import FXTradingEnv

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()

    # ==========================================
    # 📌 コマンドライン引数の設定
    # ==========================================
    parser = argparse.ArgumentParser(description="強化学習 (PPO) トレーニング (ハイブリッドMTF版)")
    parser.add_argument("--symbol", type=str, default="USDJPY", help="学習する銘柄 (例: USDJPY)")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="トレードモード")
    args = parser.parse_args()

    symbol = args.symbol
    mode = args.mode

    # 💡 モードに応じた「執行足（価格計算の基準）」の動的設定
    if mode == "short":
        price_col = "M5_close"
        target_features = ["M5_RSI", "M15_ema20_diff", "M30_ADX"] 
    elif mode == "medium":
        price_col = "H1_close"
        target_features = ["H1_RSI", "H4_ema20_diff", "D1_ADX"] 
    elif mode == "long":
        price_col = "H4_close"
        target_features = ["H4_RSI", "D1_ema20_diff", "W1_ADX"]

    logging.info(f"=== 強化学習 (PPO) トレーニング開始 ===")
    logging.info(f"🎯 ターゲット銘柄: {symbol} | モード: {mode} | 執行足: {price_col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用デバイス: {device}")

    # ==========================================
    # 📌 動的パスの設定
    # ==========================================
    DRIVE_BASE = "/content/datafetch_modeltrain"
    model_path = os.path.join(DRIVE_BASE, "saved_model", f"final_best_model_{mode}_FX.pth")
    scaled_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_scaled_FX.csv")
    raw_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_labeled_FX.csv")
    img_dir = f"/content/datafetch_modeltrain/images/{symbol}/{mode}_MTF"
    
    save_dir = os.path.join(DRIVE_BASE, "rl_models")
    os.makedirs(save_dir, exist_ok=True)
    
    # 💡 継続学習用のパス設定
    rl_model_path = os.path.join(save_dir, f"ppo_fx_trader_{symbol}_{mode}.zip")
    vec_normalize_path = os.path.join(save_dir, f"vec_normalize_{symbol}_{mode}.pkl")

    # ==========================================
    # 1. データセットのロード
    # ==========================================
    logging.info(f"データをロード中...")
    try:
        dataset = FXMultimodalDataset(csv_file=scaled_csv, img_dir=img_dir, seq_length=40)
        raw_df = pd.read_csv(raw_csv)
    except Exception as e:
        logging.error(f"❌ データの読み込みに失敗しました: {e}")
        return

    # ==========================================
    # 2. PyTorchモデル (アナリスト) のロード
    # ==========================================
    INPUT_DIM = len(dataset.features_cols) 
    pytorch_model = MultimodalFXmodel(num_tabular_features=INPUT_DIM).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pytorch_model.load_state_dict(checkpoint)
        logging.info("✅ 学習済みPyTorchモデルをロードしました。")
    else:
        logging.error(f"❌ PyTorchモデルが見つかりません: {model_path}")
        return

    # ==========================================
    # 3. Gym環境の構築 (ハイブリッド仕様)
    # ==========================================
    raw_env = FXTradingEnv(
        dataset=dataset, 
        trained_model=pytorch_model,
        price_col=price_col,
        raw_features=target_features,
        raw_data_df=raw_df
    )

    check_env(raw_env)
    logging.info("✅ 強化学習環境(ハイブリッド型)のチェック完了。")

    env = DummyVecEnv([lambda: raw_env])
    
    # ✨ 状態空間の正規化ラッパー
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ==========================================
    # 4. PPO（強化学習アルゴリズム）の実行 (継続学習対応)
    # ==========================================
    # 既に過去の強化学習モデルが存在するか判定
    if os.path.exists(rl_model_path):
        logging.info(f"🔄 既存の強化学習モデルを発見しました: {rl_model_path} から継続学習を開始します。")
        
        # モデルのロード
        rl_model = PPO.load(rl_model_path, env=env, device=device)
        
        # 過去の正規化統計情報（.pkl）の引き継ぎ
        if os.path.exists(vec_normalize_path):
            logging.info(f"📈 過去の環境正規化統計情報をロードします: {vec_normalize_path}")
            env = VecNormalize.load(vec_normalize_path, env)
            rl_model.set_env(env)
    else:
        logging.info("🆕 新規の強化学習モデルを作成して学習を開始します。")
        rl_model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            n_steps=2048, 
            batch_size=64, 
            n_epochs=10, 
            gamma=0.99, 
            gae_lambda=0.95, 
            clip_range=0.2, 
            ent_coef=0.01,
            device=device
        )

    total_timesteps = 200000 
    logging.info(f"🚀 学習を開始します (総ステップ数: {total_timesteps})...")
    rl_model.learn(total_timesteps=total_timesteps)

    # ==========================================
    # 5. モデルおよび環境統計情報の保存
    # ==========================================
    rl_model.save(rl_model_path)
    env.save(vec_normalize_path)
    
    logging.info(f"🎉 強化学習モデルを保存しました: {rl_model_path}")
    logging.info(f"✨ 環境正規化統計情報を保存しました: {vec_normalize_path}")

if __name__ == "__main__":
    main()