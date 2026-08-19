import torch
import os
import logging
import argparse
import pandas as pd
from typing import Callable,Dict,Any

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import shutil

import optuna
from stable_baselines3.common.evaluation import evaluate_policy

# 既存のPyTorchモデルとデータセットをインポート
from model import MultimodalFXmodel
from dataset import FXMultimodalDataset
from fx_env import FXTradingEnv

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def linear_schedule(initial_value:float) -> Callable[[float],float]:
    """
    線形学習率スケジューラ
    :param initial_value:初期学習率(Optunaが決定した値
    :return:残りの進行度(progress_remaining:1.0->0.0)
    """
    def func(progress_remaining:float) -> float:
        return progress_remaining * initial_value
    return func
def main():
    setup_logger()
    logging.info("train_rl:ver204.1")
    # ==========================================
    # 📌 コマンドライン引数の設定
    # ==========================================
    parser = argparse.ArgumentParser(description="強化学習 (PPO) トレーニング (ハイブリッドMTF版)")
    parser.add_argument("--symbol", type=str, default="USDJPY", help="学習する銘柄 (例: USDJPY)")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="トレードモード")
    parser.add_argument("--optimize",action = "store_true",help = "Optunaによるハイパーパラメータ最適化を実行するかどうか")
    parser.add_argument("--n_trials",type = int,default = 15,help = "Optunaの試行回数")

    args = parser.parse_args()

    symbol = args.symbol
    mode = args.mode
    run_optimize = args.optimize
    n_trials = args.n_trials

    # 💡 モードに応じた「執行足（価格計算の基準）」の動的設定
    if mode == "short":
        price_col = "M5_close"
        target_features = ["M5_RSI", "M15_ema20_diff", "M30_ADX","M5_ATR"] 
    elif mode == "medium":
        price_col = "H1_close"
        target_features = ["H1_RSI", "H4_ema20_diff", "D1_ADX","H1_ATR"] 
    elif mode == "long":
        price_col = "H4_close"
        target_features = ["H4_RSI", "D1_ema20_diff", "W1_ADX","H4_ATR"]

    logging.info(f"=== 強化学習 (SAC) トレーニング開始 ===")
    logging.info(f"🎯 ターゲット銘柄: {symbol} | モード: {mode} | 執行足: {price_col}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用デバイス: {device}")

    # ==========================================
    # 📌 動的パスの設定
    # ==========================================
    DRIVE_BASE = "/content/datafetch_modeltrain"
    model_path = os.path.join(DRIVE_BASE, "saved_model", f"final_best_model_{mode}.pth")
    scaled_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_scaled_train.csv")
    raw_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_labeled_train.csv")
    img_dir = f"/content/datafetch_modeltrain/images/{symbol}/{mode}_MTF"
    
    save_dir = os.path.join(DRIVE_BASE, "rl_models")
    os.makedirs(save_dir, exist_ok=True)
    
    # 💡 継続学習用のパス設定
    rl_model_path = os.path.join(save_dir, f"sac_fx_trader_{symbol}_{mode}.zip")
    vec_normalize_path = os.path.join(save_dir, f"vec_normalize_{symbol}_{mode}.pkl")

    # ==========================================
    # 1. データセットのロード
    # ==========================================
    logging.info(f"データをロード中...")
    try:
        raw_df = pd.read_csv(raw_csv)
        dataset = FXMultimodalDataset(csv_file=scaled_csv, img_dir=img_dir, seq_length=40)
        logging.info(f"✅ データロード完了 (有効なエピソード数: {len(dataset)})")
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


    if symbol == "GOLD":
        env_tx_cost = 0.4
        env_contract = 100
        env_exchange = 1.0
        env_init_balance = 10000

    else:
        env_tx_cost = 0.01
        env_contract = 100000
        env_exchange = 1.0
        env_init_balance = 10000

    
    def make_env():
        env = FXTradingEnv(
            dataset = dataset,
            trained_model = pytorch_model,
            price_col=price_col,
            raw_features=target_features,
            raw_data_df=raw_df,
            transaction_cost=env_tx_cost,
            contract_size=env_contract,
            exchange_rate = env_exchange,
            initial_balance=env_init_balance
        )
        env = DummyVecEnv([lambda:env])
        return VecNormalize(env,norm_obs = True,norm_reward = False,clip_obs = 10.0)

    def objective(trial:optuna.Trial) -> float:
        """
        Optunaがハイパーパラメータの組み合わせを評価するための関数。
        """
        # 1. 探索するハイパーパラメータの範囲を定義
        # 初期学習率 (対数スケールで探索: 1e-5 〜 1e-3)
        initial_lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        # バッチサイズ (2の乗数から選択)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        # 割引率 (未来の報酬の重要度: 0.9 〜 0.999)
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
        # SACのターゲットネットワーク更新率
        tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
        # ネットワーク構造 (隠れ層の広さと深さ)
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
        net_arch = {
            "tiny": [64, 64],
            "small": [128, 128],
            "medium": [256, 256]
        }[net_arch_type]
        
        # 2. 試行用の環境構築
        eval_env = make_env()
        
        # 3. SACモデルの構築 (スケジューラを適用)
        # learning_rate引数に、上で定義した linear_schedule 関数を渡します
        model = SAC(
            "MlpPolicy", 
            eval_env, 
            learning_rate=linear_schedule(initial_lr), 
            buffer_size=50000, 
            batch_size=batch_size, 
            ent_coef="auto", 
            gamma=gamma, 
            tau=tau, 
            policy_kwargs={"net_arch": net_arch},
            device=device,
            verbose=0 # 最適化中はログを非表示にしてスッキリさせる
        )
        
        # 4. 短期間の学習 (例として2万ステップ。ここで良し悪しを判定
        # ※本来は数十万ステップ必要
        try:
            model.learn(total_timesteps=20000)
        except Exception as e:
            # 万が一学習中にエラー（破産などによるNaN発生等）が起きたら最低評価を返す
            return -10000.0
            
        # 5. モデルの評価 (評価用環境で10エピソード回し、平均報酬を算出)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        
        return mean_reward
    # ==========================================
    # main実行フロー(Optuna実行 OR 即学習)
    # ==========================================
    best_param = {}

    if run_optimize:
        logging.info(f"🔍 Optunaによるハイパーパラメータ最適化を開始します (試行回数: {n_trials})")
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective,n_trials = n_trials)

        logging.info("🏆 最適化完了！ベストパラメータ:")
        for key,value in study.best_params.items():
            logging.info(f"{key}:{value}")
        best_params = study.best_params
    else:
        logging.info(f"⏭️ Optuna最適化をスキップし、デフォルト(または既存)の設定で学習します。")
        best_params = {
            "learning_rate": 0.0003,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "net_arch": "small"
        }
    
    #============================================
    #学習本番
    #============================================
    logging.info("🚀 本番の強化学習を開始します...")
    env = make_env()

    # net_archの辞書復元
    arch_map = {"tiny": [64, 64], "small": [128, 128], "medium": [256, 256]}
    net_arch_list = arch_map[best_params.get("net_arch", "small")]

    if os.path.exists(rl_model_path) and not run_optimize:
        logging.info(f"🔄 既存のモデル {rl_model_path} から継続学習を開始します。")
        rl_model = SAC.load(rl_model_path, env=env, device=device)
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            rl_model.set_env(env)
    else:
        if run_optimize:
            logging.info("✨ 最適化された新しいパラメータで新規モデルを作成します。")
        else:
            logging.info("🆕 新規の強化学習モデルを作成します。")
            
        rl_model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1, 
            # 💡 変更箇所: 固定値ではなく、スケジューラにベストな初期学習率を渡す
            learning_rate=linear_schedule(best_params["learning_rate"]), 
            buffer_size=100000, 
            batch_size=best_params["batch_size"], 
            ent_coef="auto", 
            gamma=best_params["gamma"], 
            tau=best_params["tau"], 
            policy_kwargs={"net_arch": net_arch_list},
            device=device
        )

    # 本番学習のステップ数（50万ステップ）
    total_timesteps = 500000 
    rl_model.learn(total_timesteps=total_timesteps)

    # 保存処理
    rl_model.save(rl_model_path)
    env.save(vec_normalize_path)

    # Google Drive への自動バックアップ
    drive_save_dir = "/content/drive/MyDrive/FX_AI_Models_RL"
    try:
        os.makedirs(drive_save_dir, exist_ok=True)
        shutil.copy(rl_model_path, os.path.join(drive_save_dir, os.path.basename(rl_model_path)))
        shutil.copy(vec_normalize_path, os.path.join(drive_save_dir, os.path.basename(vec_normalize_path)))
        logging.info(f"☁️ Google Drive へのバックアップ保存が完了しました: {drive_save_dir}")
    except Exception as e:
        logging.warning(f"⚠️ Google Driveへの保存をスキップしました: {e}")

if __name__ == "__main__":
    main()