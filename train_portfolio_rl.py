import os
import torch
import logging
from stable_baselines3 import SAC
import shutil
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
# 作成したモジュールをインポート
from portfolio_dataset import PortfolioFXDataset
from portfolio_env import PortfolioFXEnv
from model import MultimodalFXmodel  # 既存のアナライザモデル


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def main():
    setup_logger()
    logging.info("portfolio_portfolio_rl.py ver208.2")
    
    # 1. 基本設定
    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    mode = "short"
    merged_csv = "portfolio_merged_data_train_with_preds.csv"
    img_dir = "images"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")



    # 2. データセットの準備
    logging.info("データセットを読み込んでいます...")
    dataset = PortfolioFXDataset(merged_csv_path=merged_csv, base_img_dir=img_dir, symbols=symbols, mode=mode)

    
    #3.アナライザのモデル読み込み(省略)
        
    # 4. 強化学習環境の構築
    env = PortfolioFXEnv(dataset=dataset,symbols=symbols)
    vec_env = DummyVecEnv([lambda: env])

    vec_env = VecNormalize(
    vec_env,
    training=True,
    norm_obs=True,
    norm_reward=True,       # 報酬の正規化（学習の安定化に寄与します）
    clip_obs=10.0,          # 極端な外れ値をクリップ
    clip_reward=10.0,
    gamma=0.99,             # ※お使いのアルゴリズムの gamma (割引率) に合わせて変更してください
    epsilon=1e-8
)
    # 5. SACエージェントの定義と学習
    # 21次元の連続値アクションなので、MLPポリシーを使用します
    
    agent_save_path_base = f"saved_rl_models/sac_portfolio_agent_{mode}"
    vec_norm_path = "saved_rl_models/vec_normalize.pkl"
    replay_buffer_path = f"saved_rl_models/sac_replay_buffer_{mode}.pkl"

    if os.path.exists(f"{agent_save_path_base}.zip"):
        logging.info("既存のモデルを発見しました。")
        vec_env = VecNormalize.load(vec_norm_path,vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

        model = SAC.load(agent_save_path_base,env = vec_env,device = device)

        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            logging.info("リプレイバッファを読み込みました")
        else:
            logging.warning("リプレイバッファが見つかりません。新規メモリで再開します。")
    else:
        logging.info("🚀 新規モデルとしてSACエージェントの学習を開始します...")
        model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log="./sac_portfolio_tensorboard/")

    # 学習実行（ステップ数は適宜調整）
    model.learn(total_timesteps=1000000,reset_num_timesteps=False)
    
    # モデルの保存
    os.makedirs("saved_rl_models", exist_ok=True)

    
    model.save(agent_save_path_base)
    vec_env.save(vec_norm_path)
    model.save_replay_buffer(replay_buffer_path)
    
    logging.info("学習が完了し、モデルを保存しました！")

    drive_base_path = "/content/drive/MyDrive"
    
    if os.path.exists(drive_base_path):
        logging.info("マウント済みのGoogle Driveを検出しました。コピーを開始します...")
        drive_save_dir = os.path.join(drive_base_path, "saved_models_rl")
        os.makedirs(drive_save_dir, exist_ok=True)
        
        agent_zip_path = f"{agent_save_path_base}.zip"
        files_to_copy = [agent_zip_path, vec_norm_path,replay_buffer_path]
        
        for file_path in files_to_copy:
            if os.path.exists(file_path):
                # shutil.copy2 を使うことでファイルのメタデータ（作成日時など）も保持してコピーします
                shutil.copy2(file_path, drive_save_dir)
                logging.info(f"Driveに保存しました: {os.path.basename(file_path)}")
            else:
                logging.warning(f"コピー元ファイルが見つかりません: {file_path}")
                
        logging.info("Google Driveへの保存処理がすべて完了しました！")
    else:
        logging.warning("Google Driveがマウントされていないようです。Driveへの保存はスキップされました。")
            
        

if __name__ == "__main__":
    main()