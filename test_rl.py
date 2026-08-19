import torch
import os
import logging
import argparse
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from model import MultimodalFXmodel
from dataset import FXMultimodalDataset
from fx_env import FXTradingEnv

def setup_logger():
    logging.basicConfig(level = logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()
    logging.info("test_rl:ver204.1")
    #コマンドライン引数の指定
    parser = argparse.ArgumentParser(description = "強化学習(SAC)バックテスト")
    parser.add_argument("--symbol",type = str,default = "USDJPY",help = "テストする銘柄")
    parser.add_argument("--mode",type = str,default = "short",choices=["short","medium","long"])
    args = parser.parse_args()

    symbol = args.symbol
    mode = args.mode

    if mode == "short":
        price_col = "M5_close"
        target_features = ["M5_RSI", "M15_ema20_diff", "M30_ADX","M5_ATR"] 
    elif mode == "medium":
        price_col = "H1_close"
        target_features = ["H1_RSI", "H4_ema20_diff", "D1_ADX","H1_ATR"] 
    elif mode == "long":
        price_col = "H4_close"
        target_features = ["H4_RSI", "D1_ema20_diff", "W1_ADX","H4_ATR"]
    
    logging.info(f"===バックテスト開始:{symbol}|mode:{mode}")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #パス指定
    DRIVE_BASE = "/content/datafetch_modeltrain"
    model_path = os.path.join(DRIVE_BASE, "saved_model", f"final_best_model_{mode}.pth")
    scaled_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_scaled_test.csv")
    raw_csv = os.path.join(DRIVE_BASE, "labeled_data", f"{symbol}_{mode}_labeled_test.csv")
    img_dir = f"/content/datafetch_modeltrain/images/{symbol}/{mode}_MTF"
    
    save_dir = os.path.join(DRIVE_BASE, "rl_models")
    rl_model_path = os.path.join(save_dir, f"sac_fx_trader_{symbol}_{mode}.zip")
    vec_normalize_path = os.path.join(save_dir, f"vec_normalize_{symbol}_{mode}.pkl")

    for path in [model_path,rl_model_path,vec_normalize_path]:
        if not os.path.exists(path):
            logging.error(f"必要なファイルが見つかりません:{path}")
            return
    #1.テスト用データセットのロード
    try:
        raw_df = pd.read_csv(raw_csv)
        dataset = FXMultimodalDataset(csv_file=scaled_csv, img_dir=img_dir, seq_length=40)
        logging.info(f"✅テストデータロード完了 (ステップ数:{len(dataset)})")
    except Exception as e:
        logging.error(f"データの読み込みに失敗しました:{e}")
        return

    #2.PyTorchのモデルのロード
    INPUT_DIM = len(dataset.features_cols)
    pytorch_model = MultimodalFXmodel(num_tabular_features=INPUT_DIM).to(device)
    checkpoint = torch.load(model_path,map_location = device,weights_only = False)
    if "model_state_dict" in checkpoint:
        pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        pytorch_model.load_state_dict(checkpoint)
    #テスト時は推論モードへ
    pytorch_model.eval()
    
    if symbol == "GOLD":
        env_tx_cost = 0.4        # GOLDの想定スプレッド (例: 4.0 pips)
        env_contract = 100       # 1ロット = 100オンス
        env_exchange = 1.0     # 円換算レート（日本円口座想定）
        env_init_balance = 10000
    else:
        # USDJPYなど、通常のクロス円通貨ペアのデフォルト
        env_tx_cost = 0.01       # 例: USDJPYで1.0 pips想定なら 0.01
        env_contract = 100000    # 1ロット = 10万通貨
        env_exchange = 1.0       # すでに円建てなので 1.0
        env_init_balance = 10000

    #テスト用環境の構築
    raw_env = FXTradingEnv(
        dataset = dataset,
        trained_model = pytorch_model,
        price_col = price_col,
        raw_features=target_features,
        raw_data_df=raw_df,
        transaction_cost=env_tx_cost,
        contract_size=env_contract,
        exchange_rate=env_exchange,
        max_steps=len(dataset)-1,
        initial_balance=env_init_balance
    )
    env = DummyVecEnv([lambda:raw_env])

    #パラメータを固定
    env = VecNormalize.load(vec_normalize_path,env)
    env.training = False
    env.norm_reward = False

    #強化学習モデルのロード
    rl_model = SAC.load(rl_model_path,env = env,device = device)
    logging.info("✅すべてのモデルと環境の準備が完了しました。バックテストを開始します。")

    #バックテストの実行
    obs = env.reset()
    done = [False]
    

    total_rewards = 0.0
    steps = 0

    while not done[0]:
        action,_states = rl_model.predict(obs,deterministic = True)
        obs,reward,done,info = env.step(action)

        total_rewards += reward[0]
        steps += 1
        if steps % 100 == 0:
            logging.info(f"Step: {steps} | 現在の評価損益: {info[0].get('balance', 0):.2f}")
        current_equity = info[0]["equity"] if isinstance(info, list) else info.get("equity", 0)

    # ログ出力（例）
        if steps % 100 == 0:
            logging.info(f"Step: {steps} | 現在の資産状況: {current_equity:.2f}")
            
        if info[0].get("bankruptcy", False):
            logging.warning("⚠️ 破産ラインに到達したため、テストを強制終了しました。")
            break
    
    logging.info(f"✅バックテスト完了")
    logging.info(f"✅総ステップ数:{steps}")
    logging.info(f'最終資産状況:{info[0].get("balance","N/A")}')


if __name__ == "__main__":
    main()
    