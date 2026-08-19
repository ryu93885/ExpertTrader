import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from portfolio_dataset import PortfolioFXDataset
from portfolio_env import PortfolioFXEnv
from model import MultimodalFXmodel

def setup_logger():
    logging.basicConfig(level = logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()
    logging.info("test_portfolio.py ver208.1")
    parser = argparse.ArgumentParser("test portfolio rl model")
    parser.add_argument("--symbols",default=["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "GOLD"])
    parser.add_argument("--mode",default = "short",help = "モード選択",choices = ["short","medium","long"])
    args = parser.parse_args()
    test_csv = "portfolio_merged_data_test.csv"
    img_dir = "images"

    analyzer_model_path = f"saved_model/best_total_model_{args.mode}.pth"
    rl_model_path = "saved_rl_models/sac_portfolio_agent.zip"
    vec_norm_path = "saved_rl_models/vec_normalize.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device:{device}")

    if not os.path.exists(test_csv):
        logging.error(f"✖テストデータ{test_csv}が見つかりません.merge_portfolio_data.pyを実行してください")
        return 

    logging.info("テスト用データセットを読み込んでいます....")
    dataset = PortfolioFXDataset(merged_csv_path=test_csv,base_img_dir=img_dir,symbols = args.symbols,mode = args.mode)


    input_dim = len(dataset.feature_cols[args.symbols[0]])
    analyzer = MultimodalFXmodel(num_tabular_features=input_dim).to(device)

    if os.path.exists(analyzer_model_path):
        checkpoint = torch.load(analyzer_model_path,map_location=device,weights_only=False)
        analyzer.load_state_dict(checkpoint.get("model_state_dict",checkpoint))
        analyzer.eval()
        logging.info("✅ アナライザモデルの重みをロードしました。")

    else:
        logging.error("❌ アナライザモデルが見つかりません。")
        return

    #テスト用環境の構築とvecnormalizeの復元
    env = PortfolioFXEnv(dataset = dataset,trained_model = analyzer,symbols = args.symbols)
    vec_env = DummyVecEnv([lambda:env])

    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path,vec_env)

        vec_env.training = False
        vec_env.norm_reward = False
        logging.info("✅ VecNormalize の統計情報（平均・分散）をロードし、推論モードに設定しました。")
    else:
        logging.error(f"❌ {vec_norm_path} が見つかりません。学習が完了しているか確認してください。")
        return

    #SACエージェントのロード
    if os.path.exists(rl_model_path):
        model = SAC.load(rl_model_path,device = device)
        logging.info("✅ SACエージェントをロードしました。")
    else:
        logging.error("❌ SACモデルが見つかりません。")
        return 



    #バックテストの実行
    logging.info("🚀 テストシミュレーションを開始します...")
    obs = vec_env.reset()
    done = False

    equities = []
    balances = []

    step_count = 0


    while not done:
        action,_states = model.predict(obs,deterministic=True)
        obs,rewards,dones,infos = vec_env.step(action)

        done = dones[0]
        info = infos[0]

        equities.append(info["equity"])
        balances.append(info["balance"])

        step_count += 1
        if step_count % 100 == 0:
            logging.info(f"Step {step_count:04d} | Equity: {info['equity']:,.0f} 円 | Balance: {info['balance']:,.0f} 円")


    #結果の評価と可視化
    try:
        plt.figure(figsize=(12,6))
        plt.plot(equities,label = "Equity",color = "blue",linewidth = 1.5)
        plt.plot(balances,label = "Balance",color = "orange",alpha = 0.7,linestyle = "--")
        plt.title(f"Portfolio AI Trading Test Result({args.mode} mode)")
        plt.xlabel("Steps")
        plt.ylabel("AccountAmount(JPY)")
        plt.legend()
        plt.grid(True,alpha = 0.3)
        plt.tight_layout()

        save_path = "portfolio_test_result.png"
        plt.savefig(save_path)
        logging.info(f"🖼️ 資産推移のグラフを保存しました: {save_path}")
    except Exception as e:
        logging.error(f"グラフの生成に失敗しました。:{e}")



if __name__ == "__main__":
    main()