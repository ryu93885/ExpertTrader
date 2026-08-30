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
from rl_checkpoint_utils import resolve_model_paths

def setup_logger():
    logging.basicConfig(level = logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")

def main():
    setup_logger()
    logging.info("test_portfolio.py ver209.0")
    parser = argparse.ArgumentParser("test portfolio rl model")
    # 💡 修正: 実際に学習・precompute_portfolio.py/merge_portfolio_data.py で使われている
    # 7銘柄・並び順(portfolio_trading_bot.py の TARGET_SYMBOLS と完全一致)に修正。
    # 旧デフォルトは USDCAD/USDCHF を含んでおり、実際のシステムの銘柄構成と一致していなかった。
    parser.add_argument("--symbols",default=["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"])
    parser.add_argument("--mode",default = "short",help = "モード選択",choices = ["short","medium","long"])
    args = parser.parse_args()
    # 💡 修正: PortfolioFXEnv は prob_0/1/2・risk_val 列を必要とするが、
    # これらは precompute_portfolio.py --phase test で追加された
    # "_with_preds" 付きファイルにしか存在しない。"_with_preds" の付かない
    # 生の merge_portfolio_data.py の出力をそのまま使うと、これらの列が
    # 存在せず、エージェントが常に prob=0/risk=0 の観測で動くという
    # 気づきにくい形で結果が壊れてしまう。
    test_csv = "portfolio_merged_data_test_with_preds.csv"
    img_dir = "images"

    rl_model_path, vec_norm_path, _replay_buffer_path, model_source = resolve_model_paths(args.mode)
    if rl_model_path is None:
        logging.error(
            "❌ SACモデルが見つかりません(完了済みモデル・Drive上のコピー・"
            "途中保存のいずれも見つかりませんでした)。学習が完了しているか確認してください。"
        )
        return
    logging.info(f"📦 使用するモデル: {model_source} ({rl_model_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device:{device}")

    if not os.path.exists(test_csv):
        logging.error(
            f"✖テストデータ{test_csv}が見つかりません。"
            f"merge_portfolio_data.py の実行後、"
            f"python precompute_portfolio.py --phase test --mode {args.mode} を実行してください"
        )
        return

    logging.info("テスト用データセットを読み込んでいます....")
    dataset = PortfolioFXDataset(merged_csv_path=test_csv,base_img_dir=img_dir,symbols = args.symbols,mode = args.mode)

    # 💡 修正: PortfolioFXEnv は prob_0/1/2・risk_val を precompute_portfolio.py が
    # 事前計算した列から直接読み込む設計(train_portfolio_rl.py と同じ)で、
    # アナリストモデルを直接渡す trained_model 引数はそもそも存在しない。
    # 旧コードはここでTypeErrorになっており、かつロード済みの analyzer もどこにも
    # 使われていなかった(死んだコード)ため、丸ごと削除した。

    #テスト用環境の構築とvecnormalizeの復元
    env = PortfolioFXEnv(dataset = dataset,symbols = args.symbols)
    vec_env = DummyVecEnv([lambda:env])

    vec_env = VecNormalize.load(vec_norm_path,vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    logging.info("✅ VecNormalize の統計情報（平均・分散）をロードし、推論モードに設定しました。")

    #SACエージェントのロード
    model = SAC.load(rl_model_path,device = device)
    logging.info("✅ SACエージェントをロードしました。")



    #バックテストの実行
    logging.info("🚀 テストシミュレーションを開始します...")
    obs = vec_env.reset()
    done = False

    equities = []
    balances = []

    step_count = 0


    done_reason = None

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
            # 💡 追加: 銘柄別の建玉・価格・実現/含み損益を出力する。
            # 資産の急変(急騰・急落)がどの銘柄に由来するかを確認するため。
            positions = info.get("positions", {})
            prices = info.get("prices", {})
            unrealized = info.get("unrealized_pnl_by_symbol", {})
            realized = info.get("realized_pnl_by_symbol", {})
            for sym in positions:
                pos = positions[sym]
                r_pnl = realized.get(sym, 0.0)
                if pos != 0 or r_pnl != 0:
                    logging.info(
                        f"    [{sym}] pos={pos:+.2f}lot 価格={prices.get(sym, 0):.3f} "
                        f"含み損益={unrealized.get(sym, 0):,.0f}円 直近の実現損益={r_pnl:,.0f}円"
                    )

        if done:
            done_reason = info.get("done_reason")


    # 💡 追加: バックテストが「テスト期間を最後まで走り切った」のか「途中で資産が
    # 初期資金の50%を下回って強制終了した」のかを明示する。データセット全体の
    # 行数と実際に走ったステップ数を突き合わせることで、破産による早期終了を
    # 見落とさないようにする。
    logging.info(f"🏁 テスト終了(総ステップ数: {step_count} / テストデータ行数: {len(dataset)} / 終了理由: {done_reason})")
    if done_reason == "bankruptcy":
        logging.warning(
            "⚠️ 資産が初期資金の50%を下回ったため、テスト期間を最後まで走り切れずに"
            "シミュレーションが強制終了しました。"
        )

    #結果の評価と可視化
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
