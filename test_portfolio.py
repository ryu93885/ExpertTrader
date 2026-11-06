import os
import csv
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from portfolio_dataset import PortfolioFXDataset
from portfolio_env import PortfolioFXEnv, RISK_PER_TRADE_PCT_DEFAULT
from rl_checkpoint_utils import resolve_model_paths
from backtest_metrics import compute_performance_summary, log_performance_summary, run_backtest_episode

DRIVE_TEST_RESULTS_DIR = "/content/drive/MyDrive/FX_AI_Models/test_results"
LOCAL_TEST_RESULTS_DIR = "test_results"


def get_test_results_dir():
    if os.path.exists("/content/drive/MyDrive"):
        results_dir = DRIVE_TEST_RESULTS_DIR
    else:
        results_dir = LOCAL_TEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def setup_logger(log_file_path=None):
    handlers = [logging.StreamHandler()]
    if log_file_path is not None:
        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)


def build_env(dataset, args, episode_start_idx=None, episode_end_idx=None):
    return PortfolioFXEnv(
        dataset=dataset, symbols=args.symbols, fixed_start=True,
        entry_threshold_window=args.entry_threshold_window,
        entry_threshold_percentile=args.entry_threshold_percentile,
        max_concurrent_positions=args.max_concurrent_positions,
        risk_per_trade_pct=args.risk_per_trade_pct / 100.0,
        episode_start_idx=episode_start_idx,
        episode_end_idx=episode_end_idx,
    )


def load_vec_env(env, vec_norm_path):
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def run_split_validation(dataset, model, vec_norm_path, args, results_dir, timestamp, episode_start_idx=None):
    """
    💡 追加(2026-11-06): テスト期間全体を使った単発のバックテストだけでは、
    「特定の1期間に偶然当てはまっただけ(過学習)」なのか「どの期間でも概ね
    同程度の成績が出る(再現性がある)」のかを区別できない。テスト期間を
    --n-splits 個の連続する部分期間に分割し、各部分期間を(資金を初期値に
    リセットした)独立したエピソードとして個別に検証することで、設定の
    頻健性を確認できるようにする。

    episode_start_idx が指定されている場合(--test-from-date併用時)は、
    テストCSV全体ではなく、その開始位置以降の範囲だけを分割対象にする。
    """
    full_start = episode_start_idx if episode_start_idx is not None else dataset.seq_length
    full_end = len(dataset.df) - 1  # PortfolioFXEnvの無制限時のdone判定と同じ基準
    total_len = full_end - full_start
    split_size = total_len // args.n_splits

    logging.info("=" * 70)
    logging.info(f"🔁 再現性確認: テスト期間を{args.n_splits}分割し、各期間を個別に検証します")
    logging.info("=" * 70)

    split_summaries = []
    for i in range(args.n_splits):
        s_start = full_start + i * split_size
        s_end = full_end if i == args.n_splits - 1 else full_start + (i + 1) * split_size

        start_time = dataset.df["time"].iloc[s_start]
        end_time = dataset.df["time"].iloc[s_end]

        env_i = build_env(dataset, args, episode_start_idx=s_start, episode_end_idx=s_end)
        vec_env_i = load_vec_env(env_i, vec_norm_path)

        logging.info(f"--- 期間{i + 1}/{args.n_splits}: {start_time} 〜 {end_time} ---")
        result_i = run_backtest_episode(vec_env_i, model)

        elapsed_days_i = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() / 86400.0
        summary_i = compute_performance_summary(
            result_i["equities"], result_i["balances"], result_i["trades"],
            env_i.initial_balance, elapsed_days_i,
        )
        summary_i["period_label"] = f"{start_time} ~ {end_time}"
        split_summaries.append(summary_i)

        monthly_i = summary_i["monthly_return_pct"]
        monthly_str = f"{monthly_i:+.2f}%" if monthly_i is not None else "N/A"
        logging.info(
            f"    trades={summary_i['total_trades']} win_rate={summary_i['win_rate_pct']:.1f}% "
            f"profit_factor={summary_i['profit_factor']:.2f} monthly_return={monthly_str} "
            f"max_dd={summary_i['max_drawdown_pct']:.2f}%"
        )

    logging.info("=" * 70)
    logging.info("🔁 再現性確認 結果一覧")
    logging.info("=" * 70)
    for i, s in enumerate(split_summaries, start=1):
        monthly_i = s["monthly_return_pct"]
        monthly_str = f"{monthly_i:+.2f}%" if monthly_i is not None else "N/A"
        logging.info(
            f"  期間{i} ({s['period_label']}): monthly={monthly_str} "
            f"max_dd={s['max_drawdown_pct']:.2f}% trades={s['total_trades']}"
        )

    split_csv_path = os.path.join(results_dir, f"portfolio_split_validation_{args.mode}_{timestamp}.csv")
    fieldnames = [
        "period_label", "total_trades", "win_rate_pct", "profit_factor",
        "total_return_pct", "monthly_return_pct", "max_drawdown_pct", "net_realized_pnl",
    ]
    with open(split_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in split_summaries:
            writer.writerow({k: s.get(k) for k in fieldnames})
    logging.info(f"📄 分割検証結果をCSVにも保存しました: {split_csv_path}")


def main():
    parser = argparse.ArgumentParser("test portfolio rl model")
    # 💡 修正: 実際に学習・precompute_portfolio.py/merge_portfolio_data.py で使われている
    # 7銘柄・並び順(portfolio_trading_bot.py の TARGET_SYMBOLS と完全一致)に修正。
    # 旧デフォルトは USDCAD/USDCHF を含んでおり、実際のシステムの銘柄構成と一致していなかった。
    parser.add_argument("--symbols",default=["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"])
    parser.add_argument("--mode",default = "short",help = "モード選択",choices = ["short","medium","long"])
    # 💡 追加: 動的エントリー閾値・同時保有上限を手動で試したい場合用のオプション。
    # 省略時はNone(動的閾値オフ・同時保有無制限)で、従来の固定閾値0.2のままになる。
    parser.add_argument("--entry-threshold-window", type=int, default=None,
                         help="動的エントリー閾値のローリングウィンドウ(銘柄ごとのサンプル数)。省略時は従来の固定閾値0.2のまま")
    parser.add_argument("--entry-threshold-percentile", type=float, default=70,
                         help="動的エントリー閾値に使うパーセンタイル")
    parser.add_argument("--max-concurrent-positions", type=int, default=None,
                         help="同時保有できる最大銘柄数。省略時は無制限")
    # 💡 追加(2026-11-06): sweep_entry_threshold.pyと同様、1トレードあたりのリスク率を
    # 手動で試せるようにする。省略時はportfolio_env.py側のデフォルト(2%)のまま。
    parser.add_argument("--risk-per-trade-pct", type=float, default=RISK_PER_TRADE_PCT_DEFAULT * 100.0,
                         help="1トレードあたりのリスク率(%, 口座残高に対する割合)")
    # 💡 追加(2026-11-06): テスト期間全体の単発バックテストに加えて、期間をN分割し
    # 各期間を個別にバックテストする再現性確認を行えるようにする。1(デフォルト)なら
    # 分割なし(従来通り単発のみ)。
    parser.add_argument("--n-splits", type=int, default=1,
                         help="再現性確認用に、テスト期間をいくつの連続した部分期間に分割して"
                              "個別に検証するか。1(デフォルト)なら分割なし")
    # 💡 追加(2026-11-06): train/testの境界日を変えずに、テストCSVの末尾に新しく
    # 追加された(=学習・これまでの検証のどちらにも使われていない)期間だけを
    # 対象に検証したい場合用。省略時は従来通りテストCSVの先頭(dataset.seq_length)
    # から検証する。--n-splitsと併用した場合、分割もこの開始日以降の範囲だけを
    # 対象に行う。
    parser.add_argument("--test-from-date", type=str, default=None,
                         help="検証対象とする開始日時(例: 2026-08-27)。省略時はテストCSVの先頭から。"
                              "test CSVの末尾に新しい期間を追加した場合に、その新しい部分だけを"
                              "対象に検証するために使う")
    args = parser.parse_args()

    # 💡 追加: ログファイル・グラフの保存先を先に決定してからロガーを初期化することで、
    # 以降の全ログ出力(モデルロード状況・ステップごとの資産推移など)がファイルにも残る。
    results_dir = get_test_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(results_dir, f"test_portfolio_{args.mode}_{timestamp}.log")
    setup_logger(log_file_path)

    logging.info("test_portfolio.py ver213.0")
    if results_dir == DRIVE_TEST_RESULTS_DIR:
        logging.info(f"Google Driveを検出。テスト結果の保存先をDriveに設定します: {results_dir}")
    else:
        logging.warning(
            "Google Driveが未マウントのため、テスト結果はColabのローカルディスクに保存されます。"
            "セッション切断でVMごと失われる可能性があるので注意してください。"
        )
    logging.info(f"📝 ログをファイルにも保存します: {log_file_path}")
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

    # 💡 追加(2026-11-06): --test-from-date が指定されていれば、その日時以降の
    # 最初の行をエピソード開始位置にする(train/testの境界日自体は変更しない。
    # test CSVの末尾に新しい期間を追加した場合、その新しい部分だけを対象に
    # 検証するためのもの)。
    episode_start_idx = None
    if args.test_from_date:
        from_ts = pd.to_datetime(args.test_from_date)
        matching_idx = dataset.df.index[dataset.df["time"] >= from_ts]
        if len(matching_idx) == 0:
            logging.error(
                f"❌ --test-from-date で指定された日時以降のデータがテストCSV内に"
                f"見つかりません: {args.test_from_date}"
            )
            return
        episode_start_idx = max(int(matching_idx[0]), dataset.seq_length)
        logging.info(
            f"📅 --test-from-date={args.test_from_date} が指定されたため、"
            f"{dataset.df['time'].iloc[episode_start_idx]} 以降のみを検証対象にします"
            f"(全{len(dataset.df)}行中、{len(dataset.df) - episode_start_idx}行が対象)"
        )

    # 💡 修正: PortfolioFXEnv は prob_0/1/2・risk_val を precompute_portfolio.py が
    # 事前計算した列から直接読み込む設計(train_portfolio_rl.py と同じ)で、
    # アナリストモデルを直接渡す trained_model 引数はそもそも存在しない。
    # 旧コードはここでTypeErrorになっており、かつロード済みの analyzer もどこにも
    # 使われていなかった(死んだコード)ため、丸ごと削除した。

    #テスト用環境の構築とvecnormalizeの復元
    env = build_env(dataset, args, episode_start_idx=episode_start_idx)
    vec_env = load_vec_env(env, vec_norm_path)
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

    # 💡 追加: 決済(realized_pnl_by_symbolが非ゼロになったステップ)ごとに
    # 銘柄・損益を記録し、ループ終了後に勝率・プロフィットファクターなどの
    # トレード単位の集計値を算出するために使う。
    trades = []

    step_count = 0


    done_reason = None

    while not done:
        action,_states = model.predict(obs,deterministic=True)
        obs,rewards,dones,infos = vec_env.step(action)

        done = dones[0]
        info = infos[0]

        equities.append(info["equity"])
        balances.append(info["balance"])

        realized = info.get("realized_pnl_by_symbol", {})
        for sym, pnl in realized.items():
            if pnl != 0:
                trades.append({"step": step_count + 1, "symbol": sym, "pnl": pnl})

        step_count += 1
        if step_count % 10 == 0:
            logging.info(f"Step {step_count:04d} | Equity: {info['equity']:,.0f} 円 | Balance: {info['balance']:,.0f} 円")
            # 💡 追加: 銘柄別の建玉・価格・実現/含み損益を出力する。
            # 資産の急変(急騰・急落)がどの銘柄に由来するか確認するため。
            positions = info.get("positions", {})
            prices = info.get("prices", {})
            unrealized = info.get("unrealized_pnl_by_symbol", {})
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

    # 💡 修正: 利益率・勝率・プロフィットファクターなどの集計ロジックを
    # backtest_metrics.py に共通化(sweep_entry_threshold.py と共有するため)。
    initial_balance = env.initial_balance
    elapsed_days = None
    if step_count > 0:
        start_idx = episode_start_idx if episode_start_idx is not None else dataset.seq_length
        end_idx = min(start_idx + step_count - 1, len(dataset.df) - 1)
        start_ts = pd.to_datetime(dataset.df["time"].iloc[start_idx])
        end_ts = pd.to_datetime(dataset.df["time"].iloc[end_idx])
        elapsed_days = (end_ts - start_ts).total_seconds() / 86400.0

    summary = compute_performance_summary(equities, balances, trades, initial_balance, elapsed_days)
    log_performance_summary(summary)

    # 💡 追加(2026-11-06): 資産推移を後から可視化・分析できるよう、ステップごとの
    # 生データ(時刻・Equity・Balance)もCSVに保存する(下のPNGグラフは静止画のみで、
    # 後から詳しく見返したり自分でグラフを作り直したりすることができないため)。
    if step_count > 0:
        start_idx = episode_start_idx if episode_start_idx is not None else dataset.seq_length
        times = dataset.df["time"].iloc[start_idx: start_idx + step_count].reset_index(drop=True)
        equity_curve_path = os.path.join(results_dir, f"portfolio_equity_curve_{args.mode}_{timestamp}.csv")
        equity_df = pd.DataFrame({
            "step": range(1, step_count + 1),
            "time": times,
            "equity": equities,
            "balance": balances,
        })
        equity_df.to_csv(equity_curve_path, index=False)
        logging.info(f"📈 資産推移の生データをCSVにも保存しました: {equity_curve_path}")

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

        save_path = os.path.join(results_dir, f"portfolio_test_result_{args.mode}_{timestamp}.png")
        plt.savefig(save_path)
        logging.info(f"🖼️ 資産推移のグラフを保存しました: {save_path}")
    except Exception as e:
        logging.error(f"グラフの生成に失敗しました。:{e}")

    # 💡 追加(2026-11-06): --n-splits > 1 の場合、テスト期間全体の単発バックテストに
    # 加えて、期間を分割した再現性確認も実行する。
    if args.n_splits > 1:
        run_split_validation(dataset, model, vec_norm_path, args, results_dir, timestamp, episode_start_idx=episode_start_idx)


if __name__ == "__main__":
    main()
