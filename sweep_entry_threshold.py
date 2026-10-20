"""
動的エントリー閾値(entry_threshold_window / entry_threshold_percentile)の
パラメータ探索スクリプト。

目的:
    portfolio_trading_bot.py / portfolio_env.py の固定エントリー閾値(0.2)を、
    銘柄ごとの直近の行動分布に基づく動的閾値に置き換える変更を検証するため、
    学習済みモデル(再学習なし)をそのまま使い、複数の(window, percentile)の
    組み合わせでバックテストを繰り返し、月利換算・最大ドローダウン・勝率・
    プロフィットファクター・トレード数を比較する。

重要:
    モデルの再学習は一切行わない(model.predict()による推論のみ)。
    同時保有上限(max_concurrent_positions)はデフォルトで4に固定している
    (急な値動きに対する同時エクスポージャーを抑える安全策)。この値自体を
    変えて試したい場合は --max-concurrent-positions で指定できる。

使い方:
    python sweep_entry_threshold.py --mode medium
    python sweep_entry_threshold.py --mode medium --windows 100,200,300,500 --percentiles 40,50,60,70,80
"""

import os
import csv
import logging
import argparse
import itertools

import torch
import pandas as pd
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from portfolio_dataset import PortfolioFXDataset
from portfolio_env import PortfolioFXEnv
from rl_checkpoint_utils import resolve_model_paths
from backtest_metrics import compute_performance_summary, run_backtest_episode

DRIVE_TEST_RESULTS_DIR = "/content/drive/MyDrive/FX_AI_Models/test_results"
LOCAL_TEST_RESULTS_DIR = "test_results"


def get_test_results_dir():
    if os.path.exists("/content/drive/MyDrive"):
        results_dir = DRIVE_TEST_RESULTS_DIR
    else:
        results_dir = LOCAL_TEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_one_combo(dataset, symbols, vec_norm_path, model, window, percentile, max_concurrent):
    env = PortfolioFXEnv(
        dataset=dataset, symbols=symbols, fixed_start=True,
        entry_threshold_window=window,
        entry_threshold_percentile=percentile,
        max_concurrent_positions=max_concurrent,
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    result = run_backtest_episode(vec_env, model)

    elapsed_days = None
    if result["step_count"] > 0:
        start_idx = dataset.seq_length
        end_idx = min(start_idx + result["step_count"] - 1, len(dataset.df) - 1)
        start_ts = pd.to_datetime(dataset.df["time"].iloc[start_idx])
        end_ts = pd.to_datetime(dataset.df["time"].iloc[end_idx])
        elapsed_days = (end_ts - start_ts).total_seconds() / 86400.0

    summary = compute_performance_summary(
        result["equities"], result["balances"], result["trades"], env.initial_balance, elapsed_days
    )
    summary["entry_threshold_window"] = window
    summary["entry_threshold_percentile"] = percentile
    summary["max_concurrent_positions"] = max_concurrent
    summary["done_reason"] = result["done_reason"]
    summary["step_count"] = result["step_count"]
    return summary


def main():
    setup_logger()
    logging.info("sweep_entry_threshold.py ver1.0")

    parser = argparse.ArgumentParser(description="動的エントリー閾値のパラメータ探索(再学習なし)")
    parser.add_argument("--symbols", default=["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"])
    parser.add_argument("--mode", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--windows", type=str, default="100,200,300,500",
                         help="ローリングウィンドウのサンプル数(銘柄ごと)。カンマ区切りで複数指定")
    parser.add_argument("--percentiles", type=str, default="40,50,60,70,80",
                         help="動的閾値に使うパーセンタイル。カンマ区切りで複数指定")
    parser.add_argument("--max-concurrent-positions", type=int, default=4,
                         help="同時保有できる最大銘柄数(安全策)。全組み合わせに共通で適用")
    parser.add_argument("--target-monthly-return", type=float, default=5.0)
    parser.add_argument("--target-max-drawdown", type=float, default=20.0)
    parser.add_argument("--min-trades", type=int, default=10,
                         help="推奨候補とみなす最低トレード数(サンプル数が少なすぎる組み合わせを除外する)")
    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    percentiles = [float(p) for p in args.percentiles.split(",") if p.strip()]

    test_csv = "portfolio_merged_data_test_with_preds.csv"
    img_dir = "images"
    if not os.path.exists(test_csv):
        logging.error(f"✖テストデータ{test_csv}が見つかりません。test_portfolio.pyと同じ前準備が必要です。")
        return

    rl_model_path, vec_norm_path, _replay_buffer_path, model_source = resolve_model_paths(args.mode)
    if rl_model_path is None:
        logging.error("❌ SACモデルが見つかりません。学習が完了しているか確認してください。")
        return
    logging.info(f"📦 使用するモデル: {model_source} ({rl_model_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("テスト用データセットを読み込んでいます....")
    dataset = PortfolioFXDataset(merged_csv_path=test_csv, base_img_dir=img_dir, symbols=args.symbols, mode=args.mode)
    model = SAC.load(rl_model_path, device=device)
    logging.info("✅ モデル・データセットの読み込み完了。探索を開始します。")

    results = []

    # 💡 比較の基準となる、現行の固定閾値0.2・同時保有無制限(変更なし)のベースライン。
    logging.info("--- baseline(現行の固定閾値0.2・同時保有無制限)を実行中 ---")
    baseline = run_one_combo(dataset, args.symbols, vec_norm_path, model, window=None, percentile=None, max_concurrent=None)
    results.append(baseline)
    logging.info(
        f"baseline: trades={baseline['total_trades']} win_rate={baseline['win_rate_pct']:.1f}% "
        f"monthly_return={baseline['monthly_return_pct']} max_dd={baseline['max_drawdown_pct']:.2f}%"
    )

    combos = list(itertools.product(windows, percentiles))
    for idx, (window, percentile) in enumerate(combos, start=1):
        logging.info(f"--- [{idx}/{len(combos)}] window={window} percentile={percentile} を実行中 ---")
        summary = run_one_combo(
            dataset, args.symbols, vec_norm_path, model,
            window=window, percentile=percentile, max_concurrent=args.max_concurrent_positions,
        )
        results.append(summary)
        monthly = summary["monthly_return_pct"]
        monthly_str = f"{monthly:+.2f}%" if monthly is not None else "N/A"
        logging.info(
            f"  trades={summary['total_trades']} win_rate={summary['win_rate_pct']:.1f}% "
            f"profit_factor={summary['profit_factor']:.2f} monthly_return={monthly_str} "
            f"max_dd={summary['max_drawdown_pct']:.2f}%"
        )

    # 💡 CSVに保存(全組み合わせの比較表)
    results_dir = get_test_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"sweep_entry_threshold_{args.mode}_{timestamp}.csv")
    fieldnames = [
        "entry_threshold_window", "entry_threshold_percentile", "max_concurrent_positions",
        "total_trades", "win_rate_pct", "profit_factor", "total_return_pct", "monthly_return_pct",
        "max_drawdown_pct", "net_realized_pnl", "done_reason", "step_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    logging.info(f"📄 全組み合わせの比較結果を保存しました: {csv_path}")

    # 💡 目標(月刉5%・最大DD20%)を満たす組み合わせの中から、月利が最も高いものを推奨する。
    # サンプル数が少なすぎる(--min-trades未満)組み合わせは、偶然の結果を避けるため除外する。
    candidates = [
        r for r in results
        if r["entry_threshold_window"] is not None  # baselineは比較対象外
        and r["monthly_return_pct"] is not None
        and r["max_drawdown_pct"] <= args.target_max_drawdown
        and r["total_trades"] >= args.min_trades
    ]
    logging.info("=" * 70)
    logging.info("🏁 探索完了")
    logging.info("=" * 70)
    logging.info(
        f"baseline(現行): trades={baseline['total_trades']} "
        f"monthly_return={baseline['monthly_return_pct']} max_dd={baseline['max_drawdown_pct']:.2f}%"
    )
    if not candidates:
        logging.warning(
            f"⚠️ 目標(最大DD{args.target_max_drawdown}%以下・トレード数{args.min_trades}件以上)を"
            f"満たす組み合わせが見つかりませんでした。windows/percentilesの範囲を広げて再実行してください。"
        )
    else:
        best = max(candidates, key=lambda r: r["monthly_return_pct"])
        logging.info(
            f"✅ 推奨設定: window={best['entry_threshold_window']} "
            f"percentile={best['entry_threshold_percentile']} "
            f"max_concurrent_positions={best['max_concurrent_positions']}"
        )
        logging.info(
            f"   月利換算: {best['monthly_return_pct']:+.2f}% (目標: {args.target_monthly_return}%) / "
            f"最大DD: {best['max_drawdown_pct']:.2f}% (目標: {args.target_max_drawdown}%以下) / "
            f"トレード数: {best['total_trades']} / 勝率: {best['win_rate_pct']:.1f}% / "
            f"プロフィットファクター: {best['profit_factor']:.2f}"
        )
        if best["monthly_return_pct"] < args.target_monthly_return:
            logging.warning(
                "⚠️ この候補でも月利目標には届いていません。windows/percentilesの範囲を広げる、"
                "または別の改善案(非RLモデルの併用等)の検討をおすすめします。"
            )
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
