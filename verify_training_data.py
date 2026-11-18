"""
train.pyを実行する前に、labeled_data/*_{mode}_scaled.csvと画像フォルダが正しく
準備されているかを安価に確認する。学習データの読み込み・画像検証には時間が
かかるため、実際に学習を始める前にこのスクリプトで明らかな問題(銘柄間での
特徴量の列構成の不一致・終了日のズレ=データ更新漏れ・画像フォルダの欠落等)を
先に検出し、学習コストを無駄にしないようにする。

使い方:
  python verify_training_data.py --mode medium
  python verify_training_data.py --mode medium --exclude-symbols GOLD
"""
import argparse
import logging
import os
import sys

import pandas as pd

from dataset import EXCLUDE_COLS, HOLD_RETENTION_PERCENTILE, compute_dynamic_window, compute_next_action_idx

EXPECTED_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def check_symbol(symbol, mode):
    """
    1銘柄分のデータを確認し、(エラーのリスト, 情報dict) を返す。
    情報dictはこの銘柄の確認結果が無い場合はNone。
    """
    errors = []
    csv_path = f"labeled_data/{symbol}_{mode}_scaled.csv"
    img_dir = f"images/{symbol}/{mode}_MTF"

    if not os.path.exists(csv_path):
        errors.append(f"❌ [{symbol}] CSVが見つかりません: {csv_path}")
        return errors, None

    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "target_class" not in df.columns:
        errors.append(f"❌ [{symbol}] 想定している列(time・target_class)が見つかりません: {csv_path}")
        return errors, None
    df["time"] = pd.to_datetime(df["time"])

    if not os.path.isdir(img_dir):
        errors.append(f"❌ [{symbol}] 画像ディレクトリが見つかりません: {img_dir}")
        n_images = 0
    else:
        n_images = sum(1 for e in os.scandir(img_dir) if e.name.endswith(".png"))
        if n_images == 0:
            errors.append(f"❌ [{symbol}] 画像ディレクトリに画像が1件もありません: {img_dir}")

    target_classes = df["target_class"].values
    next_action_idx = compute_next_action_idx(target_classes)
    window, n_hold_gaps = compute_dynamic_window(target_classes, next_action_idx)
    label_counts = df["target_class"].value_counts(normalize=True)

    info = {
        "rows": len(df),
        "start": df["time"].iloc[0],
        "end": df["time"].iloc[-1],
        "n_images": n_images,
        "feature_cols": frozenset(df.columns) - set(EXCLUDE_COLS),
        "window": window,
        "n_hold_gaps": n_hold_gaps,
        "hold_pct": label_counts.get(0, 0.0) * 100,
        "sell_pct": label_counts.get(-1, 0.0) * 100,
        "buy_pct": label_counts.get(1, 0.0) * 100,
    }
    return errors, info


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="train.py実行前のデータ整合性チェック")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--exclude-symbols", type=str, default="",
                         help="チェック対象から除外する銘柄(train.pyの--exclude-symbolsと合わせる)")
    parser.add_argument("--max-date-gap-days", type=float, default=3.0,
                         help="銘柄間で終了日がこの日数以上ズレていたらエラーにする"
                              "(データ更新漏れの検出用)")
    args = parser.parse_args()

    exclude_symbols = {s.strip() for s in args.exclude_symbols.split(",") if s.strip()}
    target_symbols = [s for s in EXPECTED_SYMBOLS if s not in exclude_symbols]
    if exclude_symbols:
        logging.info(f"🚫 以下の銘柄は確認対象から除外します: {sorted(exclude_symbols)}")

    all_errors = []
    per_symbol = {}
    for symbol in target_symbols:
        errors, info = check_symbol(symbol, args.mode)
        all_errors.extend(errors)
        if info is not None:
            per_symbol[symbol] = info

    if not per_symbol:
        logging.error("❌ 確認できた銘柄が1つもありません。学習を開始できません。")
        sys.exit(1)

    # --- 特徴量の列構成が銘柄間で一致しているか(input_dimのズレを事前に検出) ---
    all_col_sets = {v["feature_cols"] for v in per_symbol.values()}
    if len(all_col_sets) > 1:
        most_common_cols = max(all_col_sets, key=lambda cols: sum(1 for v in per_symbol.values() if v["feature_cols"] == cols))
        all_errors.append(
            "❌ 銘柄間で特徴量の列構成が一致していません(train.pyのinput_dimがズレます)。"
            "多数派の構成との差分:"
        )
        for symbol, v in per_symbol.items():
            diff = v["feature_cols"] ^ most_common_cols
            if diff:
                all_errors.append(f"    [{symbol}] 列数={len(v['feature_cols'])} 差分列={sorted(diff)}")

    # --- 終了日が銘柄間で大きくズレていないか(データ更新漏れの検出) ---
    latest_end = max(v["end"] for v in per_symbol.values())
    for symbol, v in per_symbol.items():
        gap_days = (latest_end - v["end"]).total_seconds() / 86400.0
        if gap_days > args.max_date_gap_days:
            all_errors.append(
                f"❌ [{symbol}] 終了日が他銘柄より{gap_days:.1f}日古いです"
                f"({v['end']} 対 最新{latest_end})。データ更新が反映されていない可能性があります。"
            )

    # --- 結果表示 ---
    logging.info("=" * 70)
    logging.info(f"📋 学習前データ確認結果(mode={args.mode})")
    logging.info("=" * 70)
    for symbol in sorted(per_symbol.keys()):
        v = per_symbol[symbol]
        logging.info(
            f"[{symbol}] 行数={v['rows']:,} 期間={v['start']} 〜 {v['end']} 画像={v['n_images']:,}件"
        )
        logging.info(
            f"    ラベル比率: HOLD={v['hold_pct']:.1f}% SELL={v['sell_pct']:.1f}% BUY={v['buy_pct']:.1f}%"
            f" / 動的ウィンドウ(HOLD{HOLD_RETENTION_PERCENTILE}%ile)={v['window']}本"
            f"(HOLD行{v['n_hold_gaps']:,}件から算出)"
        )

    if all_errors:
        logging.info("-" * 70)
        logging.error(f"❌ 問題が{len(all_errors)}件見つかりました。学習を開始する前に確認してください:")
        for e in all_errors:
            logging.error(f"  {e}")
        sys.exit(1)

    logging.info("-" * 70)
    logging.info("✅ 問題は見つかりませんでした。学習を開始できます。")


if __name__ == "__main__":
    main()
