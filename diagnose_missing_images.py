import argparse
import os
import errno
import time
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 💡 precompute_portfolio.py と同じリトリイ方針(Errno 107/5対策)。
DRIVE_IO_MAX_RETRIES = 5
DRIVE_IO_RETRY_BACKOFF_BASE = 10


def _with_drive_retry(func, *args, **kwargs):
    last_err = None
    for attempt in range(DRIVE_IO_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno not in (errno.ENOTCONN, errno.EIO):
                raise
            last_err = e
            wait = DRIVE_IO_RETRY_BACKOFF_BASE * (2 ** attempt)
            logging.warning(
                f"⚠️ Google Driveへの接続が一時的に切断されたようです: {e}"
                f" -- {wait}秒後に再試行します({attempt + 1}/{DRIVE_IO_MAX_RETRIES})"
            )
            time.sleep(wait)
    raise last_err


def _list_dir_names(path):
    """
    💡 修正(2026-11-03の調査で判明した問題への対応): 以前は本関数がos.scandir()で
    DirEntryを返し、呼び出し元が各ファイルごとにentry.stat()を呼んでサイズ(>0)まで
    検証していた。しかしmediumモードの全期間では90,000件超×7銘柄規模のstat呼び出しが
    発生し、precompute_portfolio.py側でこのチェック自体がGoogle Drive側のレート制限を
    悪化させ、直後の推論ループまで連鎖的に遅くなる事例が発生した。本診断ツールも同じ
    負荷をDriveにかけていたため、同様にファイル名の一覧取得(os.listdir)のみに留める。
    """
    return set(os.listdir(path))


def group_contiguous(sorted_indices):
    """連続するインデックス(=master timeline上で隣り合う欠損)をブロック化する"""
    if not sorted_indices:
        return []
    groups = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def sample_evenly(items, k):
    """履歴全体に均等に広がったサンプルを取るための補助関数(先頭だけに偏らないように)"""
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def check_raw_source_existence(symbol, mode, phase, missing_time_strs):
    """
    💡 追加(2026-11-02の調査で判明した問題への対応): 欠損率の高い銘柄について、
    「画像生成の不具合」なのか「そもそも元データ(ローソシー足)自体が存在しない時間帯」
    なのかを切り分ける。split_data.py / merge_portfolio_data.py が参照する生ラベル
    CSV(labeled_data/{symbol}_{mode}_labeled[_{phase}].csv)を直接読み、画像が欠損
    している時刻が、その銘柄自身の生データにそもそも存在するかどうかを確認する。
    存在しない(=元データの時点で欠落している)割合が高ければ、画像生成の問題ではなく、
    その銘柄固有の市場休止時間(例: コモディティの日次ロールオーバー等)による、
    データ自体の欠落である可能性が高いと判断できる。
    """
    csv_dir = "labeled_data"
    candidates = [
        os.path.join(csv_dir, f"{symbol}_{mode}_labeled_{phase}.csv"),
        os.path.join(csv_dir, f"{symbol}_{mode}_labeled.csv"),
    ]
    raw_path = next((p for p in candidates if os.path.exists(p)), None)
    if raw_path is None:
        logging.info(
            f"[{symbol}] 生ラベルCSVが見つからないため、元データの存在確認はスキップします"
            f"(確認したパス: {candidates})"
        )
        return

    raw_df = _with_drive_retry(pd.read_csv, raw_path)
    raw_time_strs = set(pd.to_datetime(raw_df["time"]).dt.strftime("%Y%m%d_%H%M%S"))

    absent_from_raw = [t for t in missing_time_strs if t not in raw_time_strs]
    present_in_raw = [t for t in missing_time_strs if t in raw_time_strs]

    logging.info(
        f"[{symbol}] 元データ({raw_path})での存在確認: "
        f"サンプル{len(missing_time_strs)}件中、元データにも存在しない(=元から欠落): "
        f"{len(absent_from_raw)}件 ({len(absent_from_raw) / len(missing_time_strs) * 100:.1f}%)、"
        f"元データには存在する(=画像生成のみ失敗の疑い): {len(present_in_raw)}件"
    )
    if present_in_raw:
        logging.info(f"[{symbol}] 元データに存在するにも関わらず画像が無かった時刻の例: {present_in_raw[:10]}")


def main():
    logging.info("diagnose_missing_images.py ver1.2")
    parser = argparse.ArgumentParser(
        description="画像欠損の詳細診断(precompute_portfolio.pyのcheck_images_completeと"
                    "同じ判定基準で、全件の分布・他銘柄との相関・元データの存在有無を出力する"
                    "読み取り専用ツール)"
    )
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--seq-length", type=int, default=40)
    parser.add_argument("--max-blocks", type=int, default=10, help="銘柄ごとに表示する連続欠損ブロックの最大数")
    parser.add_argument("--raw-check-threshold-pct", type=float, default=0.5,
                         help="この欠損率(%%)を超える銘柄についてのみ、元データでの存在確認を追加で行う")
    parser.add_argument("--raw-check-samples", type=int, default=20,
                         help="元データでの存在確認を行う欠損タイムスタンプの件数(履歴全体から均等に抽出)")
    args = parser.parse_args()

    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    merged_csv = f"portfolio_merged_data_{args.phase}.csv"
    img_dir = "images"

    logging.info(f"読み込み中: {merged_csv}")
    df = _with_drive_retry(pd.read_csv, merged_csv)
    df["time_str"] = pd.to_datetime(df["time"]).dt.strftime("%Y%m%d_%H%M%S")
    time_strs = df["time_str"].values[args.seq_length - 1:]
    n = len(time_strs)
    logging.info(f"検証対象: {n}ステップ × {len(symbols)}銘柄")

    missing_sets = {}
    missing_idx_by_symbol = {}
    for sym in symbols:
        folder = os.path.join(img_dir, sym, f"{args.mode}_MTF")
        logging.info(f"[{sym}] フォルダを列挙中: {folder}")
        existing_names = _with_drive_retry(_list_dir_names, folder)

        missing_idx = []
        for i, t in enumerate(time_strs):
            filename = f"{t}.png"
            if filename not in existing_names:
                missing_idx.append(i)

        missing_idx_by_symbol[sym] = missing_idx
        missing_sets[sym] = set(time_strs[i] for i in missing_idx)
        pct = len(missing_idx) / n * 100 if n else 0.0
        logging.info(f"[{sym}] 欠損: {len(missing_idx)}/{n}件 ({pct:.3f}%)")

        if missing_idx:
            groups = group_contiguous(missing_idx)
            logging.info(f"[{sym}] 連続欠損ブロック数: {len(groups)}(多いほど「散らばっている」)")
            for g_start, g_end in groups[: args.max_blocks]:
                logging.info(
                    f"    {time_strs[g_start]} 〜 {time_strs[g_end]} ({g_end - g_start + 1}件)"
                )
            if len(groups) > args.max_blocks:
                logging.info(f"    ...他 {len(groups) - args.max_blocks} ブロック")

    logging.info("=" * 60)
    logging.info("銘柄間の欠損タイミングの相関を確認します(他銘柄も同時に欠損していれば全体的な問題、"
                 "その銘柄だけなら個別の問題を疑う)")
    for sym in symbols:
        if not missing_sets[sym]:
            continue
        others_also_missing = set()
        for t in missing_sets[sym]:
            for other in symbols:
                if other != sym and t in missing_sets[other]:
                    others_also_missing.add(t)
        overlap_pct = len(others_also_missing) / len(missing_sets[sym]) * 100
        logging.info(
            f"[{sym}] 欠損{len(missing_sets[sym])}件中、他の銘柄も同時に欠損しているもの: "
            f"{len(others_also_missing)}件 ({overlap_pct:.1f}%)"
        )

    logging.info("=" * 60)
    logging.info(
        f"欠損率が{args.raw_check_threshold_pct}%を超える銘柄について、元データでの存在確認を行います"
        "(画像生成の不具合か、そもそも元データが無い時間帯かを切り分けるため)"
    )
    any_checked = False
    for sym in symbols:
        missing_idx = missing_idx_by_symbol[sym]
        pct = len(missing_idx) / n * 100 if n else 0.0
        if pct <= args.raw_check_threshold_pct:
            continue
        any_checked = True
        sample_idx = sample_evenly(sorted(missing_idx), args.raw_check_samples)
        sample_time_strs = [time_strs[i] for i in sample_idx]
        check_raw_source_existence(sym, args.mode, args.phase, sample_time_strs)
    if not any_checked:
        logging.info(f"(閾値{args.raw_check_threshold_pct}%を超える銘柄はありませんでした)")


if __name__ == "__main__":
    main()
