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


def _list_dir_entries(path):
    with os.scandir(path) as it:
        return {entry.name: entry for entry in it}


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


def main():
    logging.info("diagnose_missing_images.py ver1.0")
    parser = argparse.ArgumentParser(
        description="画像欠損の詳細診断(precompute_portfolio.pyのcheck_images_completeと"
                    "同じ判定基準で、全件の分布・他銘柄との相関を出力する読み取り専用ツール)"
    )
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--seq-length", type=int, default=40)
    parser.add_argument("--max-blocks", type=int, default=10, help="銘柄ごとに表示する連続欠損ブロックの最大数")
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
    for sym in symbols:
        folder = os.path.join(img_dir, sym, f"{args.mode}_MTF")
        logging.info(f"[{sym}] フォルダを列挙中: {folder}")
        entries = _with_drive_retry(_list_dir_entries, folder)

        missing_idx = []
        for i, t in enumerate(time_strs):
            filename = f"{t}.png"
            entry = entries.get(filename)
            is_present = False
            if entry is not None:
                try:
                    is_present = entry.stat().st_size > 0
                except OSError:
                    is_present = False
            if not is_present:
                missing_idx.append(i)

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


if __name__ == "__main__":
    main()
