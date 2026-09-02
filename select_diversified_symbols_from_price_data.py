"""
select_diversified_symbols_from_price_data.py
------------------------------------------------
Run this LOCALLY on portfolio_merged_data_train_with_preds.csv (no need
to upload the 1.1GB file anywhere). It:

  1. Peeks at the CSV header only (no full load) to find the close-price
     columns for your 7 symbols at the chosen timeframe.
  2. Loads ONLY those columns (usecols) as float32, so a 1.1GB file with
     many extra columns (predictions, ATR, spread, OHLC, etc.) becomes a
     much smaller in-memory table -- typically well under 200MB even for
     millions of rows.
  3. Computes simple returns (pct_change) per symbol and the resulting
     correlation matrix -- this is the REAL market correlation, replacing
     the rough proxies used earlier (structural currency-overlap /
     trade-PnL correlation).
  4. Runs the same two subset-selection algorithms discussed earlier:
       - minimum average pairwise correlation (exact, brute-force; only
         C(7,k) combinations, so this is trivial to compute exactly)
       - maximum-determinant correlation submatrix (D-optimal
         diversification criterion)
     for k = 3, 4, 5, and prints/saves the best subsets.
  5. Saves a correlation heatmap + hierarchical-clustering dendrogram
     (PNG) and the full correlation matrix (CSV) so you can inspect them
     without re-running.

USAGE
-----
    python select_diversified_symbols_from_price_data.py \
        --csv portfolio_merged_data_train_with_preds.csv \
        --mode short \
        --outdir ./diversification_results

Arguments:
    --csv     Path to portfolio_merged_data_train_with_preds.csv
    --mode    "short" (M5), "medium" (M15), or "long" (H4) -- matches the
              base_tf logic in portfolio_env.py. Default: short
    --symbols Comma-separated symbol list. Default matches train_portfolio_rl.py:
              GBPUSD,EURUSD,USDJPY,AUDUSD,GBPJPY,EURJPY,GOLD
    --outdir  Where to save results. Default: ./diversification_results

If your CSV uses different column names than "{SYMBOL}_{TF}_close", edit
CLOSE_COL_TEMPLATE below (or pass --col-template).
"""

import argparse
import itertools
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

CLOSE_COL_TEMPLATE = "{sym}_{tf}_close"

MODE_TO_TF = {"short": "M5", "medium": "M15", "long": "H4"}


def parse_args():
    p = argparse.ArgumentParser(description="Find the most diversified symbol subset from real price data.")
    p.add_argument("--csv", required=True, help="Path to portfolio_merged_data_train_with_preds.csv")
    p.add_argument("--mode", default="short", choices=["short", "medium", "long"],
                    help="Matches base_tf logic in portfolio_env.py (default: short -> M5)")
    p.add_argument("--symbols", default="GBPUSD,EURUSD,USDJPY,AUDUSD,GBPJPY,EURJPY,GOLD",
                    help="Comma-separated symbol list")
    p.add_argument("--outdir", default="./diversification_results")
    p.add_argument("--col-template", default=CLOSE_COL_TEMPLATE,
                    help='Column naming pattern, e.g. "{sym}_{tf}_close"')
    p.add_argument("--date-col", default=None,
                    help="Optional datetime column name, to sort/dedupe by time before computing returns")
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    tf = MODE_TO_TF[args.mode]

    import os
    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Peek header only (no data load) to build the exact column list
    # -------------------------------------------------------------
    print(f"Reading header of {args.csv} ...")
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()

    close_cols = {}
    missing = []
    for sym in symbols:
        col = args.col_template.format(sym=sym, tf=tf)
        if col in header:
            close_cols[sym] = col
        else:
            missing.append(col)

    if missing:
        print("WARNING: the following expected columns were not found in the CSV header:")
        for m in missing:
            print(f"   - {m}")
        print("Available columns containing 'close' (first 40 shown):")
        close_like = [c for c in header if "close" in c.lower()][:40]
        for c in close_like:
            print(f"   - {c}")
        if len(close_cols) < 3:
            print("\nToo few symbol columns matched -- adjust --col-template or --mode and retry.")
            sys.exit(1)

    usecols = list(close_cols.values())
    if args.date_col and args.date_col in header:
        usecols = [args.date_col] + usecols

    print(f"Loading {len(usecols)} columns only (out of {len(header)} total) ...")

    # -------------------------------------------------------------
    # 2. Load only the needed columns, as float32 to save memory
    # -------------------------------------------------------------
    dtype_map = {c: np.float32 for c in close_cols.values()}
    df = pd.read_csv(args.csv, usecols=usecols, dtype=dtype_map)
    print(f"Loaded shape: {df.shape}  (~{df.memory_usage(deep=True).sum()/1e6:.1f} MB in memory)")

    if args.date_col and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.sort_values(args.date_col)

    price_df = df[[close_cols[s] for s in symbols if s in close_cols]].rename(
        columns={v: k for k, v in close_cols.items()}
    )
    available_symbols = list(price_df.columns)

    # -------------------------------------------------------------
    # 3. Returns & correlation matrix
    # -------------------------------------------------------------
    returns = price_df.diff().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna()
    print(f"Computed returns on {len(returns)} rows (after dropping NaN/inf).")

    corr = returns.corr()
    corr.to_csv(f"{args.outdir}/price_return_correlation.csv")
    print("\n" + "=" * 70)
    print(f"REAL price-return correlation matrix ({tf} timeframe, mode={args.mode})")
    print("=" * 70)
    print(corr.round(3))

    # -------------------------------------------------------------
    # 4. Subset-selection algorithms (same as the preliminary analysis)
    # -------------------------------------------------------------
    def select_best_subset(corr_matrix, k, top_n=3):
        results = []
        for combo in itertools.combinations(available_symbols, k):
            sub = corr_matrix.loc[list(combo), list(combo)]
            n = len(combo)
            avg_corr = (sub.values.sum() - n) / (n * n - n)
            results.append((combo, avg_corr))
        results.sort(key=lambda x: x[1])
        return results[:top_n]

    def max_det_subset(corr_matrix, k, top_n=3):
        results = []
        for combo in itertools.combinations(available_symbols, k):
            sub = corr_matrix.loc[list(combo), list(combo)].values
            det = np.linalg.det(sub)
            results.append((combo, det))
        results.sort(key=lambda x: -x[1])
        return results[:top_n]

    report_lines = []
    for k in [3, 4, 5]:
        if k > len(available_symbols):
            continue
        report_lines.append(f"\n{'='*70}\nBEST {k}-SYMBOL SUBSETS (k={k})\n{'='*70}")
        report_lines.append(f"\nTop {k}-subsets by MIN average correlation (lower = more diversified):")
        for combo, val in select_best_subset(corr, k):
            report_lines.append(f"  {combo}  avg_corr={val:.4f}")
        report_lines.append(f"\nTop {k}-subsets by MAX determinant (D-optimal, higher = more diversified):")
        for combo, val in max_det_subset(corr, k):
            report_lines.append(f"  {combo}  det={val:.4f}")

    report_text = "\n".join(report_lines)
    print(report_text)
    with open(f"{args.outdir}/subset_selection_report.txt", "w") as f:
        f.write(report_text)

    # -------------------------------------------------------------
    # 5. Heatmap + dendrogram
    # -------------------------------------------------------------
    dist = (1 - corr.clip(-1, 1)).to_numpy().copy()
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax0 = axes[0]
    im = ax0.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax0.set_xticks(range(len(available_symbols))); ax0.set_xticklabels(available_symbols, rotation=45, ha="right")
    ax0.set_yticks(range(len(available_symbols))); ax0.set_yticklabels(available_symbols)
    for i in range(len(available_symbols)):
        for j in range(len(available_symbols)):
            ax0.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax0.set_title(f"Real price-return correlation ({tf})")
    plt.colorbar(im, ax=ax0, fraction=0.046)

    ax1 = axes[1]
    dendrogram(Z, labels=available_symbols, ax=ax1)
    ax1.set_title("Hierarchical clustering (correlation distance)")
    ax1.set_ylabel("1 - correlation")

    plt.tight_layout()
    out_png = f"{args.outdir}/price_diversification.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved chart to {out_png}")
    print(f"Saved correlation matrix to {args.outdir}/price_return_correlation.csv")
    print(f"Saved subset report to {args.outdir}/subset_selection_report.txt")


if __name__ == "__main__":
    main()