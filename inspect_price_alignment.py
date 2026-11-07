import argparse
import sys
import numpy as np
import pandas as pd
 
MODE_TO_TF = {"short": "M5", "medium": "M15", "long": "H4"}
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Cheap diagnostic for price-data time alignment.")
    p.add_argument("--csv", required=True)
    p.add_argument("--mode", default="short", choices=["short", "medium", "long"])
    p.add_argument("--symbols", default="GBPUSD,EURUSD,USDJPY,AUDUSD,GBPJPY,EURJPY,GOLD")
    p.add_argument("--col-template", default="{sym}_{tf}_close")
    p.add_argument("--date-col", default=None, help="Datetime column name; auto-detected if omitted")
    p.add_argument("--n", type=int, default=3000, help="Sample size for head/middle checks")
    p.add_argument("--mid-start", type=int, default=200000,
                    help="Row offset to start the 'middle' sample from (data rows, not counting header)")
    return p.parse_args()
 
 
def load_sample(csv, usecols, n, skip_start=0):
    if skip_start == 0:
        return pd.read_csv(csv, usecols=usecols, nrows=n)
    # keep header (row 0), skip data rows [1, skip_start], then read n rows
    skiprows = range(1, skip_start + 1)
    return pd.read_csv(csv, usecols=usecols, skiprows=skiprows, nrows=n)
 
 
def quick_corr_report(sample, symbols, label):
    returns = sample[symbols].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20:
        print(f"[{label}] too few usable rows ({len(returns)}) to compute correlation.")
        return None
    corr = returns.corr()
    offdiag = corr.values[~np.eye(len(symbols), dtype=bool)]
    print(f"\n[{label}] rows used for returns: {len(returns)}")
    print(f"[{label}] correlation matrix:")
    print(corr.round(3))
    print(f"[{label}] mean |correlation| across all pairs: {np.abs(offdiag).mean():.4f}")
    print(f"[{label}] max |correlation|: {np.abs(offdiag).max():.4f}")
    return corr
 
 
def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    tf = MODE_TO_TF[args.mode]
 
    print(f"Reading header of {args.csv} ...")
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
 
    close_cols = {}
    for sym in symbols:
        col = args.col_template.format(sym=sym, tf=tf)
        if col in header:
            close_cols[sym] = col
 
    if len(close_cols) < 2:
        print("Could not find at least 2 matching close columns. Check --col-template / --mode.")
        print("Columns containing 'close':", [c for c in header if "close" in c.lower()][:30])
        sys.exit(1)
 
    date_col = args.date_col
    if date_col is None:
        candidates = [c for c in header if any(k in c.lower() for k in ["date", "time", "datetime"])]
        date_col = candidates[0] if candidates else None
        if date_col:
            print(f"Auto-detected datetime column: '{date_col}'")
        else:
            print("No datetime column auto-detected -- proceeding with row-order only "
                  "(alignment check on timestamps will be skipped).")
 
    usecols = list(close_cols.values()) + ([date_col] if date_col else [])
    symbol_cols = list(close_cols.values())
    rename_map = {v: k for k, v in close_cols.items()}
 
    # -----------------------------------------------------------
    # 1. HEAD sample
    # -----------------------------------------------------------
    head = load_sample(args.csv, usecols, args.n, skip_start=0).rename(columns=rename_map)
    print("\n" + "=" * 70)
    print(f"HEAD SAMPLE (first {len(head)} rows) -- eyeball a few rows:")
    print("=" * 70)
    print(head.head(15).to_string())
 
    # -----------------------------------------------------------
    # 2. MIDDLE sample (far from the start, still cheap to read)
    # -----------------------------------------------------------
    mid = load_sample(args.csv, usecols, args.n, skip_start=args.mid_start).rename(columns=rename_map)
    print("\n" + "=" * 70)
    print(f"MIDDLE SAMPLE (rows {args.mid_start}-{args.mid_start+len(mid)}):")
    print("=" * 70)
    print(mid.head(15).to_string())
 
    # -----------------------------------------------------------
    # 3. Timestamp cadence check
    # -----------------------------------------------------------
    if date_col:
        for label, sample in [("HEAD", head), ("MIDDLE", mid)]:
            try:
                ts = pd.to_datetime(sample[date_col], errors="coerce")
                deltas = ts.diff().dropna()
                mode_delta = deltas.mode()
                n_jumps = (deltas != mode_delta.iloc[0]).sum() if len(mode_delta) else None
                print(f"\n[{label}] timestamp step mode: {mode_delta.iloc[0] if len(mode_delta) else 'N/A'}"
                      f"  | rows with a DIFFERENT step: {n_jumps} / {len(deltas)}")
            except Exception as e:
                print(f"[{label}] could not parse '{date_col}' as datetime: {e}")
 
    # -----------------------------------------------------------
    # 4. Quick correlation smoke test on both samples
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("QUICK CORRELATION SMOKE TEST (small sample -- real FX pairs should")
    print("NOT be ~0 across the board; expect several pairs > 0.2-0.3)")
    print("=" * 70)
    quick_corr_report(head, list(close_cols.keys()), "HEAD")
    quick_corr_report(mid, list(close_cols.keys()), "MIDDLE")
 
    print("\nDone. If both samples show near-zero correlation for every pair,")
    print("that strongly suggests a time-alignment bug in how the merged CSV")
    print("was built -- investigate the merge/join step before trusting any")
    print("correlation-based analysis on the full file.")
 
 
if __name__ == "__main__":
    main()
 