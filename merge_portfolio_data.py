import pandas as pd
import numpy as np
import os
import glob
import logging
 
 
def setup_logger():
    logging.basicConfig(level=logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")
 
# portfolio_env.py の取引ロジック（エントリー価格・TP/SL・ATR・raw_pnl計算）に
# 実際に必要な、生の価格系の列だけを _labeled.csv から拾う。
# MACD/RSI/EMA等のモデル入力用テクニカル指標は、標準化済みの _scaled 側に
# 既にあるため、ここでは持ち込まない（ファイルサイズを不必要に膨らませないため）。
RAW_FIELD_SUFFIXES = ["open", "high", "low", "close", "spread", "ATR"]
 
# 学習ターゲット列は "close" 等のサフィックスに偶然マッチしてしまうため、
# 明示的に除外する（portfolio_dataset.py の exclude_exact と揃えてある）。
RAW_FIELD_EXCLUDE = {"future_close", "price_diff_pct", "target_class", "target_risk_pct"}
 
 
def merge_portfolio_data(csv_dir = "labeled_data",mode = "short",output_file = "portfolio_merged_data.csv",phase = "train"):
    setup_logger()
    logging.info("merge_portfolio_data.py ver208.3")
    output_file = f"portfolio_merged_data_{phase}.csv"
 
    csv_paths = glob.glob(os.path.join(csv_dir,f"*_{mode}_scaled_{phase}.csv"))
 
    if not csv_paths:
        logging.error(f"指定されたディレクトリが見つかりません。")
        return
    
    logging.info(f"{len(csv_paths)}銘柄データを統合します")
 
    all_dfs = {}
    all_times = set()
 
    #1.データの読み込みとマスタタイムラインの関係
    for path in csv_paths:
        symbol = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
 
        #銘柄ごとの時刻をセットに追加
        all_times.update(df["time"].tolist())
 
        rename_dict = {col:f"{symbol}_{col}" for col in df.columns if col != "time"}
        df = df.rename(columns = rename_dict)
 
 
        all_dfs[symbol] = df
        logging.info(f"読み込み完了:{symbol}(データ数:{len(df)})")
 
    # ------------------------------------------------------------------
    # 1.5 生の価格データ（_labeled_{phase}.csv）を読み込み、raw列として追加で保持する
    #     split_data.py が _scaled_{phase}.csv と全く同じ split_date / train_start_date
    #     で _labeled_{phase}.csv も独立に分割して保存しているため、ここでは同じ
    #     phase引数でそのまま読み込めばよい（time列でのマージも行うので、万一
    #     境界がズレていても該当時刻の行だけが正しく紐づく）。
    # ------------------------------------------------------------------
    raw_dfs = {}
    for symbol in all_dfs.keys():
        labeled_path = os.path.join(csv_dir, f"{symbol}_{mode}_labeled_{phase}.csv")
        if not os.path.exists(labeled_path):
            # 分割済みファイルがまだない場合は、分割前の全期間ファイルにフォールバックする
            fallback_path = os.path.join(csv_dir, f"{symbol}_{mode}_labeled.csv")
            if os.path.exists(fallback_path):
                logging.warning(f"⚠️ {labeled_path} が見つからないため、分割前の "
                                 f"{fallback_path} 全体を読み込み、time列で {phase} 期間に絞り込みます。"
                                 f"（split_data.py の実行をおすすめします）")
                labeled_path = fallback_path
            else:
                logging.warning(f"⚠️ 生価格ファイルが見つかりません: {labeled_path}"
                                 f" -- {symbol} は raw列なしで進みます（取引ロジックが標準化値を"
                                 f"参照したままになるので注意）。")
                continue
 
        raw_df = pd.read_csv(labeled_path)
        raw_df["time"] = pd.to_datetime(raw_df["time"])
 
        # 対象の生列だけを抽出（例: M5_close, M5_high, M5_low, M5_spread, M5_ATR, M15_ATR ...）
        raw_cols = ["time"]
        for col in raw_df.columns:
            if col == "time":
                continue
            if col in RAW_FIELD_EXCLUDE:
                continue
            if any(col.endswith(f"_{suf}") or col == suf for suf in RAW_FIELD_SUFFIXES):
                raw_cols.append(col)
 
        raw_df = raw_df[raw_cols]
 
        # 列名の衝突を避けるため、既存の標準化列とは別に "_raw" サフィックスを付けてリネーム
        rename_dict = {col: f"{symbol}_{col}_raw" for col in raw_df.columns if col != "time"}
        raw_df = raw_df.rename(columns=rename_dict)
 
        raw_dfs[symbol] = raw_df
        logging.info(f"生価格データ読み込み完了:{symbol} ({len(raw_cols)-1}列, データ数:{len(raw_df)})")
 
    master_timeline = pd.DataFrame({"time":sorted(list(all_times))})
    logging.info(f"マスタータイムライン作成完了(全{len(master_timeline)}ステップ)")
 
    #2.データの結合
    merged_df = master_timeline
    for symbol,df in all_dfs.items():
        merged_df = pd.merge(merged_df,df,on = "time",how = "left")
 
    # 2.5 生の価格列も同じ time キーで結合
    for symbol, raw_df in raw_dfs.items():
        merged_df = pd.merge(merged_df, raw_df, on="time", how="left")
 
    logging.info("全銘柄の結合完了。欠損値のハイブリッド補完を開始します...")
 
    #3ハイブリッド補完ルールの適用
    ffill_keywords = ["open", "high", "low", "close", "volume", "spread", "target_class", "target_risk"]
 
    for col in merged_df.columns:
        if col == "time":
            continue
        
        is_fill_target = any(kw in col.lower() for kw in ffill_keywords)
 
        if is_fill_target:
            merged_df[col] = merged_df[col].ffill()
        else:
            merged_df[col] = merged_df[col].interpolate(method = "linear",limit_direction = "both")
    merged_df = merged_df.bfill()
 
    #4.保存
    merged_df.to_csv(output_file,index = False)
    logging.info(f"ポートフォリオ統合データの作成が完了しました。->{output_file}")
 
    logging.info(f"最終データ形状:{merged_df.shape}")
 
 
if __name__ == "__main__":
    merge_portfolio_data(phase = "train")
    merge_portfolio_data(phase = "test")