import pandas as pd
import glob
import os
import joblib
import argparse
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_DIR_LABELED = "labeled_data"
INPUT_DIR_PROCESSED = "processed_data"
SAVE_MODEL_DIR = "saved_models"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="銘柄別・独立スケーリング（正規化）プログラム")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"], help="スケーリングの対象モード (デフォルト: medium)")
    parser.add_argument("--symbol") # コマンド引数の互換性維持のために残しています
    args = parser.parse_args()
    
    TARGET_MODE = args.mode
    print(f"--- 🚀 銘柄別・独立スケーリングプログラム開始 (モード: {TARGET_MODE}) ---")
    all_files = []
    if args.symbol:
        target_symbols = [s.strip().upper() for s in args.symbol.split(',')]
        for sym in target_symbols:
            labeled_path = f"{INPUT_DIR_LABELED}/{sym}_{TARGET_MODE}_labeled.csv"
            processed_path = f"{INPUT_DIR_PROCESSED}/{sym}_{TARGET_MODE}.csv"
            
            if os.path.exists(labeled_path):
                all_files.append(labeled_path)
            elif os.path.exists(processed_path):
                print(f"⚠️ {sym} のラベルデータが見つかりません。取引専用として processed_data を使用します。")
                all_files.append(processed_path)
            else:
                print(f"❌ {sym} のデータが見つかりません。")
    else:
        all_files = glob.glob(f"{INPUT_DIR_LABELED}/*_{TARGET_MODE}_labeled.csv")

    if not all_files:
        print(f"❌ {TARGET_MODE} モードの学習用データが見つかりません。")
        return
    
    print(f"📁 {len(all_files)}個の学習データを読み込み、銘柄ごとに分類中...")

    exclude_cols = ["time", "future_close", "price_diff_pct", "target_class", "target_risk_pct"]
    
    # 銘柄ごとにデータをグループ分けする辞書
    symbol_features_map = {}
    symbol_df_map = {}
    features_columns = []

    for file in all_files:
        df = pd.read_csv(file,on_bad_lines="skip")
        # ファイル名から銘柄名（例: USDJPY, XAUUSD）を抽出
        symbol = os.path.basename(file).split("_")[0]
        df.replace([np.inf,-np.inf],np.nan,inplace = True)
        df.bfill(inplace =True)
        df.fillna(1,inplace = True)

        df.reset_index(drop=True,inplace = True)
        

        if not features_columns:
            features_columns = [c for c in df.columns if c not in exclude_cols]
            print(f"📊 特徴量は以下の {len(features_columns)} 個として認識しました:\n{features_columns}")

        missing_cols = [c for c in features_columns if c not in df.columns]
        if missing_cols:
            print(f"⚠️ スキップ: {symbol} に新しい特徴量がありません。古いデータとみなします。")
            continue
        
        if symbol not in symbol_features_map:
            symbol_features_map[symbol] = []
            symbol_df_map[symbol] = []
            
        symbol_features_map[symbol].append(df[features_columns])
        symbol_df_map[symbol].append(df)

    # 🌟 銘柄ごとに独立したスケーラーを構築・適用するメインループ
    scalers_dict = {}
    
    for symbol, features_list in symbol_features_map.items():
        print(f"\n🔄 【銘柄: {symbol}】 専用スケーラーの学習と変換処理を実行中...")
        
        # 同一銘柄のデータを結合
        concat_features = pd.concat(features_list, ignore_index=True)
        
        # 銘柄単体で平均と標準偏差をフィッティング
        scaler = StandardScaler()
        scaler.fit(concat_features)
        scalers_dict[symbol] = scaler
        
        # 今後の運用の柔軟性を高めるため「個別ファイル」としても保存
        indiv_scaler_path = os.path.join(SAVE_MODEL_DIR, f"scaler_{symbol}_{TARGET_MODE}.pkl")
        joblib.dump(scaler, indiv_scaler_path)
        print(f"  ➡️ 個別スケーラーを保存: {indiv_scaler_path}")

        # この銘柄のデータにのみ、専用スケーラーを適用
        scaled_dfs = []
        for df in symbol_df_map[symbol]:
            df_copy = df.copy()
            df_copy[features_columns] = scaler.transform(df_copy[features_columns])
            scaled_dfs.append(df_copy)
            
        # 1つのマスターデータに合体して重複排除
        combined_df = pd.concat(scaled_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["time"], keep="last")
        combined_df = combined_df.sort_values(by="time").reset_index(drop=True)
        
        master_filename = os.path.join(INPUT_DIR_LABELED, f"{symbol}_{TARGET_MODE}_scaled.csv")
        combined_df.to_csv(master_filename, index=False)
        print(f"  ➡️ 変換・上書き保存完了: {master_filename} (総データ数: {len(combined_df)}件)")

    # 🌟 従来のファイル名「scaler_{TARGET_MODE}.pkl」には、全銘柄のスケーラーを辞書形式で一括保存
    dict_scaler_path = os.path.join(SAVE_MODEL_DIR, f"scaler_{TARGET_MODE}.pkl")
    if os.path.exists(dict_scaler_path):
        existing_scalers = joblib.load(dict_scaler_path)
        existing_scalers.update(scalers_dict)
        joblib.dump(existing_scalers, dict_scaler_path)
        print("\n" + "="*50)
        print(f"✅ 既存のスケーラー辞書を更新・統合保存しました: {dict_scaler_path}")
    else:
        joblib.dump(scalers_dict, dict_scaler_path)
        print("\n" + "="*50)
        print(f"✅ 全銘柄のスケーラー辞書を新規統合保存しました: {dict_scaler_path}")
    print(f"--- 🎉 {TARGET_MODE} モードの銘柄別スケーリング処理が正常に完了しました ---")

if __name__ == "__main__":
    main()