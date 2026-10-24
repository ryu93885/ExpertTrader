import os
import glob
import joblib
import logging
import argparse

# ログの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def unify_scalers(mode, target_dir):
    """
    指定されたディレクトリ内の個別スケーラーを統合し、1つの辞書ファイルを作成する
    """
    if not os.path.exists(target_dir):
        logging.error(f"❌ 指定されたディレクトリが存在しません: {target_dir}")
        return

    unified_scaler = {}
    
    # 検索パターン (例: scaler_GOLD_short.pkl, scaler_AUDUSD_short.pkl など)
    search_pattern = os.path.join(target_dir, f"scaler_*_{mode}.pkl")
    pkl_files = glob.glob(search_pattern)
    
    # すでに存在する統合ファイル自身を除外するためのファイル名
    unified_filename = f"scaler_{mode}.pkl"
    
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        
        # 統合ファイル自体はスキップ
        if filename == unified_filename:
            continue
            
        # ファイル名から銘柄名(symbol)を抽出 (想定フォーマット: scaler_SYMBOL_mode.pkl)
        parts = filename.split('_')
        if len(parts) >= 3:
            symbol = parts[1]
            try:
                # 個別スケーラーを読み込み、辞書に格納
                scaler = joblib.load(file_path)
                unified_scaler[symbol] = scaler
                logging.info(f"🔄 読み込み成功: {symbol} ({filename})")
            except Exception as e:
                logging.error(f"❌ {filename} の読み込みに失敗しました: {e}")
                
    # 統合ファイルの保存
    if unified_scaler:
        save_path = os.path.join(target_dir, unified_filename)
        joblib.dump(unified_scaler, save_path)
        logging.info(f"✅ 統合完了! 計 {len(unified_scaler)} 銘柄のスケーラーを保存しました。")
        logging.info(f"📁 保存先: {save_path}")
    else:
        logging.warning(f"⚠️ 統合可能な個別スケーラーが見つかりませんでした。(モード: {mode}, 検索先: {target_dir})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="個別銘柄のスケーラーを1つの辞書ファイルに統合します")
    parser.add_argument("--mode", type=str, default="medium", choices=["short", "medium", "long"],
                        help="統合する時間軸モード (short, medium, long)")
    parser.add_argument("--dir", type=str, default="saved_models", 
                        help="スケーラーファイルが保存されているディレクトリのパス (デフォルトはカレントディレクトリ)")
    
    args = parser.parse_args()
    
    logging.info(f"=== スケーラー統合処理開始 | モード: {args.mode} | 対象フォルダ: {args.dir} ===")
    unify_scalers(mode=args.mode, target_dir=args.dir)