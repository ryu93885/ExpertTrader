import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np  
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging
import json
import argparse
from datetime import timedelta

from model import MultimodalFXmodel
from dataset import FXMultimodalDataset

# ==========================================
# Configuration & Utils
# ==========================================
MODE_SETTINGS = {
    "short": {"base_tf": "M5", "lookahead": 12},
    "medium": {"base_tf": "M15", "lookahead": 16},
    "long": {"base_tf": "H4", "lookahead": 30}
}

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_trained_model(model_path, num_tabular_features, device):
    model = MultimodalFXmodel(num_tabular_features=num_tabular_features)
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Successfully loaded trained model from: {model_path}")
    return model

def generate_prediction_chart(df_plot, pred_data, save_path):
    """Generates a high-definition chart visualization"""
    logging.info(f"Generating visualization chart: {save_path}")
    
    tf = pred_data['tf']
    pip_value = pred_data['pip_value'] 
    
    # Rename columns for mplfinance
    rename_dict = {
        'open': 'Open',   f'{tf}_open': 'Open',
        'high': 'High',   f'{tf}_high': 'High',
        'low': 'Low',     f'{tf}_low': 'Low',
        'close': 'Close', f'{tf}_close': 'Close',
    }
    df_plot_re = df_plot.rename(columns=rename_dict)

    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, gridstyle='', y_on_right=True)
    
    fb_data = dict(
        y1=pred_data['pred_max_price'], 
        y2=pred_data['pred_min_price'], 
        where=df_plot_re.index >= pred_data['pred_start_time'],
        alpha=0.2, 
        color='blue'
    )

    summary_text = (
        f"*** AI PREDICTION SUMMARY ***\n"
        f"Symbol: {pred_data['symbol']} | TF: {pred_data['tf']}\n"
        f"Prediction Time: {pred_data['pred_start_time'].strftime('%Y-%m-%d %H:%M')}\n\n"
        f"[Classification - Direction]\n"
        f"  Signal: {pred_data['class_signal']}\n"
        f"  Confidence: {pred_data['confidence']:.2f}%\n\n"
        f"[Regression - Target Range]\n"
        f"  Current Price: {pred_data['current_price']:.3f}\n"
        f"  Predicted Max: {pred_data['pred_max_price']:.3f} (+{pred_data['pred_pips']:.1f} pips)\n"
        f"  Predicted Min: {pred_data['pred_min_price']:.3f} (-{pred_data['pred_pips']:.1f} pips)\n"
    )

    fig, axlist = mpf.plot(
        df_plot_re,
        type='candle',
        style=s,
        figsize=(12, 8),
        title=f"\nAI Forecast: {pred_data['symbol']} ({pred_data['tf']})",
        ylabel='Price',
        tight_layout=True,
        show_nontrading=True, 
        fill_between=fb_data,
        datetime_format='%Y-%m-%d %H:%M', 
        xrotation=15,                     
        returnfig=True
    )

    ax = axlist[0] 

    # Draw horizontal grids (price axis)
    y_min, y_max = ax.get_ylim()
    grid_interval = pip_value * 10.0
    
    start_price = (y_min // grid_interval) * grid_interval
    grid_prices = []
    curr_price = start_price
    while curr_price <= y_max:
        if curr_price >= y_min:
            grid_prices.append(curr_price)
        curr_price += grid_interval
    
    for p in grid_prices:
        ax.axhline(y=p, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)

    # Draw vertical grids (time axis)
    ax.grid(axis='x', color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)

    # Set x-axis limits to ensure future space is visible
    x_min = mdates.date2num(df_plot_re.index[0])
    x_max = mdates.date2num(df_plot_re.index[-1])
    x_margin = (x_max - x_min) * 0.02
    ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # Draw arrows and predicted trajectory line
    if pred_data['class_signal'] != 'HOLD':
        arrow_color = 'green' if pred_data['class_signal'] == 'BUY' else 'red'
        m_date_start = mdates.date2num(pred_data['pred_start_time'])
        m_date_end = mdates.date2num(pred_data['pred_end_time'])
        y_offset = df_plot_re['High'].max() * 0.002
        
        # Calculate target price for the trajectory line
        if pred_data['class_signal'] == 'BUY':
            target_price = pred_data['current_price'] + (pred_data['pred_pips'] * pip_value)
            xy_text = (m_date_start, pred_data['current_price'] - y_offset*3)
            xy_arrow = (m_date_start, pred_data['current_price'] + y_offset)
        else:
            target_price = pred_data['current_price'] - (pred_data['pred_pips'] * pip_value)
            xy_text = (m_date_start, pred_data['current_price'] + y_offset*3)
            xy_arrow = (m_date_start, pred_data['current_price'] - y_offset)

        # Draw entry arrow
        ax.annotate(
            '', xy=xy_arrow, xytext=xy_text,
            arrowprops=dict(facecolor=arrow_color, edgecolor='black', shrink=0.05, width=7, headwidth=15),
            horizontalalignment='center'
        )

        # Draw predicted trajectory line
        ax.plot([m_date_start, m_date_end], [pred_data['current_price'], target_price], 
                color=arrow_color, linestyle='--', linewidth=2, zorder=5, alpha=0.8)
        
        # Draw target star marker
        ax.scatter(m_date_end, target_price, color=arrow_color, marker='*', s=150, zorder=6, edgecolor='black', linewidth=0.5)

    # Draw summary text box
    fig.text(
        0.02, 0.70, summary_text, fontsize=10, color='black',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
        verticalalignment='top', fontfamily='monospace'
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# Main Inference Logic
# ==========================================
def run_inference():
    setup_logger()
    logging.info("=== Launching AI FX Prediction System ===")
    
    parser = argparse.ArgumentParser(description="Run Daily Predictions")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol(s) separated by comma")
    parser.add_argument("--mode", type=str, required=True, choices=["short", "medium", "long"], help="Trading mode")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    mode = args.mode.lower()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    settings = MODE_SETTINGS.get(mode)
    base_tf = settings["base_tf"]
    lookahead_bars = settings["lookahead"]
    
    tf_map = {"M1": "1T", "M5": "5T", "M15": "15T", "M30": "30T", "H1": "1H", "H4": "4H", "D1": "D", "W1": "W"}
    freq_str = tf_map.get(base_tf, "15T")

    # 🌟 Zスコア正規化のスケーラーを読み込む (ディレクトリは saved_models).
    scaler_path = f"saved_models/risk_scalers_{mode}.json"
    risk_scalers = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            risk_scalers = json.load(f)
        logging.info(f"Loaded risk scalers from {scaler_path}")
    else:
        logging.warning(f"Risk scalers not found at {scaler_path}. Predictions may not be properly scaled.")

    for symbol in symbols:
        logging.info(f"\n--- Starting Prediction for {symbol} ({mode}) ---")
        
        # 🌟 GOLDの描画用スケールバグを修正（XAUやGOLDも0.01に含める）
        is_jpy_or_gold = "JPY" in symbol or "XAU" in symbol or "GOLD" in symbol
        pip_value = 0.01 if is_jpy_or_gold else 0.0001
        
        # 🌟 モデルの読み込み先ディレクトリは `saved_model` に修正 (train.pyの改修に追従)
        model_path = f"saved_model/fx_multimodal_global_{mode}.pth"
        
        csv_path = f"processed_data/{symbol}_test_{mode}_processed.csv"
        img_dir = f"images/{symbol}/test/{base_tf}" 
        
        if not os.path.exists(csv_path):
            logging.error(f"Processed CSV data not found: {csv_path}")
            continue
        if not os.path.exists(img_dir):
            logging.error(f"Image directory not found: {img_dir}")
            continue

        full_df = pd.read_csv(csv_path)
        full_df['time'] = pd.to_datetime(full_df['time'])
        full_df.set_index('time', inplace=True)

        logging.info("Scanning for available data points with images...")
        valid_times = []
        for timestamp in full_df.index:
            img_path = os.path.join(img_dir, f"{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
            if os.path.exists(img_path):
                valid_times.append(timestamp)

        if not valid_times:
            logging.error("No valid images found for the given CSV timestamps. Skipping.")
            continue
            
        logging.info(f"Found {len(valid_times)} valid data points for prediction.")

        temp_df = full_df.loc[valid_times].copy()
        temp_df["target_class"] = 0
        temp_df["target_risk_pips"] = 0.0

        temp_csv_path = f"temp_inference_{symbol}.csv"
        temp_df.to_csv(temp_csv_path)
        
        # シンボル用のスケーラーを取得
        sym_scaler = risk_scalers.get(symbol)
        if not sym_scaler:
            logging.warning(f"No scaler found for {symbol}. Output will be raw Z-scores.")
            
        try:
            inference_dataset = FXMultimodalDataset(csv_file=temp_csv_path, img_dir=img_dir)
            num_tabular_features = len(inference_dataset.features_cols)
            
            dataloader = DataLoader(inference_dataset, batch_size=32, shuffle=False)
            
            model = load_trained_model(model_path, num_tabular_features, device)
            if model is None: 
                continue

            all_signals = []
            all_confs = []
            all_risks = []

            logging.info("Running inference on all valid data points...")
            with torch.no_grad():
                for images, tabulars, _, _ in dataloader:
                    images = images.to(device)
                    tabulars = tabulars.to(device)
                    
                    out_class, out_risk = model(images, tabulars)
                    
                    class_probs = F.softmax(out_class, dim=1) 
                    conf, class_idx = torch.max(class_probs, dim=1) 
                    
                    class_ids = (class_idx - 1).cpu().numpy()
                    confs = (conf * 100).cpu().numpy()
                    risks = out_risk.view(-1).cpu().numpy()
                    
                    # 🌟 Zスコアから元のPipsへ逆変換 (予測値 * 標準偏差 + 平均)
                    if sym_scaler:
                        risks = risks * sym_scaler['std'] + sym_scaler['mean']
                    
                    # 値幅は距離なので確実に正の数（絶対値）にする
                    risks = np.abs(risks)
                    
                    for cid, cf, rsk in zip(class_ids, confs, risks):
                        signal_str = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[cid]
                        all_signals.append(signal_str)
                        all_confs.append(cf)
                        all_risks.append(rsk)

            temp_df["pred_signal"] = all_signals
            temp_df["pred_confidence"] = all_confs
            temp_df["pred_risk_pips"] = all_risks
            
            out_pred_dir = "predictions"
            os.makedirs(out_pred_dir, exist_ok=True)
            all_preds_path = os.path.join(out_pred_dir, f"{symbol}_{mode}_all_predictions.csv")
            temp_df.to_csv(all_preds_path)
            logging.info(f"Saved all predictions to: {all_preds_path}")

            # 最新の予測結果で画像を生成
            latest_time = valid_times[-1]
            current_price = temp_df.loc[latest_time, f"{base_tf}_close"]
            latest_signal = all_signals[-1]
            latest_conf = all_confs[-1]
            latest_risk = all_risks[-1]
            
            logging.info(f"Latest Prediction Result [{latest_time}]: Signal={latest_signal} ({latest_conf:.1f}%), Risk={latest_risk:.1f} pips")

            future_dates = pd.date_range(start=latest_time, periods=lookahead_bars + 1, freq=freq_str)[1:]
            prediction_horizon_time = future_dates[-1]

            pred_max_price = current_price + (latest_risk * pip_value)
            pred_min_price = current_price - (latest_risk * pip_value)

            prediction_data_summary = {
                'symbol': symbol,
                'tf': base_tf,
                'pip_value': pip_value, 
                'pred_start_time': latest_time,
                'current_price': current_price,
                'class_signal': latest_signal,
                'confidence': latest_conf,
                'pred_pips': latest_risk,
                'pred_max_price': pred_max_price,
                'pred_min_price': pred_min_price,
                'lookahead_bars': lookahead_bars,
                'pred_end_time': prediction_horizon_time
            }

            PLOT_PAST_BARS = 100 
            df_history = full_df.loc[:latest_time].copy()
            
            df_history = df_history.resample(freq_str).last().dropna(subset=[f"{base_tf}_close"])
            df_history = df_history.tail(PLOT_PAST_BARS)
            
            df_future = pd.DataFrame(index=future_dates, columns=df_history.columns)
            df_plot = pd.concat([df_history, df_future])
            
            timestamp_str = latest_time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(out_pred_dir, f"{symbol}_{base_tf}_{timestamp_str}_forecast.png")
            
            generate_prediction_chart(df_plot, prediction_data_summary, save_path)
            logging.info(f"✅ Visualization generated at: {save_path}")

        finally:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path) 

    logging.info("=== All Predictions Completed ===")

if __name__ == "__main__":
    run_inference()