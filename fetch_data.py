import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime,timezone,timedelta
import logging
import os
import argparse

#definissioi of settings and conditions 
TRADING_MODES = {
    "short":{"M5":mt5.TIMEFRAME_M5,"M15":mt5.TIMEFRAME_M15,"M30":mt5.TIMEFRAME_M30},
    "medium":{"M15":mt5.TIMEFRAME_M15,"H1":mt5.TIMEFRAME_H1,"H4":mt5.TIMEFRAME_H4},
    "long":{"H4":mt5.TIMEFRAME_H4,"D1":mt5.TIMEFRAME_D1,"W1":mt5.TIMEFRAME_W1}
}

#default stock list
# 💡 修正: "XAUUSD" ではなく、このブローカーの実際のMT5銘柄名 "GOLD" に修正
# (portfolio_trading_bot.py 等、システム全体で使われている名称と統一)
DEFAULT_SYMBOLS = ["USDJPY","EURUSD","GBPUSD","EURJPY","GBPJPY","AUDUSD","GOLD"]

def setup_logger():
    #log settings:console output and file storage
    log_dir = "logs"
    os.makedirs(log_dir,exist_ok = True)
    log_filename = datetime.now().strftime(f"{log_dir}/fetch_%Y%m%d_%H%M%S.log")

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s [%(levelname)s] %(message)s',
        handlers = [
            logging.FileHandler(log_filename,encoding = "utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_filename

def check_data_gaps(df,symbol,timeframe_name,timeframe_seconds):
    #check data continuity and log any missing values
    if len(df) < 2:
        return
    
    expected_delta = timedelta(seconds = timeframe_seconds)
    actual_deltas = df['time'].diff().dropna()
    gaps = actual_deltas[actual_deltas > expected_delta]

    if not gaps.empty:
        logging.warning(f"[!]detection of missing data.({symbol} - {timeframe_name}:there are total of {len(gaps)} gaps.Largest gaps:{gaps.max()})")
    else :
        logging.info(f"[✓]data continuity varification complete({symbol} - {timeframe_name})")

def get_mt5_data(symbol,tf_name,tf_val,start_date,end_date):
    #retrive data from mt5,convert it to UTC,and perform consistency checks
    logging.info(f"start data collection")
    rates = mt5.copy_rates_range(symbol,tf_val,start_date,end_date)

    if rates is None or len(rates) == 0:
        logging.warning(f"faild to retrivedata:{symbol}({tf_name}) erorr code:{mt5.last_error()}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'],unit = 's')

    tf_seconds = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds() if len(df) > 1 else 0
    check_data_gaps(df,symbol,tf_name,tf_seconds)
    logging.info(f"download complete:{len(df)} datas({symbol} - {tf_name})")
    
    return df

def save_to_csv(df,symbol,tf_name):
    if df.empty :return
    data_dir = "data"
    os.makedirs(data_dir,exist_ok = True)
    filename = f"{data_dir}/{symbol}_{tf_name}.csv"

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df,df],ignore_index = True)

        combined_df["time"] = pd.to_datetime(combined_df["time"])
        combined_df.drop_duplicates(subset=["time"],keep = "first",inplace = True)
        combined_df.sort_values(by = "time",inplace = True)
        combined_df.to_csv(filename,index = False)
    else:
        df.to_csv(filename,index = False)
        

def main():
    log_file = setup_logger()
    logging.info("===Lundh MT5 Data Fetcher=== ")

    parser = argparse.ArgumentParser(description = "Fech FX Data")
    parser.add_argument("--symbol",type = str,help = "Trading symbol separated by comma")
    parser.add_argument("--mode",type = str,help = "Trading mode (short/medium/long/all)")
    parser.add_argument("--start",type = str)
    parser.add_argument("--end",type = str)

    args = parser.parse_args()

    if not mt5.initialize():
        logging.critical(f"failed to initialize MT5")
        return

    try:
        # 1. stock select entry
        print("\n---stock select---")
        if args.symbol:
            symbols_input = args.symbol
        else:
            print(f"default stock: {','.join(DEFAULT_SYMBOLS)}")
            symbols_input = input("Please enter the stocks you want to analyze, separated by commas (pressing Enter with a blank field will apply the default values): ").strip().upper()

        if not symbols_input:
            target_symbols = DEFAULT_SYMBOLS
            logging.info(f"no user input. use default. -> {target_symbols}")
        else:
            target_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            logging.info(f"Start processing using the user-specified stock. -> {target_symbols}")
        
        if not target_symbols:
            logging.error("no valid stock has been selected")
            return 
        
        # 2. mode select
        print("\n---mode select---")
        if args.mode:
            mode_input = args.mode.strip().lower()
        else:
            print("short/medium/long/all")
            mode_input = input("mode entry [Default: short]: ").strip().lower() or "short"

        selected_modes = list(TRADING_MODES.keys()) if mode_input == "all" else [m.strip() for m in mode_input.split(",")]
        target_timeframes = {}
        for mode in selected_modes:
            if mode in TRADING_MODES:
                target_timeframes.update(TRADING_MODES[mode])

        if not target_timeframes:
            logging.error("no valid mode was selected")
            return
        
        # 3. setting the period
        try:
            start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime.strptime(input("start date (YYYY-MM-DD): "), "%Y-%m-%d")
            end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.strptime(input("end date (YYYY-MM-DD): "), "%Y-%m-%d")
        except ValueError:
            print("[!] 日付のフォーマットが正しくありません。YYYY-MM-DD (例: 2024-01-01) の形式で入力してください。")
            return

        print(f"\n[*] Fetching data for {', '.join(target_symbols)} | Mode: {mode_input}")
        print(f"    Target Period : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

        # main loop for each stock
        for symbol in target_symbols:
            for tf_name, tf_val in target_timeframes.items():
                df = get_mt5_data(symbol, tf_name, tf_val, start_date, end_date)
                save_to_csv(df, symbol, tf_name)

        logging.info("\n===the process for all stocks has been completed===")

    except Exception as e:
        logging.error(f"an error has occured:{e}",exc_info=True)
    finally:
        mt5.shutdown()
        try:
            os.chmod(log_file,0o444)
        except:
            pass
        logging.info(f"===complete(log:{log_file})===")

if __name__ == "__main__":
    main()