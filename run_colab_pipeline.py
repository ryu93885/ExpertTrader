import subprocess
import sys
import os

def run_command(command):
    print(f"🚀 実行中: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ エラー発生: {command}")
        sys.exit(1)
    print(f"✅ 完了: {command}\n" + "-"*40)

def main():
    print("========================================")
    print(" 🌍 Google Colab ポートフォリオ学習パイプライン")
    print("========================================")
    print("run_colab_pipeline.py ver208.2")
    # 1. 必要なライブラリのインストール
    # Colabにデフォルトで入っていないものを追加します
    print("\n📦 パッケージをインストールしています...")
    run_command("pip install stable-baselines3 gymnasium mplfinance")

    # 2. 必要なディレクトリの作成
    dirs = ["images", "labeled_data", "saved_model", "saved_rl_models", "sac_portfolio_tensorboard"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"📁 必須ディレクトリを確認・作成しました。")

    # 3. データ統合プログラムの実行 (Source 47 相当)
    # ※事前に labeled_data ディレクトリにCSVを配置しておく必要があります
    print("\n🔄 データの結合処理を開始します...")
    run_command("python merge_portfolio_data.py")

    #4-0強化学習で使用する推論結果を作成する
    print("\n推論結果をcsvファイルの結合して保存します")
    run_command("python precompute_portfolio.py")

    
    # 4. 強化学習 (SAC) の実行 (Source 46 相当)
    print("\n🧠 SACエージェントの強化学習を開始します...")
    run_command("python train_portfolio_rl.py")

    print("\n🎉 全ての学習プロセスが正常に完了しました！")
    print("👉 出力された 'saved_rl_models/sac_portfolio_agent.zip' と 'saved_rl_models/vec_normalize.pkl' をダウンロードし、Windows環境のBot(portfolio_trading_bot.py)で利用してください。")

if __name__ == "__main__":
    main()