import subprocess
import sys
import os
import shutil
from datetime import datetime

MODE = "short"

def run_command(command):
    print(f"🚀 実行中: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ エラー発生: {command}")
        sys.exit(1)
    print(f"✅ 完了: {command}\n" + "-"*40)

def backup_existing_models():
    """
    💡 run_portfolio_retrain.sh と同様、観測次元やロジックが変わった後に
    既存モデルへ継続学習が誤って乗ってしまわないよう、退避してから新規学習する。
    """
    targets = [
        f"saved_rl_models/sac_portfolio_agent_{MODE}.zip",
        "saved_rl_models/vec_normalize.pkl",
        f"saved_rl_models/sac_replay_buffer_{MODE}.pkl",
    ]
    existing = [f for f in targets if os.path.exists(f)]
    if not existing:
        print("  (退避対象の既存モデルは見つかりませんでした。新規学習として開始します)")
        return

    backup_dir = f"saved_rl_models/_backup_before_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    for f in existing:
        shutil.move(f, os.path.join(backup_dir, os.path.basename(f)))
        print(f"  退避: {f} -> {backup_dir}/")

def main():
    print("========================================")
    print(" 🌍 Google Colab ポートフォリオ学習パイプライン")
    print("========================================")
    print("run_colab_pipeline.py ver209.0")
    # 1. 必要なライブラリのインストール
    # Colabにデフォルトで入っていないものを追加します
    print("\n📦 パッケージをインストールしています...")
    run_command("pip install stable-baselines3 gymnasium mplfinance")

    # 2. 必要なディレクトリの作成
    dirs = ["images", "labeled_data", "saved_model", "saved_rl_models", "sac_portfolio_tensorboard"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"📁 必須ディレクトリを確認・作成しました。")

    print("\n=== 既存のポートフォリオRLモデルを退避 ===")
    backup_existing_models()

    # 3. データ統合プログラムの実行 (Source 47 相当)
    # ※事前に labeled_data ディレクトリにCSVを配置しておく必要があります
    print("\n🔄 データの結合処理を開始します...")
    run_command("python merge_portfolio_data.py")

    # 4-0. 強化学習で使用する推論結果を作成する(train/testの両フェーズ)
    # 💡 修正: testフェーズも実行しないと、学習後に test_portfolio.py で
    # バックテストできる状態にならない(portfolio_merged_data_test_with_preds.csv が
    # 生成されないため)。
    print("\n推論結果をcsvファイルに結合して保存します (train)")
    run_command(f"python precompute_portfolio.py --phase train --mode {MODE}")
    print("\n推論結果をcsvファイルに結合して保存します (test)")
    run_command(f"python precompute_portfolio.py --phase test --mode {MODE}")

    # 4. 強化学習 (SAC) の実行 (Source 46 相当)
    print("\n🧠 SACエージェントの強化学習を開始します...")
    run_command("python train_portfolio_rl.py")

    print("\n🎉 全ての学習プロセスが正常に完了しました！")
    print(f"👉 出力された 'saved_rl_models/sac_portfolio_agent_{MODE}.zip' と 'saved_rl_models/vec_normalize.pkl' をダウンロードし、Windows環境のBot(portfolio_trading_bot.py)で利用してください。")
    print(f"👉 バックテストは 'python test_portfolio.py --mode {MODE}' で実行できます。")

if __name__ == "__main__":
    main()