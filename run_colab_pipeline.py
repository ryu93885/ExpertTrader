import subprocess
import sys
import os
import shutil
from datetime import datetime

# 💡 修正: mediumモードのみで運用しているため、shortのまま残っていると
# merge_portfolio_data.py / train_portfolio_rl.py に --mode が渡らず(後述)、
# 存在しないshortモードのデータ・画像を参照してしまう(train_portfolio_rl.py側は
# 画像読み込み失敗を握りつぶすフォールバックがあるため、エラーにならず黒画像で
# 学習が進んでしまう)。実際に使用しているモードに合わせて変更してください。
MODE = "medium"

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

    💡 追加: train_portfolio_rl.py は完了済みモデル(sac_portfolio_agent_{mode}.zip)が
    無い場合、途中保存のチェックポイント(saved_rl_models/checkpoints/ および
    Google Drive側のrl_checkpoints/)の中から最新のものを自動的に探して読み込む。
    完了済みモデルだけを退避してチェックポイントを残したままだと、「新規学習のつもり」が
    実際には古い(発散した可能性のある)途中保存から再開されてしまうため、
    チェックポイント用ディレクトリの中身もまとめて退避する。
    """
    targets = [
        f"saved_rl_models/sac_portfolio_agent_{MODE}.zip",
        "saved_rl_models/vec_normalize.pkl",
        f"saved_rl_models/sac_replay_buffer_{MODE}.pkl",
    ]
    existing = [f for f in targets if os.path.exists(f)]

    checkpoint_dirs = [
        "saved_rl_models/checkpoints",
        "/content/drive/MyDrive/FX_AI_Models/rl_checkpoints",
    ]
    existing_checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d) and os.listdir(d)]

    if not existing and not existing_checkpoint_dirs:
        print("  (退避対象の既存モデル・チェックポイントは見つかりませんでした。新規学習として開始します)")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_backup_dir = f"saved_rl_models/_backup_before_fix_{timestamp}"
    os.makedirs(local_backup_dir, exist_ok=True)
    for f in existing:
        shutil.move(f, os.path.join(local_backup_dir, os.path.basename(f)))
        print(f"  退避: {f} -> {local_backup_dir}/")

    # チェックポイントは元がDriveならDrive側に、ローカルならローカル側に退避先を揃える
    # (Driveの容量が心配な場合は、_backup_before_fix_* フォルダを後で手動削除してください)
    for d in existing_checkpoint_dirs:
        if d.startswith("/content/drive"):
            backup_root = f"/content/drive/MyDrive/FX_AI_Models/_backup_before_fix_{timestamp}"
        else:
            backup_root = local_backup_dir
        os.makedirs(backup_root, exist_ok=True)
        dest = os.path.join(backup_root, os.path.basename(d.rstrip("/")))
        shutil.move(d, dest)
        print(f"  退避: {d}/ -> {dest}/")

def main():
    print("========================================")
    print(" 🌍 Google Colab ポートフォリオ学習パイプライン")
    print("========================================")
    print("run_colab_pipeline.py ver209.0")
    # 1. 必要なライブラリのインストール
    # Colabにデフォルトで入っていないものを追加します
    print("\n📦 パッケージをインストールしています...")
    # 💡 sac_grad_clip.py の SACWithGradClip が stable-baselines3==2.9.0 の
    # SAC.train()実装に依存しているため、バージョンを固定している。
    run_command("pip install stable-baselines3==2.9.0 gymnasium mplfinance")

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
    run_command(f"python merge_portfolio_data.py --mode {MODE}")

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
    run_command(f"python train_portfolio_rl.py --mode {MODE}")

    print("\n🎉 全ての学習プロセスが正常に完了しました！")
    print(f"👉 出力された 'saved_rl_models/sac_portfolio_agent_{MODE}.zip' と 'saved_rl_models/vec_normalize.pkl' をダウンロードし、Windows環境のBot(portfolio_trading_bot.py)で利用してください。")
    print(f"👉 バックテストは 'python test_portfolio.py --mode {MODE}' で実行できます。")

if __name__ == "__main__":
    main()