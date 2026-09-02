import os
import re
import glob
import argparse
import torch
import logging
import shutil
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule
# 作成したモジュールをインポート
from portfolio_dataset import PortfolioFXDataset
from portfolio_env import PortfolioFXEnv
from model import MultimodalFXmodel  # 既存のアナライザモデル
from sac_grad_clip import SACWithGradClip
from rl_checkpoint_utils import get_checkpoint_dirs,resolve_model_paths

# 💡 学習率スケジュール: 固定値をトライ&エラーで探すのではなく、学習が進むにつれて
# 自動的に更新幅を小さくしていく(SB3標準のLinearSchedule)。80万ステップ付近で
# critic_loss・actor_loss・ent_coefが同時に悪化する現象が観測されたため、後半ほど
# 大きな更新が入りにくくすることで、同種の発散を抑える狙い。
# なお LinearSchedule は model.learn() が呼ばれるたびに progress_remaining=1 から
# 再スタートする(reset_num_timesteps=False でも learn() の呼び出し単位でリセットされる)。
# Colab側で「1回の実行=total_timesteps分だけ追加学習する」という使い方をしている
# 現状の運用とは相性がよく、再開のたびに学習率が高くなりすぎないようにできる。
LR_SCHEDULE_START = 3e-4  # SB3のSACデフォルトと同じ
LR_SCHEDULE_END = 3e-5
LR_SCHEDULE_END_FRACTION = 1.0  # 1回のlearn()呼び出し全体をかけて減衰させる

# 💡 途中保存の頻度(env stepベース)。fps=80前後の実績から、約10分に1回のペース。
CHECKPOINT_SAVE_FREQ = 50_000


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def main():
    setup_logger()
    logging.info("portfolio_portfolio_rl.py ver210.0")

    # 💡 修正: mode がCLI引数化されておらず常に"short"固定だったため、mediumモードで
    # 運用していても PortfolioFXDataset が images/{symbol}/short_MTF/ を探しに行き、
    # 画像が見つからず全銘柄・全ステップが黒画像(ゼロテンソル)にすり替わったまま
    # 気づかれずに学習が進んでしまう(例外を握りつぶすフォールバックがあるため)
    # 事故があった。precompute_portfolio.py等と同じく --mode を明示指定できるようにする。
    parser = argparse.ArgumentParser(description="ポートフォリオRL(SAC)の学習")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "medium", "long"], help="対象モード")
    args = parser.parse_args()

    # 1. 基本設定
    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    mode = args.mode
    merged_csv = "portfolio_merged_data_train_with_preds.csv"
    img_dir = "images"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")



    # 2. データセットの準備
    logging.info("データセットを読み込んでいます...")
    dataset = PortfolioFXDataset(merged_csv_path=merged_csv, base_img_dir=img_dir, symbols=symbols, mode=mode)


    #3.アナライザのモデル読み込み(省略)

    # 4. 強化学習環境の構築
    env = PortfolioFXEnv(dataset=dataset,symbols=symbols)
    vec_env = DummyVecEnv([lambda: env])

    vec_env = VecNormalize(
    vec_env,
    training=True,
    norm_obs=True,
    norm_reward=True,       # 報酬の正規化（学習の安定化に寄与します）
    clip_obs=10.0,          # 極端な外れ値をクリップ
    clip_reward=10.0,
    gamma=0.99,             # ※お使いのアルゴリズムの gamma (割引率) に合わせて変更してください
    epsilon=1e-8
)
    # 5. SACエージェントの定義と学習
    # 21次元の連続値アクションなので、MLPポリシーを使用します

    agent_save_path_base = f"saved_rl_models/sac_portfolio_agent_{mode}"
    vec_norm_path = "saved_rl_models/vec_normalize.pkl"
    replay_buffer_path = f"saved_rl_models/sac_replay_buffer_{mode}.pkl"

    checkpoint_dirs = get_checkpoint_dirs()
    checkpoint_save_dir = checkpoint_dirs[0]
    if checkpoint_save_dir.startswith("/content/drive"):
        logging.info(f"Google Driveを検出。途中保存の保存先をDriveに設定します: {checkpoint_save_dir}")
    else:
        logging.warning(
            "Google Driveが未マウントのため、途中保存はColabのローカルディスクに保存されます。"
            "セッション切断(タイムアウト等)でVMごと失われる可能性があるので注意してください。"
        )

    lr_schedule = LinearSchedule(start=LR_SCHEDULE_START, end=LR_SCHEDULE_END, end_fraction=LR_SCHEDULE_END_FRACTION)

        # 💡 修正: 以前はローカルの完了済みモデルとローカル/Driveのチェックポイントしか
    # 見ておらず、「Google Driveにコピーされた完了済みモデル」だけが存在するケース
    # (ローカルディスクが失われたが、学習自体は完走してDriveにバックアップ済みの場合)
    # を見落としていた。resolve_model_paths() で3種類の保存先を優先順位付きで検索する
    # (test_portfolio.py と同じロジックを共有)。
    resumed_model_path, resumed_vecnorm_path, resumed_replay_path, model_source = resolve_model_paths(mode)

    if resumed_model_path is not None:
        logging.info(f"既存のモデルを検出しました({model_source}): {resumed_model_path}")
        vec_env = VecNormalize.load(resumed_vecnorm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

        model = SACWithGradClip.load(resumed_model_path, env=vec_env, device=device, learning_rate=lr_schedule)

        if resumed_replay_path is not None and os.path.exists(resumed_replay_path):
            model.load_replay_buffer(resumed_replay_path)
            logging.info("リプレイバッファを読み込みました")
        else:
            # 💡 途中保存にはディスク容量節約のためreplay bufferを含めていないため、
            # チェックポイントから再開する場合は空バッファから始まる。learning_starts分の
            # ウォームアップ後すぐに通常通り学習が再開されるため、致命的な影響はない。
            logging.warning("リプレイバッファが見つかりません。新規メモリで再開します。")
    else:
        logging.info("🚀 新規モデルとしてSACエージェントの学習を開始します...")
        model = SACWithGradClip(
            "MlpPolicy", vec_env, verbose=1,
            tensorboard_log="./sac_portfolio_tensorboard/",
            learning_rate=lr_schedule,
        )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=checkpoint_save_dir,
        name_prefix=f"sac_portfolio_agent_{mode}",
        save_replay_buffer=False,  # 容量節約(見つからなければ新規メモリで再開する扱いのため)
        save_vecnormalize=True,
        verbose=1,
    )

    # 学習実行（ステップ数は適宜調整）
    model.learn(total_timesteps=1000000, reset_num_timesteps=False, callback=checkpoint_callback)

    # モデルの保存
    os.makedirs("saved_rl_models", exist_ok=True)


    model.save(agent_save_path_base)
    vec_env.save(vec_norm_path)
    model.save_replay_buffer(replay_buffer_path)

    logging.info("学習が完了し、モデルを保存しました！")

    drive_base_path = "/content/drive/MyDrive"

    if os.path.exists(drive_base_path):
        logging.info("マウント済みのGoogle Driveを検出しました。コピーを開始します...")
        drive_save_dir = os.path.join(drive_base_path, "saved_models_rl")
        os.makedirs(drive_save_dir, exist_ok=True)

        agent_zip_path = f"{agent_save_path_base}.zip"
        files_to_copy = [agent_zip_path, vec_norm_path,replay_buffer_path]

        for file_path in files_to_copy:
            if os.path.exists(file_path):
                # shutil.copy2 を使うことでファイルのメタデータ（作成日時など）も保持してコピーします
                shutil.copy2(file_path, drive_save_dir)
                logging.info(f"Driveに保存しました: {os.path.basename(file_path)}")
            else:
                logging.warning(f"コピー元ファイルが見つかりません: {file_path}")

        logging.info("Google Driveへの保存処理がすべて完了しました！")
    else:
        logging.warning("Google Driveがマウントされていないようです。Driveへの保存はスキップされました。")



if __name__ == "__main__":
    main()
