import os
import re
import glob

# 💡 train_portfolio_rl.py / test_portfolio.py の両方で、ポートフォリオRLモデルの
# 保存場所の解決ロジックを共有するためのユーティリティ。
#
# train_portfolio_rl.py が実際に保存しうる場所は3種類ある:
#   1. ローカルの完了済みモデル: saved_rl_models/sac_portfolio_agent_{mode}.zip
#      (model.learn()が最後まで完走した場合にのみ作られる)
#   2. Google Driveへコピーされた完了済みモデル(Driveマウント時のみ):
#      /content/drive/MyDrive/saved_models_rl/sac_portfolio_agent_{mode}.zip
#   3. 途中保存のチェックポイント(CheckpointCallback。ローカル or Drive):
#      {dir}/sac_portfolio_agent_{mode}_{steps}_steps.zip
#
# Colabのセッションが切れてローカルディスクが失われた場合や、model.learn()が
# 完走せずに途中保存だけが残っている場合など、実行環境によって実際にどこに
# モデルがあるかが変わるため、優先順位(完了済み > 途中保存、ローカル/Drive問わず
# 見つかったものを使う)に沿って自動的に解決する。

DRIVE_FINAL_MODEL_DIR = "/content/drive/MyDrive/saved_models_rl"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/FX_AI_Models/rl_checkpoints"
LOCAL_FINAL_MODEL_DIR = "saved_rl_models"
LOCAL_CHECKPOINT_DIR = "saved_rl_models/checkpoints"


def get_checkpoint_dirs():
    """
    途中保存の保存先候補ディレクトリを返す(先頭が実際の保存先、全体が検索対象)。
    Google Driveがマウントされていれば優先的にそちらへ保存する。
    """
    dirs = []
    if os.path.exists("/content/drive/MyDrive"):
        dirs.append(DRIVE_CHECKPOINT_DIR)
    dirs.append(LOCAL_CHECKPOINT_DIR)
    return dirs


def find_latest_checkpoint(checkpoint_dirs, mode):
    """
    途中保存されたモデル本体(.zip)の中から、最もステップ数が大きいものを探す。
    複数の保存先候補(Drive・ローカル)をまたいで検索し、無ければNoneを返す。
    """
    best_path, best_steps = None, -1
    for d in checkpoint_dirs:
        pattern = os.path.join(d, f"sac_portfolio_agent_{mode}_*_steps.zip")
        for path in glob.glob(pattern):
            m = re.search(r"_(\d+)_steps\.zip$", path)
            if not m:
                continue
            steps = int(m.group(1))
            if steps > best_steps:
                best_path, best_steps = path, steps
    return best_path


def checkpoint_companion_path(model_ckpt_path, checkpoint_type):
    """
    モデル本体のチェックポイントパスから、対応するvecnormalize/replay_bufferの
    パスを組み立てる(CheckpointCallbackの命名規則: {prefix}_{type}{steps}_steps.{ext})。
    """
    m = re.match(r"^(.*)_(\d+)_steps\.zip$", os.path.basename(model_ckpt_path))
    prefix, steps = m.group(1), m.group(2)
    return os.path.join(os.path.dirname(model_ckpt_path), f"{prefix}_{checkpoint_type}{steps}_steps.pkl")


def resolve_model_paths(mode):
    """
    実際に使うべきモデル(.zip)・VecNormalize統計(.pkl)・リプレイバッファ(.pkl)の
    パスを、以下の優先順位で自動的に解決する:
        1. ローカルの完了済みモデル
        2. Google Drive上の完了済みモデル(コピーされていれば)
        3. 途中保存のチェックポイント(ローカル・Driveのうち最もステップ数が大きいもの)

    戻り値: (model_path, vecnorm_path, replay_buffer_path, description) のタプル。
    replay_buffer_path は見つからない場合 None になりうる(呼び出し側で
    os.path.exists を確認すること。途中保存には元々含まれていない設計のため)。
    何も見つからなければ (None, None, None, None) を返す。
    """
    local_model = os.path.join(LOCAL_FINAL_MODEL_DIR, f"sac_portfolio_agent_{mode}.zip")
    local_vecnorm = os.path.join(LOCAL_FINAL_MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(local_model) and os.path.exists(local_vecnorm):
        local_replay = os.path.join(LOCAL_FINAL_MODEL_DIR, f"sac_replay_buffer_{mode}.pkl")
        return local_model, local_vecnorm, local_replay, "ローカルの完了済みモデル"

    drive_model = os.path.join(DRIVE_FINAL_MODEL_DIR, f"sac_portfolio_agent_{mode}.zip")
    drive_vecnorm = os.path.join(DRIVE_FINAL_MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(drive_model) and os.path.exists(drive_vecnorm):
        drive_replay = os.path.join(DRIVE_FINAL_MODEL_DIR, f"sac_replay_buffer_{mode}.pkl")
        return drive_model, drive_vecnorm, drive_replay, "Google Drive上の完了済みモデル"

    latest_ckpt = find_latest_checkpoint(get_checkpoint_dirs(), mode)
    if latest_ckpt is not None:
        vecnorm_ckpt = checkpoint_companion_path(latest_ckpt, "vecnormalize_")
        if os.path.exists(vecnorm_ckpt):
            replay_ckpt = checkpoint_companion_path(latest_ckpt, "replay_buffer_")
            return latest_ckpt, vecnorm_ckpt, replay_ckpt, f"途中保存のチェックポイント({latest_ckpt})"

    return None, None, None, None