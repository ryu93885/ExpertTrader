import os
import re
import glob
import time
import logging

# 💡 train_portfolio_rl.py / test_portfolio.py の両方で、ポートフォリオRLモデルの
# 保存場所の解決ロジックを共有するためのユーティリティ。
#
# train_portfolio_rl.py が実際に保存しうる場所は3種類ある:
#   1. ローカルの完了済みモデル: saved_rl_models/sac_portfolio_agent_{mode}.zip
#      (model.learn()が最後まで完走した場合にのみ作られる)
#   2. Google Driveへコピーされた完了済みモデル(Driveマウント時のみ):
#      /content/drive/MyDrive/saved_models_rl/sac_portfolio_agent_{mode}.zip
#   3. 途中保存のチチェッキヽイント(CheckpointCallback。ローカル or Drive):
#      {dir}/sac_portfolio_agent_{mode}_{steps}_steps.zip
#
# 💡 修正(2026-11-05): 以前は「完了済み(ローカル/Drive) > 途中保存」の固定優先順位
# だったが、これは「完了済みが常に最も学習が進んだ状態である」という前提に依存して
# いた。手動でDriveからファイルをダウンロードして配置していた運用では問題になら
# なかったが、run_colab_pipeline.py経由でColab上に全自動移植した後は、ある学習実行が
# 完走せずに中断される(セキーン切断等、本プロジェクトで頻発する事複)と、
# チチェッキヽイントの方が完了済みモデルより学習が進んでいても気づかれず、古い完了済み
# モデルが黙ってそのまま使われ続けてしまう危険があった。
# そのため、存在する候補(ローカル完了済み・Drive完了済み・最新チチェッキヽイント)を
# すべて集め、保存時刻(mtime)が最も新しいものを選ぶ方式に変更する。shutil.copy2は
# メタデータ(mtime含む)を保持するため、ローカル完了済みとDriveへコピーされた完了済み
# のmtimeは通常一致し、この比較に悪影響はない。同じmtimeの場合のみ、ローカル完了済み
# > Drive完了済み > チチェッキヽイントの順で優先する(従来の優先順位を同値時の
# タイブレーデクとして維持する)。

DRIVE_FINAL_MODEL_DIR = "/content/drive/MyDrive/saved_models_rl"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/FX_AI_Models/rl_checkpoints"
LOCAL_FINAL_MODEL_DIR = "saved_rl_models"
LOCAL_CHECKPOINT_DIR = "saved_rl_models/checkpoints"


def get_checkpoint_dirs():
    """
    途中保存の保存先候補ディレキトリを返す(先頭が実際の保存先、全体が検索対象)。
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
    モデル本体のチチェッキヽイントパスから、対応するvecnormalize/replay_bufferの
    パスを組み立てる(CheckpointCallbackの命名規則: {prefix}_{type}{steps}_steps.{ext})。
    """
    m = re.match(r"^(.*)_(\d+)_steps\.zip$", os.path.basename(model_ckpt_path))
    prefix, steps = m.group(1), m.group(2)
    return os.path.join(os.path.dirname(model_ckpt_path), f"{prefix}_{checkpoint_type}{steps}_steps.pkl")


def resolve_model_paths(mode):
    """
    実際に使うべきモデル(.zip)・VecNormalize統計(.pkl)・リプレイバファ(.pkl)の
    パスを解決する。存在する候補(ローカル完了済み・Drive完了済み・最新チチェッキヽイント)
    を保存時刻(mtime)で比較し、最も新しいものを選ぶ(詳細はファイル先頭のコメント参照)。

    戻り値: (model_path, vecnorm_path, replay_buffer_path, description) のタプル。
    replay_buffer_path は見つからない場合 None になりうる(呼び出し側で
    os.path.exists を確認すること。途中保存には元々含まれていない設計のため)。
    何も見つからなければ (None, None, None, None) を返す。
    """
    # (mtime, priority, model_path, vecnorm_path, replay_path, description)
    # priority は同じmtimeの場合のタイブレーキ用(小さいほど優先)。
    candidates = []

    local_model = os.path.join(LOCAL_FINAL_MODEL_DIR, f"sac_portfolio_agent_{mode}.zip")
    local_vecnorm = os.path.join(LOCAL_FINAL_MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(local_model) and os.path.exists(local_vecnorm):
        local_replay = os.path.join(LOCAL_FINAL_MODEL_DIR, f"sac_replay_buffer_{mode}.pkl")
        candidates.append((
            os.path.getmtime(local_model), 0, local_model, local_vecnorm, local_replay,
            "ローカルの完了済みモデル",
        ))

    drive_model = os.path.join(DRIVE_FINAL_MODEL_DIR, f"sac_portfolio_agent_{mode}.zip")
    drive_vecnorm = os.path.join(DRIVE_FINAL_MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(drive_model) and os.path.exists(drive_vecnorm):
        drive_replay = os.path.join(DRIVE_FINAL_MODEL_DIR, f"sac_replay_buffer_{mode}.pkl")
        candidates.append((
            os.path.getmtime(drive_model), 1, drive_model, drive_vecnorm, drive_replay,
            "Google Drive上の完了済みモデル",
        ))

    latest_ckpt = find_latest_checkpoint(get_checkpoint_dirs(), mode)
    if latest_ckpt is not None:
        vecnorm_ckpt = checkpoint_companion_path(latest_ckpt, "vecnormalize_")
        if os.path.exists(vecnorm_ckpt):
            replay_ckpt = checkpoint_companion_path(latest_ckpt, "replay_buffer_")
            candidates.append((
                os.path.getmtime(latest_ckpt), 2, latest_ckpt, vecnorm_ckpt, replay_ckpt,
                f"途中保存のチチェッキヽイント({latest_ckpt})",
            ))

    if not candidates:
        return None, None, None, None

    if len(candidates) > 1:
        logging.info("📋 モデル候補を保存時刻(mtime)で比較します:")
        for c_mtime, _priority, _model_path, _vecnorm_path, _replay_path, c_description in candidates:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c_mtime))
            logging.info(f"   [候補] {c_description}  (保存時刻: {when})")

    best = max(candidates, key=lambda c: (c[0], -c[1]))
    _best_mtime, best_priority, model_path, vecnorm_path, replay_path, description = best

    has_completed_candidate = any(priority in (0, 1) for _, priority, *_ in candidates)
    if best_priority == 2 and has_completed_candidate:
        logging.warning(
            "⚠️ 完了済みモデルより保存時刻が新しいチチェッキヽイントが見つかったため、"
            f"そちらを使用します: {description}"
        )
        logging.warning(
            "   完了済みモデルは、学習が完走せず中断された古い状態である可能性があります。"
        )

    return model_path, vecnorm_path, replay_path, description
