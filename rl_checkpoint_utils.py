import os
import re
import glob
import time
import csv
import logging
import hashlib
import json
import shutil
from datetime import datetime

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


# 💡 追加(2026-11-21の調査で判明した問題への対応): 2026-11-20の学習で、学習完了後の
# 「完了済みモデル」が実は学習後半の不安定化で崩れた状態であり、2,500,000ステップ時点の
# チェックポイントの方が実バックテストで明確に良い成績(総合収益率+5.29%・最大DD12.41%)
# だったことが判明した。resolve_model_paths()は「保存時刻が最も新しいもの」を機械的に
# 選ぶだけで、「それが実際に検証済みの良いモデルか」までは判断できない。
# 今後、保存時刻の新しいチェックポイント・完了済みモデルに気づかず切り替わってしまう
# ことを防ぐため、「検証の結果これが最良である」と明示的に記録したモデルのSHA256ハッシュと
# 照合する仕組みを追加する。ファイル名やmtimeではなく中身のハッシュで比較するため、
# コピー・リネームされても正しく同一性を判定できる。
BEST_MODEL_RECORD_LOCAL_DIR = "saved_rl_models"
BEST_MODEL_RECORD_DRIVE_DIR = "/content/drive/MyDrive/saved_models_rl"


def _best_model_record_paths(mode, local_dir=None):
    """
    local_dirを省略した場合はBEST_MODEL_RECORD_LOCAL_DIR(Colab側の相対パス)を使う。
    portfolio_trading_bot.py(Windows側の本番ボット)はRL_MODEL_DIRが別の場所
    (例: "../datafetch_modeltrain/saved_rl_models")にあるため、呼び出し側で
    明示的に指定する。
    """
    filename = f"best_model_record_{mode}.json"
    local_path = os.path.join(local_dir or BEST_MODEL_RECORD_LOCAL_DIR, filename)
    drive_path = os.path.join(BEST_MODEL_RECORD_DRIVE_DIR, filename)
    return local_path, drive_path


def compute_file_sha256(path):
    """モデルファイル(.zip)の内容からSHA256ハッシュを計算する。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_best_model_record(mode, local_dir=None):
    """
    記録されている「検証済み最良モデル」の情報を読み込む。
    ローカルに無ければGoogle Driveからの復元を試みる(local_dir指定時、Driveは
    通常存在しないため実質スキップされる)。記録自体が存在しない場合はNoneを返す
    (まだ一度もsave_best_model_record/--mark-as-bestが使われていない状態)。
    """
    local_path, drive_path = _best_model_record_paths(mode, local_dir)
    if not os.path.exists(local_path) and os.path.exists(drive_path):
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copy2(drive_path, local_path)

    if not os.path.exists(local_path):
        return None
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"⚠️ best_model_record_{mode}.jsonの読み込みに失敗しました: {e}")
        return None


def get_merged_csv_date_range(csv_path="portfolio_merged_data_train_with_preds.csv"):
    """
    💡 追加(2026-11-21の検討): 「どの期間のデータで学習したか」を可視化するため、
    portfolio_merged_data_*_with_preds.csvの"time"列から先頭・末尾の日時を取得する。
    pandas等の重い依存を避けるため、csvモジュールで該当列だけを読む
    (このファイル自体はtorch/pandas非依存のまま保つ)。取得できない場合は
    Noneを返す(呼び出し元で必須にはしない)。
    """
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            time_idx = header.index("time")
            first_time, last_time = None, None
            for row in reader:
                if not row:
                    continue
                if first_time is None:
                    first_time = row[time_idx]
                last_time = row[time_idx]
        if first_time is None:
            return None
        return {"start": first_time, "end": last_time}
    except Exception as e:
        logging.warning(f"⚠️ {csv_path}の学習データ期間の取得に失敗しました: {e}")
        return None


def save_best_model_record(mode, model_path, performance_summary=None, note="", local_dir=None,
                            num_timesteps=None, train_data_range=None):
    """
    指定したモデルファイルを「検証済み最良モデル」として記録する(ローカル+Drive)。
    test_portfolio.pyの--mark-as-bestから、実際のバックテスト結果を確認した上で
    呼ばれることを想定している(自動では呼ばれない)。
    num_timesteps/train_data_rangeは、手動コピー運用でも「いつ・どの期間のデータで・
    何ステップ学習したものか」を後から確認できるようにするための追加情報(任意)。
    """
    record = {
        "sha256": compute_file_sha256(model_path),
        "model_filename": os.path.basename(model_path),
        "marked_at": datetime.now().isoformat(timespec="seconds"),
        "performance_summary": performance_summary or {},
        "note": note,
    }
    if num_timesteps is not None:
        record["num_timesteps"] = int(num_timesteps)
    if train_data_range is not None:
        record["train_data_range"] = train_data_range
    local_path, drive_path = _best_model_record_paths(mode, local_dir)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logging.info(f"📝 [{mode}] 検証済み最良モデルとして記録しました: {local_path}")

    if os.path.exists("/content/drive/MyDrive"):
        try:
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            shutil.copy2(local_path, drive_path)
        except Exception as e:
            logging.warning(f"⚠️ best_model_record_{mode}.jsonのGoogle Driveへのバックアップに失敗しました: {e}")
    return record


def verify_against_best_record(model_path, mode, local_dir=None):
    """
    model_pathのモデルが、記録されている「検証済み最良モデル」と内容が一致するかを
    検証し、結果をログに出力する。記録がまだ無い場合はエラーにはせず、その旨を伝えるのみ。
    local_dirは記録(best_model_record_{mode}.json)の置き場所を指定する
    (省略時はBEST_MODEL_RECORD_LOCAL_DIR。portfolio_trading_bot.py等、Colab側と
    異なる場所にモデルを置く場合に指定する)。

    戻り値: True(一致) / False(不一致) / None(記録が存在しない)
    """
    record = load_best_model_record(mode, local_dir)
    if record is None:
        logging.info(
            f"ℹ️ [{mode}] 検証済み最良モデルの記録がまだありません"
            "(test_portfolio.pyで--mark-as-bestを使うと記録できます)。"
        )
        return None

    current_hash = compute_file_sha256(model_path)
    if current_hash == record.get("sha256"):
        logging.info(f"✅ [{mode}] 使用するモデルは、記録されている検証済み最良モデルと一致しています。")
        return True

    logging.warning(
        f"⚠️ [{mode}] 使用しようとしているモデル({os.path.basename(model_path)})は、"
        f"記録されている検証済み最良モデル({record.get('model_filename')}、"
        f"記録日時={record.get('marked_at')})と一致しません。意図した状態か確認してください。"
    )
    perf = record.get("performance_summary")
    if perf:
        logging.warning(f"   記録されている最良モデルの成績: {perf}")
    return False


def log_model_summary(model, model_path, mode, local_dir=None):
    """
    💡 追加(2026-11-21の検討への対応): 手動コピー運用では、使おうとしているモデルが
    本当に意図したもの(いつ・どのステップ数まで・どの期間のデータで学習したか)を
    目視で確認しづらいという課題があったため、モデルを使用する直前にその来歴を
    まとめて表示する。verify_against_best_record(ハッシュの一致判定)に加えて、
    モデル自身が持つ情報(num_timesteps)・記録済みの最良モデル情報(学習ステップ数・
    学習データ期間・バックテスト成績)を合わせて表示する。
    """
    logging.info("=" * 60)
    logging.info(f"📦 [{mode}] 使用するモデルの情報")
    logging.info("=" * 60)
    logging.info(f"ファイル: {model_path}")
    if os.path.exists(model_path):
        mtime_str = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(timespec="seconds")
        logging.info(
            f"ファイルの更新日時(mtime): {mtime_str}"
            "(コピー方法によっては元の保存日時がそのまま引き継がれている場合があります)"
        )
    if hasattr(model, "num_timesteps"):
        logging.info(f"このモデル自体が記憶している学習ステップ数(num_timesteps): {model.num_timesteps:,}")

    match = verify_against_best_record(model_path, mode, local_dir)

    record = load_best_model_record(mode, local_dir)
    if record is not None:
        logging.info(f"--- 記録されている検証済み最良モデルの情報(記録日時={record.get('marked_at')}) ---")
        if "num_timesteps" in record:
            logging.info(f"    学習ステップ数: {record['num_timesteps']:,}")
        train_range = record.get("train_data_range")
        if train_range:
            logging.info(f"    学習データ期間: {train_range.get('start')} 〜 {train_range.get('end')}")
        perf = record.get("performance_summary")
        if perf:
            logging.info(f"    記録時のバックテスト成績: {perf}")
    logging.info("=" * 60)
    return match
