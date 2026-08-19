#!/usr/bin/env bash
# Google Driveにアップロードした単体のアナライザモデル(.pth)だけを取得し、
# saved_model/ 内の既存ファイルを上書きする。
# labeled_data/images 等をまとめて取得する fetch_training_data.sh とは別に、
# モデルファイル1つだけを差し替えたい場合に使う。
#
# 使い方:
#   bash fetch_latest_model.sh "<Google DriveのファイルIDまたは共有URL>" [mode]
#
#   mode を省略した場合は "short" として saved_model/best_total_model_short.pth
#   を上書きする。medium/long を更新したい場合は第2引数で指定する。
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "使い方: bash fetch_latest_model.sh \"<Google DriveのファイルIDまたは共有URL>\" [mode]"
  exit 1
fi

SRC="$1"
MODE="${2:-short}"
DEST="saved_model/best_total_model_${MODE}.pth"

mkdir -p saved_model

if [ -f "$DEST" ]; then
  BACKUP="${DEST}.bak_$(date +%Y%m%d_%H%M%S)"
  mv "$DEST" "$BACKUP"
  echo "=== 既存モデルを退避: $DEST -> $BACKUP ==="
fi

pip install --quiet gdown

echo "=== Google Driveから最新モデルをダウンロード中 ==="
gdown --fuzzy "$SRC" -O "$DEST"

echo ""
echo "=== 完了 ==="
ls -lh "$DEST"
