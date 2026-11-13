#!/usr/bin/env bash
# Google Driveにアップロード済みの学習データZIP(labeled_data/images/saved_model等)を
# Codespacesへダウンロードし、リポジトリ直下に展開する。
#
# 事前条件:
#   - ZIPをGoogle Driveにアップロード済みであること(export_dataset_zip.py で作成した
#     ものでも、labeled_data/images/saved_model フォルダを手動でzip化したものでも可)
#   - そのZIPが「リンクを知っている全員が閲覧可」等、gdownからダウンロード可能な
#     共有設定になっていること
#   - ZIP内部のパスが labeled_data/... images/... saved_model/... のように、
#     リポジトリ直下からの相対パスになっていること(export_dataset_zip.py の
#     出力はこの形式です)
#
# 使い方:
#   bash fetch_training_data.sh "<Google DriveのファイルIDまたは共有URL>"
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "使い方: bash fetch_training_data.sh \"<Google DriveのファイルIDまたは共有URL>\""
  exit 1
fi

SRC="$1"
ZIP_PATH="training_data.zip"

echo "=== gdownのインストール ==="
pip install --quiet --upgrade gdown

echo ""
echo "=== Google Driveからダウンロード中 ==="
# 💡 修正: gdownの新しいバージョン(6.x系)では --fuzzy オプションが廃止され、
# URLからのID自動抽出が標準動作になっているため、フラグを付けずに呼び出す。
gdown "$SRC" -O "$ZIP_PATH"

echo ""
echo "=== リポジトリ直下に展開中 ==="
unzip -q -o "$ZIP_PATH" -d .
rm -f "$ZIP_PATH"

echo ""
echo "=== 完了。配置内容を確認します ==="
for d in labeled_data images saved_model; do
  if [ -d "$d" ]; then
    count=$(find "$d" -type f | wc -l)
    echo "  $d/ : ${count} ファイル"
  else
    echo "  $d/ : 見つかりませんでした(ZIPの中身を確認してください)"
  fi
done
