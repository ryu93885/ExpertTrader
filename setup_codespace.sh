#!/usr/bin/env bash
# GitHub Codespaces (または同等のLinux環境) でポートフォリオRLの再学習を行うための
# セットアップスクリプト。datafetch_modeltrain ブランチのルートで実行してください。
#
#   bash setup_codespace.sh
#
set -euo pipefail

BRANCH="datafetch_modeltrain"

echo "=== 1. ブランチの確認・最新化 ==="
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull origin "$BRANCH"

echo ""
echo "=== 2. Python依存パッケージのインストール ==="
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== 3. 必要ディレクトリの作成(データ自体はここには含まれません) ==="
mkdir -p labeled_data processed_data images saved_model saved_models saved_rl_models test_data predictions logs

echo ""
echo "=== セットアップ完了 ==="
echo "以下のデータが揃っているか確認してください(git管理外のため、別途配置が必要です):"
echo "  - labeled_data/*_short_scaled_train.csv / *_short_scaled_test.csv"
echo "  - labeled_data/*_short_labeled_train.csv / *_short_labeled_test.csv"
echo "  - images/{symbol}/short_MTF/*.png (7銘柄分)"
echo "  - saved_model/best_total_model_short.pth (学習済みアナライザ)"
echo ""
echo "揃っていれば、次のステップとして run_portfolio_retrain.sh を実行してください。"
