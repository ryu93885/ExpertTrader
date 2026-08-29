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
echo "  ※ {mode} には実際に使用しているモード(例: medium)を当てはめてください。"
echo "  - labeled_data/*_{mode}_scaled_train.csv / *_{mode}_scaled_test.csv"
echo "    (無ければ、銘柄ごとに split_data.py --symbol <SYMBOL> --mode <mode> を実行してください)"
echo "  - labeled_data/*_{mode}_labeled_train.csv / *_{mode}_labeled_test.csv"
echo "  - images/{symbol}/{mode}_MTF/*.png (7銘柄分)"
echo "  - saved_model/best_total_model_{mode}.pth (学習済みアナライザ)"
echo ""
echo "揃っていれば、run_portfolio_retrain.sh 冒頭の MODE=\"medium\" が実際のモードと"
echo "一致していることを確認したうえで、次のステップとして実行してください。"
