#!/usr/bin/env bash
# ポートフォリオ強化学習(SAC)の再学習パイプライン。
# portfolio_env.py / merge_portfolio_data.py の整合性修正を反映した状態で、
# 1. データ統合 (merge_portfolio_data.py)
# 2. アナライザ推論結果の事前計算 (precompute_portfolio.py)
# 3. SACエージェントの学習 (train_portfolio_rl.py)
# を順番に実行する。setup_codespace.sh を先に実行し、データが揃っていることを
# 確認してから使ってください。
#
#   bash run_portfolio_retrain.sh
#
set -euo pipefail

MODE="short"
BACKUP_DIR="saved_rl_models/_backup_before_fix_$(date +%Y%m%d_%H%M%S)"

echo "=== 既存のポートフォリオRLモデルを退避 ==="
echo "(壊れた環境 [spread単位/contract_size/USDJPYレート] で学習された旧モデルのため、"
echo " 継続学習ではなく新規学習として再開します)"
mkdir -p "$BACKUP_DIR"
moved=0
for f in \
  "saved_rl_models/sac_portfolio_agent_${MODE}.zip" \
  "saved_rl_models/vec_normalize.pkl" \
  "saved_rl_models/sac_replay_buffer_${MODE}.pkl"
do
  if [ -f "$f" ]; then
    mv "$f" "$BACKUP_DIR/"
    echo "  退避: $f -> $BACKUP_DIR/"
    moved=1
  fi
done
if [ "$moved" -eq 0 ]; then
  echo "  (退避対象の既存モデルは見つかりませんでした。新規学習として開始します)"
fi

echo ""
echo "=== 1/3: データ統合 (merge_portfolio_data.py) ==="
python merge_portfolio_data.py

echo ""
echo "=== 2/3: アナライザ推論結果の事前計算 (precompute_portfolio.py) ==="
python precompute_portfolio.py

echo ""
echo "=== 3/3: ポートフォリオRL(SAC)の学習 (train_portfolio_rl.py) ==="
python train_portfolio_rl.py

echo ""
echo "=== 完了 ==="
echo "学習済みモデル: saved_rl_models/sac_portfolio_agent_${MODE}.zip"
echo "旧モデルのバックアップ: $BACKUP_DIR/"
