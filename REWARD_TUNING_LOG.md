# 報酬関数・学習安定化 修正ログ

`datafetch_modeltrain`ブランチの強化学習(SAC)エージェントについて、
2026-09-01のバックテストで確認された学習崩壊(critic_lossの発散・破産)を受けて
実施する、報酬関数および学習安定化に関する修正を記録する。

## 記録方針

- 修正は実施順に追記し、**サマリ表は常に最新のエントリを一番上に追加**する
- 各エントリは「問題点」「修正前のプログラム」「修正後のプログラム」「期待される効果」を最低限記載する
- 実際に学習・テストして結果が出た時点で、該当エントリに「検証結果」を追記する
- ステータスは `未検証` → `検証中` → `効果あり` / `効果なし` / `一部効果あり` のいずれかで管理する

## サマリ

| # | 日付 | 対象ファイル | 概要 | ステータス |
|---|---|---|---|---|
| 001 | 2026-09-01 | portfolio_env.py | 資産最高値ボーナス(NEW_HIGH_BONUS_COEF)に上限を設定 | 未検証 |

---

## #001: 資産最高値ボーナス(NEW_HIGH_BONUS_COEF)に上限を設定

- **日付**: 2026-09-01
- **対象ファイル**: `portfolio_env.py`
- **関連する崩壊事象**: 2026-09-01実施の継続学習(cumulative total_timesteps 約2.2M〜2.5M)にて、
  critic_loss が一時的に 201,000 まで急上昇(通常時は数百〜数千台)。同時にent_coefが
  0.056→0.027まで単調に低下し、テスト時にUSDJPYが上限ロット(-5.00lot)のまま600ステップ以上
  固定されるなど、方策の探索性喪失を疑わせる挙動が見られた。テストは総ステップ6103/10148で
  破産(初期資金の50%割れ)により強制終了。

### 問題点

`NEW_HIGH_BONUS_COEF`による報酬加算は `(equity - high_water_mark)` に比例して無制限に増加する。
他の報酬項目(実現損益・含み損益)は概ね1〜10程度のスケールに収まっている一方、資産が急騰した
瞬間だけ突出して大きい報酬(例: 100万円の含み益急増で+20.0相当)が発生し得る。VecNormalizeの
報酬正規化は移動平均ベースのため、このような突発的な外れ値に対しては正規化が追いつかず、
critic学習を不安定化させる一因になっている可能性がある。また、含み益(未実現)ベースで発火する
ボーナスであるため、実現利益に直結しない「見せかけの資産上昇」を過大評価している可能性もあり、
2,000,000ステップ時点でも明確な利益化に至っていない一因とも考えられる。

### 修正前のプログラム

```python
NEW_HIGH_BONUS_COEF = 2.0
DRAWDOWN_PENALTY_COEF = 10.0
REALIZED_PNL_WEIGHT = 1.0
UNREALIZED_PNL_WEIGHT = 0.2
MIN_RR_RATIO = 1.5
CONSECUTIVE_LOSS_PENALTY_COEF = 0.5
COUNTERFACTUAL_PENALTY_COEF = 1.0
if self.equity > self.high_water_mark:
reward += (self.equity - self.high_water_mark)/self.initial_balance * 100 * NEW_HIGH_BONUS_COEF

###修正後プログラム
NEW_HIGH_BONUS_COEF = 2.0
NEW_HIGH_BONUS_MAX = 5.0  # 💡 追加: 1ステップあたりの上限。資産急騰時に他の報酬項目より
                          # 桁違いに大きくなり、criticの学習を不安定化させていた可能性への対策。
DRAWDOWN_PENALTY_COEF = 10.0
REALIZED_PNL_WEIGHT = 1.0
UNREALIZED_PNL_WEIGHT = 0.2
MIN_RR_RATIO = 1.5
CONSECUTIVE_LOSS_PENALTY_COEF = 0.5
COUNTERFACTUAL_PENALTY_COEF = 1.0

if self.equity > self.high_water_mark:
 new_high_bonus = (self.equity - self.high_water_mark)/self.initial_balance * 100 * NEW_HIGH_BONUS_COEF
 reward += min(new_high_bonus, NEW_HIGH_BONUS_MAX)


資産急騰時の報酬の突出を抑えることで、VecNormalizeの正規化が追従しやすくなり、
criticのTDターゲットの不連続(≒ critic_lossの急上昇)が緩和されることを期待する
ボーナス自体は残すため、「新高値を更新する」という行動へのインセンティブは維持される
この修正のみで崩壊が再発しない場合、報酬関数側の要因が支配的だったと判断できる
崩壊が再発する場合は、次のステップとして #002(entropy下限のクランプ、sac_grad_clip.py)を検討する
