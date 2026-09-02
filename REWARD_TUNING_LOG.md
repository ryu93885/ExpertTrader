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
| 002 | 2026-09-01 | sac_grad_clip.py | entropy係数(ent_coef)の下限を0.01に設定 | 効果あり |
| 001 | 2026-09-01 | portfolio_env.py | 資産最高値ボーナス(NEW_HIGH_BONUS_COEF)に上限を設定 | 効果あり |

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
```

```python
        if self.equity > self.high_water_mark:
            reward += (self.equity - self.high_water_mark)/self.initial_balance * 100 * NEW_HIGH_BONUS_COEF
```

### 修正後のプログラム

```python
NEW_HIGH_BONUS_COEF = 2.0
NEW_HIGH_BONUS_MAX = 5.0  # 💡 追加: 1ステップあたりの上限。資産急騰時に他の報酬項目より
                          # 桁違いに大きくなり、criticの学習を不安定化させていた可能性への対策。
DRAWDOWN_PENALTY_COEF = 10.0
REALIZED_PNL_WEIGHT = 1.0
UNREALIZED_PNL_WEIGHT = 0.2
MIN_RR_RATIO = 1.5
CONSECUTIVE_LOSS_PENALTY_COEF = 0.5
COUNTERFACTUAL_PENALTY_COEF = 1.0
```

```python
        if self.equity > self.high_water_mark:
            new_high_bonus = (self.equity - self.high_water_mark)/self.initial_balance * 100 * NEW_HIGH_BONUS_COEF
            reward += min(new_high_bonus, NEW_HIGH_BONUS_MAX)
```

### 期待される効果

- 資産急騰時の報酬の突出を抑えることで、VecNormalizeの正規化が追従しやすくなり、
  criticのTDターゲットの不連続(≒ critic_lossの急上昇)が緩和されることを期待する
- ボーナス自体は残すため、「新高値を更新する」という行動へのインセンティブは維持される
- この修正のみで崩壊が再発しない場合、報酬関数側の要因が支配的だったと判断できる
- 崩壊が再発する場合は、次のステップとして #002(entropy下限のクランプ、sac_grad_clip.py)を検討する

### 検証結果

#002と同時に適用した新規学習にて、critic_lossの異常スパイクは一切観測されず、
テストも10107/10107ステップを完走(破産なし)、資産レンジ9.84M〜11.2M、
終値+11.1%という会話全体で最良の結果となった。ただし#002と同時適用のため、
このスコア改善のうちどこまでが本修正単体の効果かは厳密には切り分けられない。


---

## #002: entropy係数(ent_coef)の下限を0.01に設定

- **日付**: 2026-09-01
- **対象ファイル**: `sac_grad_clip.py`
- **関連する崩壊事象**: #001適用後の新規学習にて、total_timesteps 46.6万時点でent_coefが
  0.185→0.0000748まで単調に崩壊。critic_loss・actor_lossも同時にほぼ0へ収束しており、
  方策が探索性を失いつつある兆候が見られた。critic_lossの異常なスパイクは今回は観測されず、
  #001は一定の効果があったと考えられるが、entropy崩壊という別の根本原因が残っていたことが判明。

### 問題点

SACのent_coef(entropy自動調整の温度パラメータ)に下限が設定されておらず、auto-tuningが
際限なく0に近づくことを許容していた。2026-09-01の新規学習では、わずか46.6万ステップで
ent_coefが0.0000748まで低下し、これは前回の学習崩壊(破産)直前に観測された値
(0.027、cumulative約240万ステップ時点)よりも1桁以上低い水準に、5分の1未満のステップ数で
到達している。ent_coefがほぼ0になると方策はほぼ決定論的になり探索が失われ、リプレイバッファに
溜まる経験の多様性が急減する。これによりcriticが狭い経験分布に過剰適合し、テスト時にポジション
サイズが特定の値に固定され続ける(前回のUSDJPY -5.00lot固定など)という挙動につながって
いたと考えられる。

### 修正前のプログラム

```python
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

GRAD_CLIP_MAX_NORM = 10.0


class SACWithGradClip(SAC):
```

```python
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
```

### 修正後のプログラム

```python
import math
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

GRAD_CLIP_MAX_NORM = 10.0
ENT_COEF_MIN = 0.01


class SACWithGradClip(SAC):
```

```python
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.clamp(torch.exp(self.log_ent_coef.detach()), min=ENT_COEF_MIN)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                with torch.no_grad():
                    self.log_ent_coef.data.clamp_(min=math.log(ENT_COEF_MIN))
```

### 期待される効果

- ent_coefが0.01を下回らなくなるため、学習後半でも一定水準の探索性が維持され、
  方策が決定論的に固まってしまう(特定のポジションサイズに固執する)ことを防ぐ
- 探索性が保たれることで、リプレイバッファの経験の多様性が維持され、
  criticの過学習的な収束(loss→0)も緩和されることを期待する
- ENT_COEF_MIN=0.01は、今回のログで健全に見えた147,422ステップ時点の値(0.0148)に
  近い水準を参考にした、精密なチューニングを要しない緩めの安全弁

### 検証結果

ent_coefがtotal_timesteps 207,379以降、正確に0.01でピン留めされることを確認。
ent_coef_lossが強く負(-126〜-163)であるにもかかわらず値が下限を下回らず、
クランプが設計通り機能している。actor_lossも前回のように0へ収束せず-13〜-14で
安定して推移し、探索性の崩壊は再発しなかった。#001と合わせて適用した結果、
テストは10107/10107ステップ完走・破産なし・終値+11.1%と、会話全体で最良の
結果となった。
