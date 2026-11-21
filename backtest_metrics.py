import logging
from collections import defaultdict

import numpy as np

# 💡 test_portfolio.py に実装した性能集計ロジックを、sweep_entry_threshold.py
# (動的エントリー閾値のパラメータ探索)でも再利用するために共通化したモジュール。
# portfolio_env.py の step() が返す info["realized_pnl_by_symbol"] は決済が
# 発生したステップのみ非ゼロになるため、これを「決済1件=1トレード」として
# 集計すれば、勝率・プロフィットファクターなどが算出できる(portfolio_env.py
# 自体や学習ロジックには依存しない)。


def _max_drawdown_pct(equities):
    peak = -float("inf")
    max_dd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100.0)
    return max_dd


def compute_performance_summary(equities, balances, trades, initial_balance, elapsed_days=None):
    """
    equities, balances: 各ステップのEquity/Balanceのリスト
    trades: [{"step":int, "symbol":str, "pnl":float}, ...] (決済のみ)
    initial_balance: バックテスト開始時の資金
    elapsed_days: バックテスト対象期間の実日数(わかる場合のみ。月利換算に使う)

    戻り値: 各種集計値を格納したdict。
    """
    final_equity = equities[-1] if equities else initial_balance
    final_balance = balances[-1] if balances else initial_balance
    total_return_pct = (final_equity - initial_balance) / initial_balance * 100.0

    max_drawdown_pct = _max_drawdown_pct(equities)

    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    win_rate_pct = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = sum(t["pnl"] for t in losses)  # 負の値(または0)
    net_realized_pnl = gross_profit + gross_loss
    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0

    # 💡 追加: 月利換算(複利ベース)。目標(月刉5%)と直接比較できるようにする。
    monthly_return_pct = None
    if elapsed_days and elapsed_days > 0:
        months = elapsed_days / 30.4375
        if months > 0:
            growth = 1.0 + total_return_pct / 100.0
            if growth > 0:
                monthly_return_pct = (growth ** (1.0 / months) - 1.0) * 100.0

    trades_by_symbol = defaultdict(list)
    for t in trades:
        trades_by_symbol[t["symbol"]].append(t["pnl"])

    return {
        "initial_balance": initial_balance,
        "final_equity": final_equity,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "monthly_return_pct": monthly_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": win_rate_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_realized_pnl": net_realized_pnl,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trades_by_symbol": dict(trades_by_symbol),
    }


def log_performance_summary(summary, log=logging):
    log.info("=" * 60)
    log.info("📊 パフォーマンス集計")
    log.info("=" * 60)
    log.info(f"初期資金: {summary['initial_balance']:,.0f}円")
    log.info(f"最終Equity: {summary['final_equity']:,.0f}円 / 最終Balance: {summary['final_balance']:,.0f}円")
    log.info(f"総合収益率(Equity基準): {summary['total_return_pct']:+.2f}%")
    if summary["monthly_return_pct"] is not None:
        log.info(f"月利換算(複利): {summary['monthly_return_pct']:+.2f}%")
    log.info(f"最大ドローダウン: {summary['max_drawdown_pct']:.2f}%")
    log.info(f"総トレード数(決済件数): {summary['total_trades']}")
    log.info(f"勝率: {summary['win_rate_pct']:.1f}% (勝ち{summary['win_count']}件 / 負け{summary['loss_count']}件)")
    log.info(
        f"プロフィットファクター: {summary['profit_factor']:.2f} "
        f"(総利益{summary['gross_profit']:,.0f}円 / 総損失{summary['gross_loss']:,.0f}円)"
    )
    log.info(f"平均利益: {summary['avg_win']:,.0f}円 / 平均損失: {summary['avg_loss']:,.0f}円")
    log.info(f"純実現損益合計: {summary['net_realized_pnl']:,.0f}円")

    if summary["trades_by_symbol"]:
        log.info("--- 銘柄別 ---")
        for sym, pnls in sorted(summary["trades_by_symbol"].items()):
            sym_wins = [p for p in pnls if p > 0]
            sym_win_rate = len(sym_wins) / len(pnls) * 100.0
            log.info(
                f"    [{sym}] 件数={len(pnls)} 勝率={sym_win_rate:.1f}% 実現損益合計={sum(pnls):,.0f}円"
            )
    log.info("=" * 60)


# 💡 追加(2026-11-18の改修): 「GOLDのような低エッジの銘柄が、ポートフォリオ全体の
# リスク低減(分散効果)に寄与している可能性」を、再学習せずに既存のバックテスト結果
# (equities・trades)だけから確認するための分析。他銘柄との相関、および各銘柄を
# 除外した場合の仮想Equity曲線・最大ドローダウンを算出する。
#
# ⚠️ 限界: 「その銘柄の実現損益を差し引く」という近似であり、max_concurrent_positions
# 等による「その銘柄が使っていた枠が他の取引に回っていたら」という玉数の再配分までは
# 再現できない。あくまで一次近似の参考値であり、確定的な結論(特に「共有モデルの学習
# 自体への影響」)にはならない。それを確定させるには、--exclude-symbolsを使った
# 比較学習が必要(train.py参照)。
def analyze_symbol_contribution(equities, trades, symbols):
    """
    equities: 各ステップのEquityのリスト
    trades: [{"step":int, "symbol":str, "pnl":float}, ...] (決済のみ)
    symbols: 分析対象の銘柄リスト(例: PortfolioFXEnvに渡したsymbols)

    戻り値:
      {
        "correlations": {(sym_a, sym_b): 相関係数 or None, ...},
        "what_if_excluding": {
            symbol: {"max_drawdown_pct": ..., "final_equity": ..., "drawdown_delta_pct": ...}
        },
        "actual_max_drawdown_pct": ...,
        "actual_final_equity": ...,
      }
    """
    n = len(equities)

    # 各銘柄の「実現損益の累積系列」をステップ単位で再構築する
    # (決済が起きたステップでのみ増減し、それ以外は直前の値を保持する)
    cumulative_by_symbol = {sym: [0.0] * n for sym in symbols}
    for t in trades:
        sym = t["symbol"]
        if sym not in cumulative_by_symbol:
            continue
        step_idx = min(max(t["step"] - 1, 0), n - 1)  # trades の step は1始まり
        cumulative_by_symbol[sym][step_idx] += t["pnl"]
    for sym in symbols:
        series = cumulative_by_symbol[sym]
        running = 0.0
        for i in range(n):
            running += series[i]
            series[i] = running

    # 1ステップごとの実現損益の変化(リターン代わり)の相関係数
    diffs_by_symbol = {}
    for sym in symbols:
        series = cumulative_by_symbol[sym]
        diffs_by_symbol[sym] = np.diff(series) if len(series) > 1 else np.array([])

    correlations = {}
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            da, db = diffs_by_symbol[sym_a], diffs_by_symbol[sym_b]
            if len(da) < 2 or np.std(da) == 0 or np.std(db) == 0:
                correlations[(sym_a, sym_b)] = None  # 分散0(無取引等)は相関を定義できない
            else:
                correlations[(sym_a, sym_b)] = float(np.corrcoef(da, db)[0, 1])

    actual_max_dd = _max_drawdown_pct(equities)
    what_if_excluding = {}
    for sym in symbols:
        excluded_equity = [equities[i] - cumulative_by_symbol[sym][i] for i in range(n)]
        dd = _max_drawdown_pct(excluded_equity)
        what_if_excluding[sym] = {
            "max_drawdown_pct": dd,
            "final_equity": excluded_equity[-1] if excluded_equity else None,
            "drawdown_delta_pct": dd - actual_max_dd,  # 正なら「除外後の方が悪化する」=分散に寄与していた可能性
        }

    return {
        "correlations": correlations,
        "what_if_excluding": what_if_excluding,
        "actual_max_drawdown_pct": actual_max_dd,
        "actual_final_equity": equities[-1] if equities else None,
    }


def log_symbol_contribution_analysis(analysis, log=logging):
    log.info("=" * 60)
    log.info("🔗 銘柄間の相関・除外時の仮想シミュレーション(再学習なしの参考値)")
    log.info("=" * 60)
    log.info("--- 実現損益の変化の相関係数(1に近いほど同じ動き、0に近いほど無関係、負なら逆の動き) ---")
    for (sym_a, sym_b), corr in sorted(analysis["correlations"].items()):
        corr_str = f"{corr:+.2f}" if corr is not None else "算出不可(片方が無取引等)"
        log.info(f"    {sym_a} - {sym_b}: {corr_str}")

    log.info(
        f"--- 各銘柄を除外した場合の仮想最大ドローダウン(実際: {analysis['actual_max_drawdown_pct']:.2f}%) ---"
    )
    log.info("    ⚠️ その銘柄の実現損益を差し引いた近似値。玉数上限による取引機会の再配分は考慮していません。")
    for sym, info in sorted(analysis["what_if_excluding"].items()):
        direction = "悪化(分散に寄与していた可能性)" if info["drawdown_delta_pct"] > 0.01 else (
            "改善(リスクを増やしていた可能性)" if info["drawdown_delta_pct"] < -0.01 else "ほぼ変化なし"
        )
        log.info(
            f"    [{sym}除外時] 最大ドローダウン={info['max_drawdown_pct']:.2f}% "
            f"(差分 {info['drawdown_delta_pct']:+.2f}pt, {direction})"
        )
    log.info("=" * 60)


def run_backtest_episode(vec_env, model, env_for_dates=None):
    """
    学習済みモデルを1エピソード(データ末尾 or 破産まで)分、決定論的に実行し、
    equities/balances/tradesとdone_reasonを返す。test_portfolio.pyとsweep_entry_threshold.py
    の両方で同じループを使うための共通化。
    """
    obs = vec_env.reset()
    done = False
    equities, balances, trades = [], [], []
    step_count = 0
    done_reason = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done = dones[0]
        info = infos[0]

        equities.append(info["equity"])
        balances.append(info["balance"])

        realized = info.get("realized_pnl_by_symbol", {})
        for sym, pnl in realized.items():
            if pnl != 0:
                trades.append({"step": step_count + 1, "symbol": sym, "pnl": pnl})

        step_count += 1
        if done:
            done_reason = info.get("done_reason")

    return {
        "equities": equities,
        "balances": balances,
        "trades": trades,
        "step_count": step_count,
        "done_reason": done_reason,
    }
