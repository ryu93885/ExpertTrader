import logging
from collections import defaultdict

# 💡 test_portfolio.py に実装した性能集計ロジックを、sweep_entry_threshold.py
# (動的エントリー閾値のパラメータ探索)でも再利用するために共通化したモジュール。
# portfolio_env.py の step() が返す info["realized_pnl_by_symbol"] は決済が
# 発生したステップのみ非ゼロになるため、これを「決済1件=1トレード」として
# 集計すれば、勝率・プロフィットファクターなどが算出できる(portfolio_env.py
# 自体や学習ロジックには依存しない)。


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

    peak = -float("inf")
    max_drawdown_pct = 0.0
    for eq in equities:
        peak = max(peak, eq)
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak - eq) / peak * 100.0)

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
