#!/usr/bin/env python3
"""T63: minimal regression test for min_keep_turnover_ratio greedy-filter fallback."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trading import PaperTradingEngine


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _build_engine_stub() -> PaperTradingEngine:
    eng = PaperTradingEngine.__new__(PaperTradingEngine)
    eng.current_regime = {}
    eng.config = {"objectives": {"max_weight_per_asset": 0.25}}
    return eng


def main() -> int:
    engine = _build_engine_stub()
    total_equity = 100000.0
    trades = [
        {
            "ticker": "AAA",
            "side": "BUY",
            "desired_trade_value": 500.0,
            "delta_weight": 0.0050,
            "priority": "normal",
        },
        {
            "ticker": "BBB",
            "side": "BUY",
            "desired_trade_value": 1500.0,
            "delta_weight": 0.0005,
            "priority": "normal",
        },
        {
            "ticker": "CCC",
            "side": "SELL",
            "desired_trade_value": 2000.0,
            "delta_weight": 0.0005,
            "priority": "normal",
        },
    ]

    filtered, diag = PaperTradingEngine.filter_trades_greedy(
        engine,
        trades,
        total_equity=total_equity,
        turnover_cap=None,
        max_trades_per_cycle=10,
        min_keep_trades=0,
        min_keep_turnover_ratio=0.03,
        min_trade_notional=200.0,
        min_trade_delta_w=0.002,
        cost_bps=0.0008,
        hard_predicate=lambda tr: bool(tr.get("hard", False)),
        hard_sell_overweight=False,
    )
    if not isinstance(diag, dict):
        return _fail("diag is not dict")
    if not bool(diag.get("min_keep_turnover_triggered", False)):
        return _fail(f"expected min_keep_turnover_triggered=True, got {diag.get('min_keep_turnover_triggered')}")
    if int(diag.get("min_keep_turnover_added", 0) or 0) <= 0:
        return _fail(f"expected added trades > 0, got {diag.get('min_keep_turnover_added')}")

    ratio_after = float(diag.get("planner_turnover_ratio_after", 0.0) or 0.0)
    reason = str(diag.get("min_keep_turnover_reason", ""))
    if ratio_after + 1e-12 < 0.03 and reason != "blocked_by_cap":
        return _fail(f"ratio_after {ratio_after:.6f} below threshold without blocked_by_cap")
    if len(filtered) < 2:
        return _fail(f"expected fallback to add trades, got n_filtered={len(filtered)}")

    # Default behavior unchanged when cfg=0.0 (feature off).
    filtered_default, diag_default = PaperTradingEngine.filter_trades_greedy(
        engine,
        trades,
        total_equity=total_equity,
        turnover_cap=None,
        max_trades_per_cycle=10,
        min_keep_trades=0,
        min_keep_turnover_ratio=0.0,
        min_trade_notional=200.0,
        min_trade_delta_w=0.002,
        cost_bps=0.0008,
        hard_predicate=lambda tr: bool(tr.get("hard", False)),
        hard_sell_overweight=False,
    )
    if bool(diag_default.get("min_keep_turnover_triggered", False)):
        return _fail("expected triggered=False when cfg=0.0")
    if int(diag_default.get("min_keep_turnover_added", 0) or 0) != 0:
        return _fail("expected min_keep_turnover_added=0 when cfg=0.0")
    if str(diag_default.get("min_keep_turnover_reason", "")) != "disabled":
        return _fail(f"expected reason=disabled when cfg=0.0, got {diag_default.get('min_keep_turnover_reason')}")
    if len(filtered_default) != 1:
        return _fail(f"default behavior changed unexpectedly, expected 1 filtered trade got {len(filtered_default)}")

    print("[PASS] min_keep_turnover_minimal")
    print(
        "[INFO] "
        f"before={float(diag.get('planner_turnover_ratio_before', 0.0) or 0.0):.4f} "
        f"after={ratio_after:.4f} "
        f"triggered={bool(diag.get('min_keep_turnover_triggered', False))} "
        f"added={int(diag.get('min_keep_turnover_added', 0) or 0)} "
        f"reason={reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

