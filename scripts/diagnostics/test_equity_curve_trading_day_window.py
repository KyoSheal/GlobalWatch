#!/usr/bin/env python3
"""Step 2 test: trading-day window filter for equity curve."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_equity_window import filter_df_by_trading_day_window


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _build_sample_df():
    # 10 business days, 2 points each day.
    business_days = pd.bdate_range("2026-01-05", periods=10, freq="B")
    rows = []
    equity = 100000.0
    for day in business_days:
        t1 = pd.Timestamp(day).replace(hour=9, minute=30)
        t2 = pd.Timestamp(day).replace(hour=12, minute=30)
        rows.append({"time": t1, "equity": equity})
        rows.append({"time": t2, "equity": equity + 50.0})
        equity += 100.0
    return pd.DataFrame(rows)


def main() -> int:
    df = _build_sample_df()

    week_df, week_avail, week_req = filter_df_by_trading_day_window(df, "1W")
    week_kept = int(week_df["trade_date"].nunique()) if "trade_date" in week_df.columns and not week_df.empty else 0
    if week_kept > 5:
        return _fail(f"1W unique trade_date={week_kept}, expected <= 5")
    if week_req != 5:
        return _fail(f"1W required_days={week_req}, expected 5")

    month_df, month_avail, month_req = filter_df_by_trading_day_window(df, "1M")
    month_kept = int(month_df["trade_date"].nunique()) if "trade_date" in month_df.columns and not month_df.empty else 0
    if month_kept != 10:
        return _fail(f"1M unique trade_date={month_kept}, expected 10 (till now)")
    if month_req != 21:
        return _fail(f"1M required_days={month_req}, expected 21")
    if month_avail != 10 or week_avail != 10:
        return _fail(f"available_days mismatch: week={week_avail}, month={month_avail}, expected 10")

    print(f"[PASS] equity_curve_trading_day_window 1W_kept={week_kept} 1M_kept={month_kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

