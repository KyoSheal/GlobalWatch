#!/usr/bin/env python3
"""Regression: UI equity sanitize removes weekend/off-hours/blackout points."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_time_filter import in_market_window, parse_iso_to_utc, sanitize_equity_rows, to_market_dt


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    rows = [
        {"time": "2026-02-13T10:30:00-05:00", "equity": 100000.0},  # keep
        {"time": "2026-02-15T12:00:00-05:00", "equity": 100000.0},  # blackout+weekend
        {"time": "2026-02-13T20:00:00-05:00", "equity": 100050.0},  # off-hours
        {"time": "2026-02-14T10:00:00-05:00", "equity": 100040.0},  # weekend
    ]

    clean_rows, stats = sanitize_equity_rows(
        rows,
        market_tz="America/New_York",
        open_time_et="09:30",
        close_time_et="16:00",
        open_grace_min=15,
        close_grace_min=10,
        drop_weekends=True,
        drop_offhours=True,
        blackout_dates_market={"2026-02-15"},
    )

    if len(clean_rows) != 1:
        return _fail(f"expected 1 kept row, got {len(clean_rows)} stats={stats}")

    for row in clean_rows:
        dt_utc = parse_iso_to_utc(row.get("time"))
        dt_market = to_market_dt(dt_utc, "America/New_York")
        if dt_market.date().isoformat() == "2026-02-15":
            return _fail("blackout date 2026-02-15 still present")
        if dt_market.weekday() >= 5:
            return _fail("weekend row still present")
        if not in_market_window(dt_market, "09:30", "16:00", 15, 10):
            return _fail("off-hours row still present")

    if int(stats.get("dropped_blackout", 0)) < 1:
        return _fail(f"expected dropped_blackout>=1 stats={stats}")
    if int(stats.get("dropped_offhours", 0)) < 1:
        return _fail(f"expected dropped_offhours>=1 stats={stats}")
    if int(stats.get("dropped_weekend", 0)) < 1:
        return _fail(f"expected dropped_weekend>=1 stats={stats}")

    print("[PASS] ui_equity_sanitize_weekend_blackout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
