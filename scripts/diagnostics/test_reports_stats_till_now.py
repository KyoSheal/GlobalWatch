#!/usr/bin/env python3
"""Step 3.4 regression: reports/stats should use till-now effective window."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_reports_window import select_window_effective_count
from ui_window_presets import get_window_preset


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    preset = get_window_preset("1M")
    if int(preset.get("trading_days", -1)) != 21:
        return _fail(f"1M trading_days expected 21, got {preset.get('trading_days')}")

    win = select_window_effective_count(12, "1M")
    if str(win.get("status", "")) != "till_now":
        return _fail(f"status expected till_now, got {win}")
    if int(win.get("required_days", -1)) != 21:
        return _fail(f"required_days expected 21, got {win.get('required_days')}")
    if int(win.get("effective_days", -1)) != 12:
        return _fail(f"effective_days expected 12, got {win.get('effective_days')}")

    msg = str(win.get("message", ""))
    if "Till now" not in msg or "12/21" not in msg:
        return _fail(f"message mismatch: {msg!r}")

    entries = list(range(12))
    selected = entries[-int(win["effective_days"]):]
    if len(selected) != 12:
        return _fail(f"selected length expected 12, got {len(selected)}")

    print("[PASS] reports_stats_till_now available=12 required=21 effective=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

