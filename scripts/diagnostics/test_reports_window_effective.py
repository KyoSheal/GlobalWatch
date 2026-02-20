#!/usr/bin/env python3
"""Step 3.1 test: reports/statistics effective window status logic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_reports_window import select_window_effective_count


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    no_data = select_window_effective_count(0, "1M")
    if str(no_data.get("status", "")) != "no_data":
        return _fail(f"expected no_data status, got {no_data}")

    till_now = select_window_effective_count(12, "1M")
    if str(till_now.get("status", "")) != "till_now":
        return _fail(f"expected till_now status, got {till_now}")
    if int(till_now.get("effective_days", -1)) != 12:
        return _fail(f"expected effective_days=12, got {till_now}")
    msg = str(till_now.get("message", ""))
    if "Till now" not in msg or "12/21" not in msg:
        return _fail(f"unexpected till_now message: {msg!r}")

    ok = select_window_effective_count(30, "1M")
    if str(ok.get("status", "")) != "ok":
        return _fail(f"expected ok status, got {ok}")
    if int(ok.get("effective_days", -1)) != 21:
        return _fail(f"expected effective_days=21, got {ok}")

    print("[PASS] reports_window_effective")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

