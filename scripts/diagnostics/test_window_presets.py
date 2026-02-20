#!/usr/bin/env python3
"""Step 1 test: window presets + till-now formatter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_window_presets import WINDOW_PRESETS, format_till_now, get_window_preset


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    if len(WINDOW_PRESETS) != 9:
        return _fail(f"len(WINDOW_PRESETS)={len(WINDOW_PRESETS)} expected 9")

    fallback = get_window_preset("abc")
    if int(fallback.get("trading_days", -1)) != 21:
        return _fail(f"fallback trading_days={fallback.get('trading_days')} expected 21")

    text = format_till_now(12, 21)
    if "Till now" not in text or "12/21" not in text:
        return _fail(f"format_till_now text mismatch: {text!r}")

    print("[PASS] window_presets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
