#!/usr/bin/env python3
"""T23: optional test for 'Use Active Profile' helper behavior."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_profile_ui_utils import set_filter_to_active


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    session_state = {"diag_risk_profile_filter": "All"}
    snapshot = {"active_risk_profile": "high"}

    selected = set_filter_to_active(session_state, snapshot)

    if selected != "high":
        return _fail(f"returned selected profile {selected!r}, expected 'high'")
    if session_state.get("diag_risk_profile_filter") != "high":
        return _fail(
            "session_state['diag_risk_profile_filter'] "
            f"is {session_state.get('diag_risk_profile_filter')!r}, expected 'high'"
        )

    print("[PASS] ui_use_active_profile_button")
    print(
        f"[INFO] selected={selected} "
        f"session_filter={session_state.get('diag_risk_profile_filter')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

