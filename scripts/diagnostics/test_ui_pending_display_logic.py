#!/usr/bin/env python3
"""T20: pure UI logic test for risk-profile pending/active display."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_profile_ui_utils import format_risk_profile_status


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    # requested == active => active wording
    same = format_risk_profile_status("mid", "mid")
    if same.get("pending") is not False:
        return _fail(f"expected pending=False for same profiles, got {same.get('pending')!r}")
    req_text_same = str(same.get("requested_text", "")).lower()
    if "active" not in req_text_same:
        return _fail(f"requested_text missing active marker: {same.get('requested_text')!r}")

    # requested != active => pending wording
    diff = format_risk_profile_status("mid", "high")
    if diff.get("pending") is not True:
        return _fail(f"expected pending=True for different profiles, got {diff.get('pending')!r}")
    req_text_diff = str(diff.get("requested_text", "")).lower()
    if "pending" not in req_text_diff:
        return _fail(f"requested_text missing pending marker: {diff.get('requested_text')!r}")

    # missing/invalid => active defaults to mid
    missing = format_risk_profile_status(None, "")
    if str(missing.get("active", "")).strip().lower() != "mid":
        return _fail(f"missing active did not default to mid: {missing!r}")
    if str(missing.get("requested", "")).strip().lower() != "mid":
        return _fail(f"missing requested did not default to active(mid): {missing!r}")

    invalid = format_risk_profile_status("not_valid", "bad")
    if str(invalid.get("active", "")).strip().lower() != "mid":
        return _fail(f"invalid active did not normalize to mid: {invalid!r}")
    if str(invalid.get("requested", "")).strip().lower() != "mid":
        return _fail(f"invalid requested did not normalize to active(mid): {invalid!r}")

    print("[PASS] ui_pending_display_logic")
    print(
        f"[INFO] same={same.get('requested_text')} | diff={diff.get('requested_text')} | "
        f"missing_active={missing.get('active')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

