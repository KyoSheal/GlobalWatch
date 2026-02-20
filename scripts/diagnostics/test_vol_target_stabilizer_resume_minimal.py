#!/usr/bin/env python3
"""T62: vol-target stabilizer resume persistence minimal regression test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vol_target_stabilizer import stabilize_vol_target_scale


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _step(raw: float, state: dict) -> dict:
    out = stabilize_vol_target_scale(
        raw_scale=float(raw),
        prev_smooth_scale=float(state.get("smooth_scale", 1.0)),
        release_counter=int(state.get("release_counter", 0)),
        deadzone_release_eps=0.01,
        deadzone_reduce_eps=0.00,
        ema_alpha_down=0.60,
        ema_alpha_up=0.20,
        release_confirm_cycles=3,
    )
    return {
        "smooth_scale": float(out.get("smooth_scale", 1.0)),
        "release_counter": int(out.get("release_counter", 0)),
        "last_applied_scale": float(out.get("applied_scale", 1.0)),
        "last_raw_scale": float(out.get("raw_scale", 1.0)),
        "last_hold_reason": str(out.get("hold_reason", "")),
    }


def main() -> int:
    # Phase 1: simulate first engine instance.
    s = {"smooth_scale": 1.0, "release_counter": 0}
    for raw in (1.0, 0.85, 0.88):
        s = _step(raw, s)

    if abs(float(s["smooth_scale"]) - 1.0) < 1e-9:
        return _fail(f"expected pre-resume smooth_scale != 1.0, got {s['smooth_scale']}")
    if not isinstance(s["release_counter"], int):
        return _fail("release_counter must be int before resume")
    if str(s.get("last_hold_reason", "")) == "uninitialized":
        return _fail("hold_reason must not be uninitialized before resume")

    # Simulate persisted snapshot["vol_target_state"] and restore in a new instance.
    dumped = {
        "smooth_scale": float(s["smooth_scale"]),
        "release_counter": int(s["release_counter"]),
        "last_applied_scale": float(s["last_applied_scale"]),
        "last_raw_scale": float(s["last_raw_scale"]),
        "last_update_utc": "2026-02-20T00:00:00+00:00",
    }

    # Phase 2: simulate resumed instance continuing from persisted state.
    resumed = dict(dumped)
    resumed_step1 = _step(0.90, resumed)
    if resumed_step1["last_applied_scale"] >= 0.999999:
        return _fail(
            f"resume continuity broken: first post-resume applied jumped to 1.0 ({resumed_step1['last_applied_scale']})"
        )
    if not isinstance(resumed_step1["release_counter"], int):
        return _fail("release_counter must stay int after resume")
    if str(resumed_step1.get("last_hold_reason", "")) == "uninitialized":
        return _fail("hold_reason must not be uninitialized after resume")

    resumed = resumed_step1
    for raw in (0.95, 0.99, 1.0):
        resumed = _step(raw, resumed)

    if not isinstance(resumed["release_counter"], int):
        return _fail("release_counter must remain int through post-resume sequence")

    print("[PASS] vol_target_stabilizer_resume_minimal")
    print(
        f"[INFO] smooth_pre={dumped['smooth_scale']:.6f} "
        f"smooth_post={resumed['smooth_scale']:.6f} "
        f"release_counter={resumed['release_counter']} "
        f"hold_reason={resumed.get('last_hold_reason','')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

