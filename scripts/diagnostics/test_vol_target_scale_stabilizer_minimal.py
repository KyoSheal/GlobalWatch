#!/usr/bin/env python3
"""T61: minimal regression tests for vol-target scale stabilizer."""

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


def _run_sequence(raw_values, *, prev=1.0, counter=0):
    out = []
    smooth = float(prev)
    release_counter = int(counter)
    for raw in raw_values:
        step = stabilize_vol_target_scale(
            raw_scale=float(raw),
            prev_smooth_scale=float(smooth),
            release_counter=int(release_counter),
            deadzone_release_eps=0.01,
            deadzone_reduce_eps=0.00,
            ema_alpha_down=0.60,
            ema_alpha_up=0.20,
            release_confirm_cycles=3,
        )
        out.append(step)
        smooth = float(step.get("smooth_scale", smooth))
        release_counter = int(step.get("release_counter", release_counter))
    return out


def main() -> int:
    # Case 1: boundary jitter around 0.98~1.00 should not bounce to 1.0 every step.
    jitter = _run_sequence([0.995, 0.999, 0.994, 0.998, 0.996, 0.999], prev=1.0, counter=0)
    applied = [float(x["applied_scale"]) for x in jitter]
    if all(abs(x - 1.0) <= 1e-12 for x in applied):
        return _fail(f"jitter sequence should not apply 1.0 at every step: {applied}")

    # Case 2: sudden drop must follow down quickly (within 1-2 steps).
    drop = _run_sequence([0.85, 0.85], prev=1.0, counter=0)
    first = float(drop[0]["applied_scale"])
    second = float(drop[1]["applied_scale"])
    if first > 0.92:
        return _fail(f"drop follow is too slow on first step: {first}")
    if second > 0.88:
        return _fail(f"drop follow is too slow on second step: {second}")

    # Case 3: release back to 1.0 only after 3 consecutive raw>=0.99.
    release = _run_sequence([0.995, 0.996, 0.997], prev=0.90, counter=0)
    a0 = float(release[0]["applied_scale"])
    a1 = float(release[1]["applied_scale"])
    a2 = float(release[2]["applied_scale"])
    h2 = str(release[2].get("hold_reason", ""))
    if a0 >= 0.999999 or a1 >= 0.999999:
        return _fail(f"release should not confirm before 3rd cycle: {[a0, a1, a2]}")
    if abs(a2 - 1.0) > 1e-12:
        return _fail(f"release should confirm to 1.0 on 3rd cycle: {[a0, a1, a2]}")
    if h2 != "release_confirmed":
        return _fail(f"unexpected hold_reason on 3rd release cycle: {h2}")

    print("[PASS] vol_target_scale_stabilizer_minimal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

