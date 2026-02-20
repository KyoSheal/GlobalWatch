"""Vol-target scale stabilizer with hysteresis/deadzone and asymmetric smoothing."""

from __future__ import annotations

from typing import Dict


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def stabilize_vol_target_scale(
    raw_scale: float,
    prev_smooth_scale: float,
    release_counter: int,
    *,
    deadzone_release_eps: float = 0.01,
    deadzone_reduce_eps: float = 0.0,
    ema_alpha_down: float = 0.60,
    ema_alpha_up: float = 0.20,
    release_confirm_cycles: int = 3,
) -> Dict[str, float | int | str]:
    """Return stabilized scale and updated state.

    Behavior:
    - Faster response on risk increase (raw scale down): `ema_alpha_down`
    - Slower release on risk decrease (raw scale up): `ema_alpha_up`
    - Release to 1.0 only after `release_confirm_cycles` consecutive
      cycles with `raw_scale >= 1 - deadzone_release_eps`.
    """

    raw = _clip01(float(raw_scale))
    prev = _clip01(float(prev_smooth_scale))
    counter = max(0, int(release_counter))

    alpha_down = _clip01(float(ema_alpha_down))
    alpha_up = _clip01(float(ema_alpha_up))
    release_eps = max(0.0, float(deadzone_release_eps))
    reduce_eps = max(0.0, float(deadzone_reduce_eps))
    confirm_n = max(1, int(release_confirm_cycles))
    release_threshold = 1.0 - release_eps

    scale_diff = raw - prev
    if raw < (prev - reduce_eps):
        alpha = alpha_down
        phase = "down_fast"
    elif raw > prev:
        alpha = alpha_up
        phase = "up_slow"
    else:
        alpha = alpha_up
        phase = "steady"

    smooth = _clip01(prev + alpha * scale_diff)
    applied = smooth
    hold_reason = phase
    next_counter = counter

    if raw >= release_threshold:
        next_counter = counter + 1
        hold_reason = "release_confirming"
        if next_counter >= confirm_n:
            applied = 1.0
            smooth = 1.0
            next_counter = 0
            hold_reason = "release_confirmed"
    else:
        next_counter = 0

    return {
        "raw_scale": raw,
        "smooth_scale": _clip01(smooth),
        "applied_scale": _clip01(applied),
        "release_counter": int(next_counter),
        "hold_reason": str(hold_reason),
    }

