from __future__ import annotations

import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_trading import resolve_portfolio_cov_rc_abort_buffer_decision


def _check(name: str, condition: bool) -> bool:
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] {name}")
    return bool(condition)


def _run(rc: float, enabled: bool, streak: int, remaining: int, trigger: int = 3, delta: float = 0.02, active: int = 3):
    return resolve_portfolio_cov_rc_abort_buffer_decision(
        portfolio_rc_fraction=rc,
        previous_gate_decision=None,
        base_rc_limit=0.30,
        hysteresis_band=0.03,
        buffer_enabled=enabled,
        trigger_consecutive_aborts=trigger,
        relax_delta=delta,
        active_cycles=active,
        prev_abort_streak=streak,
        prev_buffer_cycles_remaining=remaining,
        allow_buffer=True,
    )


def main() -> int:
    ok = True

    d_disabled = _run(rc=0.31, enabled=False, streak=5, remaining=2)
    ok &= _check("disabled mode does not trigger buffer", not d_disabled["abort_buffer_active"] and not d_disabled["abort_buffer_triggered_this_cycle"])

    d_trigger = _run(rc=0.31, enabled=True, streak=2, remaining=0)
    ok &= _check("third abort triggers buffer", d_trigger["abort_buffer_triggered_this_cycle"] and d_trigger["abort_buffer_active"])
    ok &= _check("trigger cycle degraded allow true", d_trigger["degraded_allow"] and d_trigger["final_gate_decision"] == "ALLOW")

    d_active = _run(rc=0.31, enabled=True, streak=0, remaining=int(d_trigger["active_cycles_remaining"]))
    ok &= _check("active buffer cycle uses relaxed decision", d_active["abort_buffer_active"] and d_active["final_gate_decision"] == "ALLOW")

    d_restore = _run(rc=0.31, enabled=True, streak=0, remaining=0)
    ok &= _check("after buffer expires returns to base behavior", (not d_restore["abort_buffer_active"]) and d_restore["final_gate_decision"] == "ABORT")

    print("[RC_ABORT_BUFFER_CASE] " + json.dumps(d_trigger, ensure_ascii=False))
    print("[RC_ABORT_BUFFER_ACTIVE] " + json.dumps(d_active, ensure_ascii=False))
    print("[RC_ABORT_BUFFER_RESTORE] " + json.dumps(d_restore, ensure_ascii=False))
    if ok:
        print("VERIFY_SUMMARY pass=5 fail=0")
        return 0
    print("VERIFY_SUMMARY pass<5 fail>0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
