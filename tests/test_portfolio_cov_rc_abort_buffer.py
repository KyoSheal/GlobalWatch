from __future__ import annotations

import paper_trading


def _run_case(
    *,
    rc: float,
    enabled: bool,
    streak: int,
    remaining: int,
    trigger: int = 3,
    delta: float = 0.02,
    active_cycles: int = 3,
):
    return paper_trading.resolve_portfolio_cov_rc_abort_buffer_decision(
        portfolio_rc_fraction=rc,
        previous_gate_decision=None,
        base_rc_limit=0.30,
        hysteresis_band=0.03,
        buffer_enabled=enabled,
        trigger_consecutive_aborts=trigger,
        relax_delta=delta,
        active_cycles=active_cycles,
        prev_abort_streak=streak,
        prev_buffer_cycles_remaining=remaining,
        allow_buffer=True,
    )


def test_abort_buffer_disabled_is_backward_compatible():
    d1 = _run_case(rc=0.31, enabled=False, streak=0, remaining=0)
    d2 = _run_case(rc=0.31, enabled=False, streak=1, remaining=0)
    assert d1["abort_buffer_active"] is False
    assert d1["abort_buffer_triggered_this_cycle"] is False
    assert d1["final_gate_decision"] == "ABORT"
    assert d2["final_gate_decision"] == "ABORT"


def test_abort_buffer_not_triggered_before_threshold():
    d1 = _run_case(rc=0.31, enabled=True, streak=0, remaining=0, trigger=3)
    d2 = _run_case(rc=0.31, enabled=True, streak=1, remaining=0, trigger=3)
    assert d1["abort_streak"] == 1
    assert d2["abort_streak"] == 2
    assert d1["abort_buffer_triggered_this_cycle"] is False
    assert d2["abort_buffer_triggered_this_cycle"] is False
    assert d2["final_gate_decision"] == "ABORT"


def test_abort_buffer_triggers_and_allows_on_threshold_cycle():
    d3 = _run_case(rc=0.31, enabled=True, streak=2, remaining=0, trigger=3, delta=0.02, active_cycles=3)
    assert d3["abort_buffer_triggered_this_cycle"] is True
    assert d3["abort_buffer_active"] is True
    assert d3["degraded_allow"] is True
    assert d3["final_gate_decision"] == "ALLOW"
    assert abs(float(d3["effective_rc_limit"]) - 0.32) < 1e-12
    assert d3["active_cycles_remaining"] == 2
    assert d3["abort_streak"] == 0


def test_abort_buffer_active_cycle_uses_relaxed_limit():
    d_active = _run_case(rc=0.31, enabled=True, streak=0, remaining=2, trigger=3, delta=0.02, active_cycles=3)
    assert d_active["abort_buffer_active"] is True
    assert d_active["abort_buffer_triggered_this_cycle"] is False
    assert d_active["final_gate_decision"] == "ALLOW"
    assert d_active["degraded_allow"] is True
    assert d_active["active_cycles_remaining"] == 1


def test_abort_buffer_relaxed_still_abort_no_recursive_relax():
    d = _run_case(rc=0.40, enabled=True, streak=2, remaining=0, trigger=3, delta=0.02, active_cycles=3)
    assert d["abort_buffer_triggered_this_cycle"] is True
    assert d["abort_buffer_active"] is True
    assert d["final_gate_decision"] == "ABORT"
    assert d["degraded_allow"] is False
    assert abs(float(d["effective_rc_limit"]) - 0.32) < 1e-12


def test_abort_buffer_expires_and_returns_to_base_logic():
    d_last_active = _run_case(rc=0.31, enabled=True, streak=0, remaining=1, trigger=3, delta=0.02, active_cycles=3)
    assert d_last_active["final_gate_decision"] == "ALLOW"
    assert d_last_active["active_cycles_remaining"] == 0

    d_after = _run_case(
        rc=0.31,
        enabled=True,
        streak=int(d_last_active.get("abort_streak", 0)),
        remaining=int(d_last_active.get("active_cycles_remaining", 0)),
        trigger=3,
        delta=0.02,
        active_cycles=3,
    )
    assert d_after["abort_buffer_active"] is False
    assert d_after["final_gate_decision"] == "ABORT"
