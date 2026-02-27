from __future__ import annotations

import uuid

from risk_profile_state import (
    RiskProfileStateManager,
    request_risk_profile_change,
    write_risk_profile_state,
)


def test_mid_to_high_reload_if_changed(tmp_path):
    state_path = tmp_path / "outputs" / "state" / "risk_profile_state.json"
    mgr = RiskProfileStateManager(str(state_path), default_requested="mid")
    initial = mgr.load(ensure=True)
    assert str(initial.get("requested", "")) == "mid"

    change = request_risk_profile_change(
        str(state_path),
        requested="high",
        source="pytest",
        actor="test_mid_to_high_reload_if_changed",
        run_id="pytest",
        cycle_id=1,
    )
    assert bool(change.get("changed", False))

    changed = mgr.reload_if_changed(force=False)
    assert changed is True
    assert mgr.get_requested() == "high"


def test_reload_if_changed_when_mtime_same_but_version_changes(tmp_path, monkeypatch):
    state_path = tmp_path / "outputs" / "state" / "risk_profile_state.json"
    mgr = RiskProfileStateManager(str(state_path), default_requested="mid")
    mgr.load(ensure=True)

    prev_mtime = mgr.last_mtime
    prev_version = str(mgr.last_version or "")
    new_version = f"pytest-{uuid.uuid4().hex[:12]}"
    assert new_version != prev_version

    write_risk_profile_state(
        str(state_path),
        requested="high",
        set_by="pytest",
        version=new_version,
    )

    monkeypatch.setattr(mgr, "_get_mtime", lambda: prev_mtime)
    changed = mgr.reload_if_changed(force=False)

    assert changed is True
    assert mgr.get_requested() == "high"
    assert str(mgr.last_version) == new_version
    assert bool((mgr.last_reload_diag or {}).get("mtime_changed", True)) is False
    assert bool((mgr.last_reload_diag or {}).get("version_changed", False)) is True
