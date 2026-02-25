import json
import os
import tempfile

from risk_profile_state import (
    RiskProfileStateManager,
    request_risk_profile_change,
    resolve_risk_profile_events_path,
)


def _read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = str(line or "").strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def test_risk_profile_audit_append_only_and_noop_policy():
    with tempfile.TemporaryDirectory() as td:
        state_path = os.path.join(td, "outputs", "state", "risk_profile_state.json")
        events_path = resolve_risk_profile_events_path(state_path=state_path)

        manager = RiskProfileStateManager(state_path, default_requested="mid")
        state0 = manager.load(ensure=True)
        assert str(state0.get("requested", "")) == "mid"

        changed1 = request_risk_profile_change(
            state_path,
            requested="high",
            source="ui",
            actor="pytest",
            run_id="run_x",
            cycle_id=123,
        )
        assert bool(changed1.get("changed", False)) is True

        rows1 = _read_jsonl(events_path)
        assert len(rows1) == 1
        e1 = rows1[-1]
        assert str(e1.get("old", "")) == "mid"
        assert str(e1.get("new", "")) == "high"
        assert str(e1.get("ts", "")) != ""
        assert str(e1.get("state_version", "")) != ""
        assert str(e1.get("source", "")) == "ui"

        changed2 = request_risk_profile_change(
            state_path,
            requested="high",
            source="ui",
            actor="pytest",
            run_id="run_x",
            cycle_id=124,
        )
        assert bool(changed2.get("changed", True)) is False
        rows2 = _read_jsonl(events_path)
        assert len(rows2) == 1

