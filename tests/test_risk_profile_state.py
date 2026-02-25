import json
import os
import tempfile
import time

from risk_profile_state import RiskProfileStateManager, write_risk_profile_state


def test_risk_profile_state_atomic_write_and_parse():
    with tempfile.TemporaryDirectory() as td:
        state_path = os.path.join(td, "outputs", "state", "risk_profile_state.json")
        payload = write_risk_profile_state(state_path, requested="high", set_by="ui")

        assert os.path.exists(state_path)
        with open(state_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        assert isinstance(obj, dict)
        assert obj.get("requested") == "high"
        assert obj.get("version")
        assert payload.get("requested") == "high"


def test_risk_profile_manager_reload_if_changed_and_restart():
    with tempfile.TemporaryDirectory() as td:
        state_path = os.path.join(td, "outputs", "state", "risk_profile_state.json")
        manager = RiskProfileStateManager(state_path, default_requested="mid")
        initial = manager.load(ensure=True)

        assert initial.get("requested") == "mid"
        before_version = str(initial.get("version") or "")

        # Keep this robust on Windows mtime precision.
        time.sleep(1.05)
        write_risk_profile_state(state_path, requested="high", set_by="ui")

        changed = manager.reload_if_changed()
        assert changed is True
        assert manager.get_requested() == "high"
        assert str(manager.state.get("version") or "") != before_version

        # Restart-equivalent: new manager should still load the persisted requested profile.
        manager2 = RiskProfileStateManager(state_path, default_requested="mid")
        state2 = manager2.load(ensure=True)
        assert state2.get("requested") == "high"

