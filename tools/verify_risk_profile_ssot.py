"""Quick SSOT verification for risk profile state persistence."""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk_profile_state import RiskProfileStateManager, write_risk_profile_state


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gw_risk_profile_ssot_") as td:
        state_path = os.path.join(td, "outputs", "state", "risk_profile_state.json")

        # a) initialize state=mid
        mgr = RiskProfileStateManager(state_path, default_requested="mid")
        s0 = mgr.load(ensure=True)
        if str(s0.get("requested", "")) != "mid":
            return _fail("init requested is not mid")

        # b) write state=high
        s1 = write_risk_profile_state(state_path, requested="high", set_by="verify_script")
        if str(s1.get("requested", "")) != "high":
            return _fail("write requested=high failed")

        # c) manager reload reads requested/applied (applied simulated here as requested)
        changed = mgr.reload_if_changed(force=True)
        if not changed:
            return _fail("reload_if_changed did not detect update")
        requested = mgr.get_requested()
        applied = requested
        print(f"[RISK_PROFILE] requested={requested} applied={applied} source=state_file path={state_path}")
        if requested != "high":
            return _fail("manager requested is not high after reload")

        # d) new manager restart-equivalent still reads high
        mgr2 = RiskProfileStateManager(state_path, default_requested="mid")
        s2 = mgr2.load(ensure=True)
        if str(s2.get("requested", "")) != "high":
            return _fail("restart-equivalent load did not persist high")

    print("PASS: risk_profile SSOT verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
