"""Quick verification for risk profile audit jsonl append behavior."""

from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from risk_profile_state import request_risk_profile_change, resolve_risk_profile_events_path


def _read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gw_risk_profile_audit_") as td:
        state_path = os.path.join(td, "outputs", "state", "risk_profile_state.json")
        events_path = resolve_risk_profile_events_path(state_path=state_path)

        c1 = request_risk_profile_change(
            state_path,
            requested="mid",
            source="verify_init",
            actor="verify",
            run_id="verify_run",
            cycle_id=1,
        )
        # init may be no-op (default mid); both outcomes acceptable.
        _ = c1

        c2 = request_risk_profile_change(
            state_path,
            requested="high",
            source="verify_script",
            actor="verify",
            run_id="verify_run",
            cycle_id=2,
        )
        c3 = request_risk_profile_change(
            state_path,
            requested="mid",
            source="verify_script",
            actor="verify",
            run_id="verify_run",
            cycle_id=3,
        )
        if not bool(c2.get("changed", False)) or not bool(c3.get("changed", False)):
            print("FAIL: expected two real changes")
            return 1

        rows = _read_jsonl(events_path)
        if len(rows) < 2:
            print("FAIL: expected at least 2 audit rows")
            return 1

        tail = rows[-2:]
        print("Last two events:")
        for row in tail:
            print(json.dumps(row, ensure_ascii=False))

        if str(tail[0].get("old")) != "mid" or str(tail[0].get("new")) != "high":
            print("FAIL: first change is not mid->high")
            return 1
        if str(tail[1].get("old")) != "high" or str(tail[1].get("new")) != "mid":
            print("FAIL: second change is not high->mid")
            return 1

    print("PASS: risk_profile audit verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

