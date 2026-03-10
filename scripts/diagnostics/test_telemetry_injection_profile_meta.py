#!/usr/bin/env python3
"""T13: verify telemetry wrapper auto-injects risk profile meta into payload."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json
from paper_trading import PaperTradingEngine


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    out_dir = ROOT / "outputs" / "test_telemetry_injection_profile_meta"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build temp config so telemetry writes are isolated.
    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing base config: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("execution", {})["risk_profile"] = "mid"
    rep = cfg.setdefault("reporting", {})
    rep["out_dir"] = str(out_dir)
    rep["telemetry_enabled"] = True
    rep["snapshot_live_path"] = str(out_dir / "snapshot_live.json")
    rep["runtime_control_path"] = str(out_dir / "runtime_control.json")
    rep["risk_profile_state_path"] = str(out_dir / "state" / "risk_profile_state.json")
    cfg_path = out_dir / "paper_config_t13.json"
    atomic_write_json(str(cfg_path), cfg, indent=2)

    # Fresh start.
    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(str(cfg_path))
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    # Unit event call through wrapper under test.
    engine._telemetry_log_event("UNIT_TEST_EVENT", cycle_id=1, payload={"foo": 1})

    events_path = out_dir / "telemetry" / "events.jsonl"
    rows = _read_jsonl(events_path)
    if not rows:
        return _fail(f"no events found at {events_path}")

    last = rows[-1]
    if str(last.get("event", "")) != "UNIT_TEST_EVENT":
        return _fail(f"last event is {last.get('event')!r}, expected 'UNIT_TEST_EVENT'")

    payload = last.get("payload", {})
    if not isinstance(payload, dict):
        return _fail("last payload is not a dict")

    required = [
        "active_risk_profile",
        "requested_risk_profile",
        "risk_profile_overrides_hash",
        "risk_profile_template_version",
        "risk_profile_applied_cycle_id",
        "risk_profile_applied_at_utc",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        return _fail(f"payload missing injected fields: {missing}")

    active = str(payload.get("active_risk_profile", "") or "").strip().lower()
    if active not in {"low", "mid", "high", "ultra"}:
        return _fail(f"active_risk_profile invalid: {active!r}")

    print("[PASS] telemetry_injection_profile_meta")
    print(
        f"[INFO] active={payload.get('active_risk_profile')} "
        f"requested={payload.get('requested_risk_profile')} "
        f"hash={payload.get('risk_profile_overrides_hash')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
