#!/usr/bin/env python3
"""T11: runtime risk profile reject for invalid requests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, safe_read_json
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
    out_dir = ROOT / "outputs" / "test_runtime_reject"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing base config: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("execution", {})["risk_profile"] = "mid"
    rep = cfg.setdefault("reporting", {})
    rep["out_dir"] = str(out_dir)
    rep["snapshot_live_path"] = str(out_dir / "snapshot_live.json")
    rep["runtime_control_path"] = str(out_dir / "runtime_control.json")
    rep["telemetry_enabled"] = True
    rep["write_snapshot_on_profile_apply"] = True
    cfg_path = out_dir / "paper_config_runtime_reject.json"
    atomic_write_json(str(cfg_path), cfg, indent=2)

    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(str(cfg_path))
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    # baseline snapshot
    cycle_id = 200
    engine.current_cycle = cycle_id
    snap_init = engine._build_profile_apply_snapshot(cycle_id=cycle_id, now_utc=datetime.now(timezone.utc))
    ok = bool(
        engine.write_live_snapshot(
            snap_init,
            source="init_profile_state",
            emit_telemetry=False,
            emit_cycle_metrics=False,
            lightweight=True,
        )
    )
    if not ok:
        return _fail("initial snapshot write failed")

    before_snapshot = safe_read_json(str(out_dir / "snapshot_live.json"), retries=3, sleep_ms=20) or {}
    before_active = str(before_snapshot.get("active_risk_profile", "")).strip().lower()
    before_source = str(before_snapshot.get("source", "")).strip()

    # invalid request
    invalid_requested = "manual"
    request_id = "T11-REQ-INVALID"
    runtime_payload = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "requested_risk_profile": invalid_requested,
    }
    atomic_write_json(str(out_dir / "runtime_control.json"), runtime_payload, indent=2)

    # next cycle apply attempt
    next_cycle = cycle_id + 1
    engine.current_cycle = next_cycle
    engine._maybe_apply_runtime_risk_profile(cycle_id=next_cycle, now_utc=datetime.now(timezone.utc))

    # assert active unchanged
    if str(engine.active_risk_profile).strip().lower() != "mid":
        return _fail(f"engine.active_risk_profile changed to {engine.active_risk_profile!r}, expected 'mid'")

    # snapshot should not become risk_profile_apply on rejected request
    after_snapshot = safe_read_json(str(out_dir / "snapshot_live.json"), retries=3, sleep_ms=20) or {}
    after_active = str(after_snapshot.get("active_risk_profile", "")).strip().lower()
    after_source = str(after_snapshot.get("source", "")).strip()
    if after_active != "mid":
        return _fail(f"snapshot active_risk_profile changed to {after_active!r}, expected 'mid'")
    if before_source != "risk_profile_apply" and after_source == "risk_profile_apply":
        return _fail("snapshot source switched to risk_profile_apply on rejected request")

    # events should include reject with invalid_profile reason
    events_path = out_dir / "telemetry" / "events.jsonl"
    events = _read_jsonl(events_path)
    rejected = []
    for row in events:
        if str(row.get("event", "")) != "RISK_PROFILE_APPLY_REJECTED":
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("reason", "")).strip() == "invalid_profile":
            rejected.append(row)
    if not rejected:
        return _fail("missing RISK_PROFILE_APPLY_REJECTED with reason=invalid_profile")

    print("[PASS] runtime_risk_profile_reject")
    print(
        f"[INFO] active_before={before_active} active_after={after_active} "
        f"source_before={before_source!r} source_after={after_source!r}"
    )
    print("[INFO] reject_event_reason=invalid_profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

