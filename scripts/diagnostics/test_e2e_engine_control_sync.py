#!/usr/bin/env python3
"""T30: e2e runtime control -> next-cycle profile apply sync."""

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
    out_dir = ROOT / "outputs" / "test_e2e_engine_control_sync"
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
    cfg_path = out_dir / "paper_config_t30.json"
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

    if str(engine.active_risk_profile).strip().lower() != "mid":
        return _fail(f"initial active profile expected mid, got {engine.active_risk_profile!r}")

    # Cycle 1 (no control request): remain mid.
    cycle1 = 600
    engine.current_cycle = cycle1
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle1, now_utc=datetime.now(timezone.utc))
    snap1 = engine._build_profile_apply_snapshot(cycle_id=cycle1, now_utc=datetime.now(timezone.utc))
    if not engine.write_live_snapshot(
        snap1,
        source="cycle_1",
        emit_telemetry=False,
        emit_cycle_metrics=False,
        lightweight=True,
    ):
        return _fail("cycle1 snapshot write failed")
    if str(engine.active_risk_profile).strip().lower() != "mid":
        return _fail(f"cycle1 active expected mid, got {engine.active_risk_profile!r}")

    # UI-equivalent request: high.
    try:
        engine.write_runtime_control_request("high", request_id="T30-REQ-HIGH")
    except Exception as exc:
        return _fail(f"write_runtime_control_request failed: {exc}")

    # Cycle 2 start: apply requested profile.
    cycle2 = cycle1 + 1
    engine.current_cycle = cycle2
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle2, now_utc=datetime.now(timezone.utc))

    if str(engine.active_risk_profile).strip().lower() != "high":
        return _fail(f"cycle2 active expected high, got {engine.active_risk_profile!r}")
    if str(engine.requested_risk_profile).strip().lower() != "high":
        return _fail(f"cycle2 requested expected high, got {engine.requested_risk_profile!r}")

    # Step6 immediate snapshot should reflect apply source + active high.
    live_path = out_dir / "snapshot_live.json"
    snap_after_apply = safe_read_json(str(live_path), retries=3, sleep_ms=20) or {}
    if str(snap_after_apply.get("source", "")).strip() != "risk_profile_apply":
        return _fail(f"expected apply snapshot source='risk_profile_apply', got {snap_after_apply.get('source')!r}")
    if str(snap_after_apply.get("active_risk_profile", "")).strip().lower() != "high":
        return _fail(
            f"expected apply snapshot active_risk_profile='high', got {snap_after_apply.get('active_risk_profile')!r}"
        )

    # Telemetry should include apply events.
    events = _read_jsonl(out_dir / "telemetry" / "events.jsonl")
    names = [str(r.get("event", "")) for r in events]
    if "RISK_PROFILE_APPLIED" not in names:
        return _fail("events missing RISK_PROFILE_APPLIED")
    if "RISK_PROFILE_APPLY_SNAPSHOT_WRITE" not in names:
        return _fail("events missing RISK_PROFILE_APPLY_SNAPSHOT_WRITE")

    print("[PASS] e2e_engine_control_sync")
    print(
        f"[INFO] cycle1_active=mid cycle2_active={engine.active_risk_profile} "
        f"snapshot_source={snap_after_apply.get('source')} "
        f"applied_cycle={engine.risk_profile_applied_cycle_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

