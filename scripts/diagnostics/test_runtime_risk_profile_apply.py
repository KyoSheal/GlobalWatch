#!/usr/bin/env python3
"""T10: runtime risk profile apply on next cycle + Step6 snapshot safety."""

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
    out_dir = ROOT / "outputs" / "test_runtime_apply"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) temp config
    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing base config: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("execution", {})["risk_profile"] = "mid"
    rep = cfg.setdefault("reporting", {})
    rep["out_dir"] = str(out_dir)
    rep["snapshot_live_path"] = str(out_dir / "snapshot_live.json")
    rep["runtime_control_path"] = str(out_dir / "runtime_control.json")
    rep["risk_profile_state_path"] = str(out_dir / "state" / "risk_profile_state.json")
    rep["telemetry_enabled"] = True
    rep["write_snapshot_on_profile_apply"] = True
    cfg_path = out_dir / "paper_config_runtime_apply.json"
    atomic_write_json(str(cfg_path), cfg, indent=2)

    # 2) fresh init
    old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(str(cfg_path))
    finally:
        if old_checkpoint_env is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

    # 3) initial snapshot (mid)
    base_cycle = 100
    engine.current_cycle = base_cycle
    snap_init = engine._build_profile_apply_snapshot(cycle_id=base_cycle, now_utc=datetime.now(timezone.utc))
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

    # 4) baseline metrics lines
    metrics_path = out_dir / "telemetry" / "metrics.jsonl"
    metrics_before = len(_read_jsonl(metrics_path))

    # 5) simulate UI request to high
    request_id = "T10-REQ-HIGH"
    try:
        written_path = engine.write_runtime_control_request("high", request_id=request_id)
    except Exception as exc:
        return _fail(f"write_runtime_control_request raised: {exc}")
    if Path(written_path).resolve() != (out_dir / "runtime_control.json").resolve():
        return _fail(f"runtime_control path mismatch: {written_path}")

    # 6) apply at next cycle boundary
    next_cycle = base_cycle + 1
    engine.current_cycle = next_cycle
    engine._maybe_apply_runtime_risk_profile(cycle_id=next_cycle, now_utc=datetime.now(timezone.utc))

    # Assertions: engine state
    if str(engine.active_risk_profile).strip().lower() != "high":
        return _fail(f"active_risk_profile={engine.active_risk_profile!r}, expected 'high'")
    if str(engine.requested_risk_profile).strip().lower() != "high":
        return _fail(f"requested_risk_profile={engine.requested_risk_profile!r}, expected 'high'")
    if int(engine.risk_profile_applied_cycle_id or -1) != int(next_cycle):
        return _fail(
            f"risk_profile_applied_cycle_id={engine.risk_profile_applied_cycle_id!r}, expected {next_cycle}"
        )
    if not str(engine.risk_profile_overrides_hash or "").strip():
        return _fail("risk_profile_overrides_hash is empty")

    # Assertions: snapshot live immediately updated by Step6
    snapshot = safe_read_json(str(out_dir / "snapshot_live.json"), retries=3, sleep_ms=20) or {}
    if not isinstance(snapshot, dict):
        return _fail("snapshot_live.json unreadable")
    if str(snapshot.get("source", "")).strip() != "risk_profile_apply":
        return _fail(f"snapshot source={snapshot.get('source')!r}, expected 'risk_profile_apply'")
    if str(snapshot.get("active_risk_profile", "")).strip().lower() != "high":
        return _fail(f"snapshot active_risk_profile={snapshot.get('active_risk_profile')!r}, expected 'high'")
    if str(snapshot.get("requested_risk_profile", "")).strip().lower() != "high":
        return _fail(
            f"snapshot requested_risk_profile={snapshot.get('requested_risk_profile')!r}, expected 'high'"
        )

    # Assertions: events
    events_path = out_dir / "telemetry" / "events.jsonl"
    events = _read_jsonl(events_path)
    event_names = [str(e.get("event", "")) for e in events]
    if "RISK_PROFILE_APPLIED" not in event_names:
        return _fail("events missing RISK_PROFILE_APPLIED")
    if "RISK_PROFILE_APPLY_SNAPSHOT_WRITE" not in event_names:
        return _fail("events missing RISK_PROFILE_APPLY_SNAPSHOT_WRITE")

    # Assertions: Step6 should not add CYCLE_METRICS
    metrics_after = len(_read_jsonl(metrics_path))
    if metrics_after != metrics_before:
        return _fail(
            f"metrics.jsonl changed by profile apply: before={metrics_before}, after={metrics_after}"
        )

    print("[PASS] runtime_risk_profile_apply")
    print(
        f"[INFO] cycle={next_cycle} active={engine.active_risk_profile} requested={engine.requested_risk_profile} "
        f"hash={engine.risk_profile_overrides_hash}"
    )
    print(
        f"[INFO] events_found=RISK_PROFILE_APPLIED,RISK_PROFILE_APPLY_SNAPSHOT_WRITE "
        f"metrics_lines={metrics_after}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
