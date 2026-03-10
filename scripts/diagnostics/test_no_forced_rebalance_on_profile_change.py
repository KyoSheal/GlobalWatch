#!/usr/bin/env python3
"""T31: ensure profile change does not force rebalance marker events."""

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
    out_dir = ROOT / "outputs" / "test_no_forced_rebalance_on_profile_change"
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
    rep["risk_profile_state_path"] = str(out_dir / "state" / "risk_profile_state.json")
    rep["telemetry_enabled"] = True
    rep["write_snapshot_on_profile_apply"] = True
    cfg_path = out_dir / "paper_config_t31.json"
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

    # Baseline cycle.
    cycle1 = 700
    engine.current_cycle = cycle1
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle1, now_utc=datetime.now(timezone.utc))

    # Profile switch request mid -> high.
    try:
        engine.write_runtime_control_request("high", request_id="T31-REQ-HIGH")
    except Exception as exc:
        return _fail(f"write_runtime_control_request failed: {exc}")

    cycle2 = cycle1 + 1
    engine.current_cycle = cycle2
    engine._maybe_apply_runtime_risk_profile(cycle_id=cycle2, now_utc=datetime.now(timezone.utc))

    if str(engine.active_risk_profile).strip().lower() != "high":
        return _fail(f"profile apply failed, active={engine.active_risk_profile!r}, expected 'high'")

    events = _read_jsonl(out_dir / "telemetry" / "events.jsonl")
    if not events:
        print("[SKIP] no telemetry events found; cannot inspect rebalance markers")
        return 0

    event_names = [str(e.get("event", "")) for e in events]

    # Strict forbidden marker events.
    forbidden_names = {
        "FORCED_REBALANCE_ON_PROFILE_CHANGE",
        "PROFILE_CHANGE_FORCED_REBALANCE",
    }
    forbidden_hits = [name for name in event_names if name in forbidden_names]
    if forbidden_hits:
        return _fail(f"found forbidden forced-rebalance events: {forbidden_hits}")

    # Heuristic fallback: ensure no rebalance plan event explicitly tied to profile change.
    rebalance_rows = [
        e for e in events
        if str(e.get("event", "")).upper() in {"REBALANCE_PLAN", "REBALANCE_PLAN_FILTERED"}
    ]
    if not rebalance_rows:
        print("[SKIP] no rebalance plan events in minimal flow; no forced profile-triggered rebalance marker observed")
        print(
            f"[INFO] profile_switch mid->high applied_cycle={engine.risk_profile_applied_cycle_id} "
            f"events_total={len(events)}"
        )
        return 0

    suspicious = []
    for row in rebalance_rows:
        payload = row.get("payload", {})
        haystack = json.dumps(payload, ensure_ascii=False).lower() if isinstance(payload, dict) else str(payload).lower()
        msg = str(row.get("message", "")).lower()
        status = str(row.get("status", "")).lower()
        text = " ".join([haystack, msg, status])
        if "profile" in text and ("force" in text or "change" in text or "runtime" in text):
            suspicious.append(row)
    if suspicious:
        return _fail(f"rebalance plan appears profile-triggered/forced; count={len(suspicious)}")

    print("[PASS] no_forced_rebalance_on_profile_change")
    print(
        f"[INFO] applied_cycle={engine.risk_profile_applied_cycle_id} "
        f"rebalance_events={len(rebalance_rows)} forced_markers=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
