#!/usr/bin/env python3
"""T14: write_live_snapshot signature regression and lightweight behavior."""

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


def _minimal_payload(engine: PaperTradingEngine, cycle_id: int, ts_iso: str) -> dict:
    active = str(getattr(engine, "active_risk_profile", "") or "mid").strip().lower()
    if active not in {"low", "mid", "high", "ultra"}:
        active = "mid"
    requested = str(getattr(engine, "requested_risk_profile", "") or active).strip().lower()
    if requested not in {"low", "mid", "high", "ultra"}:
        requested = active

    cash = float(getattr(engine, "cash", 0.0) or 0.0)
    return {
        "timestamp": ts_iso,
        "cycle": int(cycle_id),
        "cycle_id": int(cycle_id),
        "status": str(getattr(engine, "status", "RUNNING") or "RUNNING"),
        "cash": cash,
        "positions_value": 0.0,
        "total_equity": cash,
        "total_return": 0.0,
        "drawdown": 0.0,
        "positions": {},
        "positions_detail": {},
        "active_risk_profile": active,
        "requested_risk_profile": requested,
        "risk_profile_template_version": int(getattr(engine, "risk_profile_template_version", 1) or 1),
        "risk_profile_overrides_hash": str(getattr(engine, "risk_profile_overrides_hash", "") or "test-hash"),
        "risk_profile_applied_cycle_id": int(getattr(engine, "risk_profile_applied_cycle_id", cycle_id) or cycle_id),
        "risk_profile_applied_at_utc": str(getattr(engine, "risk_profile_applied_at_utc", ts_iso) or ts_iso),
    }


def main() -> int:
    out_dir = ROOT / "outputs" / "test_write_live_snapshot_signature"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing base config: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    rep = cfg.setdefault("reporting", {})
    rep["out_dir"] = str(out_dir)
    rep["snapshot_live_path"] = str(out_dir / "snapshot_live.json")
    rep["runtime_control_path"] = str(out_dir / "runtime_control.json")
    rep["risk_profile_state_path"] = str(out_dir / "state" / "risk_profile_state.json")
    rep["telemetry_enabled"] = True
    cfg_path = out_dir / "paper_config_t14.json"
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

    cycle_id = 300
    engine.current_cycle = cycle_id
    ts_iso = datetime.now(timezone.utc).isoformat()
    snapshot = _minimal_payload(engine, cycle_id=cycle_id, ts_iso=ts_iso)

    # 1) Legacy signature call still works.
    orig_build_live_snapshot = engine.build_live_snapshot
    legacy_flags = {"heavy_called": False}

    def _fake_build_live_snapshot(snap):
        legacy_flags["heavy_called"] = True
        return _minimal_payload(engine, cycle_id=cycle_id, ts_iso=ts_iso)

    engine.build_live_snapshot = _fake_build_live_snapshot  # type: ignore[assignment]
    try:
        ok_legacy = engine.write_live_snapshot(snapshot, source="legacy_signature")
    except Exception as exc:
        engine.build_live_snapshot = orig_build_live_snapshot  # type: ignore[assignment]
        return _fail(f"legacy write_live_snapshot call raised: {exc}")
    finally:
        engine.build_live_snapshot = orig_build_live_snapshot  # type: ignore[assignment]

    if not isinstance(ok_legacy, bool):
        return _fail(f"legacy call did not return bool: {type(ok_legacy)}")
    if not ok_legacy:
        return _fail("legacy write_live_snapshot returned False")
    if not legacy_flags["heavy_called"]:
        return _fail("legacy call did not use heavy path (build_live_snapshot was not called)")

    # 2) lightweight=True should skip heavy path and emit_cycle_metrics=False should not append metrics.
    metrics_path = out_dir / "telemetry" / "metrics.jsonl"
    metrics_before = len(_read_jsonl(metrics_path))

    orig_build_live_snapshot = engine.build_live_snapshot
    orig_build_light_payload = engine._build_live_snapshot_payload
    lw_flags = {"heavy_called": False, "light_called": False}

    def _fake_heavy_should_not_run(_snap):
        lw_flags["heavy_called"] = True
        raise RuntimeError("heavy path invoked unexpectedly")

    def _fake_light_payload(_snap):
        lw_flags["light_called"] = True
        return _minimal_payload(engine, cycle_id=cycle_id, ts_iso=ts_iso)

    engine.build_live_snapshot = _fake_heavy_should_not_run  # type: ignore[assignment]
    engine._build_live_snapshot_payload = _fake_light_payload  # type: ignore[assignment]
    try:
        ok_light = engine.write_live_snapshot(
            snapshot,
            source="lightweight_signature",
            emit_telemetry=True,
            emit_cycle_metrics=False,
            lightweight=True,
        )
    except Exception as exc:
        engine.build_live_snapshot = orig_build_live_snapshot  # type: ignore[assignment]
        engine._build_live_snapshot_payload = orig_build_light_payload  # type: ignore[assignment]
        return _fail(f"lightweight write_live_snapshot raised: {exc}")
    finally:
        engine.build_live_snapshot = orig_build_live_snapshot  # type: ignore[assignment]
        engine._build_live_snapshot_payload = orig_build_light_payload  # type: ignore[assignment]

    if not isinstance(ok_light, bool):
        return _fail(f"lightweight call did not return bool: {type(ok_light)}")
    if not ok_light:
        return _fail("lightweight write_live_snapshot returned False")
    if lw_flags["heavy_called"]:
        return _fail("heavy path was invoked under lightweight=True")
    if not lw_flags["light_called"]:
        return _fail("lightweight payload builder was not called under lightweight=True")

    metrics_after = len(_read_jsonl(metrics_path))
    if metrics_after != metrics_before:
        return _fail(
            f"metrics.jsonl changed with emit_cycle_metrics=False: before={metrics_before}, after={metrics_after}"
        )

    print("[PASS] write_live_snapshot_signature_regression")
    print(
        f"[INFO] legacy_ok={ok_legacy} lightweight_ok={ok_light} "
        f"heavy_called_light={lw_flags['heavy_called']} light_called={lw_flags['light_called']}"
    )
    print(f"[INFO] metrics_lines_before={metrics_before} metrics_lines_after={metrics_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
