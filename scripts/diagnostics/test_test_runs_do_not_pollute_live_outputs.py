#!/usr/bin/env python3
"""Ensure dryrun/test runs are isolated under outputs/test and do not pollute live aliases."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, safe_read_json
from paper_trading import PaperTradingEngine


MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    sandbox = ROOT / "outputs" / "test_isolation_guard"
    live_root = sandbox / "outputs"
    test_root = live_root / "test"
    if sandbox.exists():
        shutil.rmtree(sandbox, ignore_errors=True)
    live_root.mkdir(parents=True, exist_ok=True)

    # Seed live aliases/registry with sentinel payloads.
    live_latest = live_root / "LATEST.json"
    live_registry = live_root / "registry.jsonl"
    live_snapshot = live_root / "snapshot_live.json"
    atomic_write_json(
        str(live_latest),
        {"schema_version": 1, "run_id": "LIVE-SENTINEL", "updated_at_utc": "2026-02-01T00:00:00+00:00"},
        indent=2,
    )
    live_registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ts_utc": "2026-02-01T00:00:00+00:00",
                "run_id": "LIVE-SENTINEL",
                "action": "start",
                "run_kind": "live",
                "out_dir": str((live_root / "2026-02" / "LIVE-SENTINEL").resolve()),
                "month_dir": "2026-02",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    atomic_write_json(
        str(live_snapshot),
        {"schema_version": 2, "run_id": "LIVE-SENTINEL", "cycle": 99, "cash": 30310.0},
        indent=2,
    )
    latest_before = live_latest.read_text(encoding="utf-8")
    registry_before = live_registry.read_text(encoding="utf-8")
    snapshot_before = live_snapshot.read_text(encoding="utf-8")
    latest_mtime_before = live_latest.stat().st_mtime
    registry_mtime_before = live_registry.stat().st_mtime
    snapshot_mtime_before = live_snapshot.stat().st_mtime

    base_cfg = json.loads((ROOT / "paper_config.json").read_text(encoding="utf-8"))
    base_cfg["run_mode"] = "dryrun"
    rep = base_cfg.setdefault("reporting", {})
    rep["base_out_dir"] = str(live_root)
    rep["out_dir"] = str(live_root / "manual_dryrun_out")
    rep["snapshot_live_path"] = str(live_snapshot)
    rep["telemetry_enabled"] = False
    rep["enable_daily_report"] = False
    rep["enable_eod_report"] = False
    rep["daily_report_dirs"] = [str(live_root / "Daily Report")]
    cfg_path = sandbox / "cfg_dryrun_isolation.json"
    atomic_write_json(str(cfg_path), base_cfg, indent=2)

    old_checkpoint = os.environ.get("GW_CHECKPOINT_ACTION")
    os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
    try:
        engine = PaperTradingEngine(str(cfg_path))
    finally:
        if old_checkpoint is None:
            os.environ.pop("GW_CHECKPOINT_ACTION", None)
        else:
            os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint

    if str(engine.run_kind) not in {"dryrun", "test", "diagnostics"}:
        return _fail(f"unexpected run_kind={engine.run_kind!r}")

    reporting_cfg = engine.config.get("reporting", {})
    effective_base = Path(str(reporting_cfg.get("base_out_dir"))).resolve()
    effective_out_dir = Path(str(reporting_cfg.get("out_dir"))).resolve()
    effective_snapshot = Path(str(reporting_cfg.get("snapshot_live_path"))).resolve()

    if effective_base != test_root.resolve():
        return _fail(f"base_out_dir not routed to outputs/test: {effective_base}")
    if str(test_root.resolve()) not in str(effective_out_dir):
        return _fail(f"out_dir not under outputs/test: {effective_out_dir}")
    if effective_snapshot != (test_root / "snapshot_live.json").resolve():
        return _fail(f"snapshot_live_path not routed to outputs/test alias: {effective_snapshot}")

    # Trigger writes to ensure registry/snapshot updates happen in test root only.
    engine._write_run_start_record()
    ok = engine.write_live_snapshot(
        {
            "timestamp": "2026-02-16T00:00:00+00:00",
            "cycle": 1,
            "cash": 29999.0,
            "positions_value": 1.0,
            "total_equity": 30000.0,
            "positions": {},
        },
        source="test_isolation_guard",
        emit_telemetry=False,
        emit_cycle_metrics=False,
        lightweight=True,
    )
    if not ok:
        return _fail("write_live_snapshot failed")

    # Live aliases/registry must remain untouched.
    if live_latest.read_text(encoding="utf-8") != latest_before:
        return _fail("live LATEST.json content changed")
    if live_registry.read_text(encoding="utf-8") != registry_before:
        return _fail("live registry.jsonl content changed")
    if live_snapshot.read_text(encoding="utf-8") != snapshot_before:
        return _fail("live snapshot_live.json content changed")
    if live_latest.stat().st_mtime != latest_mtime_before:
        return _fail("live LATEST.json mtime changed")
    if live_registry.stat().st_mtime != registry_mtime_before:
        return _fail("live registry.jsonl mtime changed")
    if live_snapshot.stat().st_mtime != snapshot_mtime_before:
        return _fail("live snapshot_live.json mtime changed")

    # Ensure test artifacts were actually written.
    test_latest = test_root / "LATEST.json"
    test_registry = test_root / "registry.jsonl"
    test_snapshot = test_root / "snapshot_live.json"
    if not test_latest.exists() or not test_registry.exists() or not test_snapshot.exists():
        return _fail("expected test aliases missing under outputs/test")
    test_latest_obj = safe_read_json(str(test_latest), retries=2, sleep_ms=10) or {}
    if str(test_latest_obj.get("run_kind", "")) not in {"dryrun", "test", "diagnostics"}:
        return _fail("test LATEST.json missing non-live run_kind")

    # No month dirs should be created directly under live root by this dryrun.
    polluted_month_dirs = [p.name for p in live_root.iterdir() if p.is_dir() and MONTH_RE.match(p.name)]
    if polluted_month_dirs:
        return _fail(f"dryrun created live month dirs: {polluted_month_dirs}")

    print("[PASS] test_runs_do_not_pollute_live_outputs")
    print(f"[INFO] run_kind={engine.run_kind} base={effective_base}")
    print(f"[INFO] out_dir={effective_out_dir}")
    print(f"[INFO] snapshot={effective_snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

