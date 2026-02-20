#!/usr/bin/env python3
"""T12: risk-profile fields propagated across artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, safe_read_json
from paper_trading import debug_run_system_s1_s5


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


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def main() -> int:
    out_dir = ROOT / "outputs" / "test_risk_profile_artifacts_fields"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build temp config with risk_profile=high and telemetry enabled.
    base_cfg_path = ROOT / "paper_config.json"
    if not base_cfg_path.exists():
        return _fail(f"missing base config: {base_cfg_path}")
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("execution", {})["risk_profile"] = "high"
    rep = cfg.setdefault("reporting", {})
    rep["telemetry_enabled"] = True
    rep["out_dir"] = str(out_dir)
    cfg_path = ROOT / "outputs" / "test_risk_profile_artifacts_fields_config.json"
    atomic_write_json(str(cfg_path), cfg, indent=2)

    # Run shortest existing dryrun entry.
    rc = debug_run_system_s1_s5(config_path=str(cfg_path), outdir=str(out_dir))
    if int(rc) != 0:
        return _fail(f"debug_run_system_s1_s5 returned {rc}")

    # Resolve effective output paths under test-isolated layout.
    test_base_out = ROOT / "outputs" / "test"
    latest_ptr = safe_read_json(str(test_base_out / "LATEST.json"), retries=3, sleep_ms=20)
    latest_run_out_dir = None
    if isinstance(latest_ptr, dict):
        latest_out_raw = str(latest_ptr.get("out_dir", "") or "").strip()
        if latest_out_raw:
            latest_run_out_dir = Path(latest_out_raw)

    snapshot_path = _first_existing(
        [
            out_dir / "snapshot_live.json",
            test_base_out / "snapshot_live.json",
            (latest_run_out_dir / "snapshot_live.json") if latest_run_out_dir is not None else Path("__none__"),
        ]
    )
    if snapshot_path is None:
        return _fail("snapshot_live.json missing or unreadable")

    # A) snapshot fields
    snap = safe_read_json(str(snapshot_path), retries=3, sleep_ms=20)
    if not isinstance(snap, dict):
        return _fail("snapshot_live.json missing or unreadable")
    required_snapshot_fields = [
        "requested_risk_profile",
        "active_risk_profile",
        "risk_profile_template_version",
        "risk_profile_overrides_hash",
        "risk_profile_applied_cycle_id",
        "risk_profile_applied_at_utc",
        "source",
    ]
    missing_snapshot = [k for k in required_snapshot_fields if k not in snap]
    if missing_snapshot:
        return _fail(f"snapshot missing fields: {missing_snapshot}")

    # B) events any-of list + payload active_risk_profile
    run_id = str(snap.get("run_id", "") or "").strip()
    run_dir_by_id = None
    if run_id:
        run_id_matches = list((test_base_out).glob(f"*/*/{run_id}"))
        if run_id_matches:
            run_dir_by_id = run_id_matches[0]

    events_path = _first_existing(
        [
            out_dir / "telemetry" / "events.jsonl",
            (latest_run_out_dir / "telemetry" / "events.jsonl") if latest_run_out_dir is not None else Path("__none__"),
            (run_dir_by_id / "telemetry" / "events.jsonl") if run_dir_by_id is not None else Path("__none__"),
        ]
    )
    if events_path is None:
        return _fail("events.jsonl missing")
    events_rows = _read_jsonl(events_path)
    target_events = {
        "RISK_GATE_DECISION",
        "VOL_TARGET_APPLY",
        "PRICE_FETCH",
        "NEWS_OVERLAY",
        "REBALANCE_PLAN_FILTERED",
    }
    matched = []
    for row in events_rows:
        evt = str(row.get("event", "") or "")
        if evt not in target_events:
            continue
        payload = row.get("payload", {})
        if isinstance(payload, dict) and str(payload.get("active_risk_profile", "")).strip():
            matched.append(evt)
    if not matched:
        return _fail(
            "events missing any target event with payload.active_risk_profile "
            f"(targets={sorted(target_events)})"
        )

    # C) metrics CYCLE_METRICS payload fields
    metrics_path = _first_existing(
        [
            out_dir / "telemetry" / "metrics.jsonl",
            (latest_run_out_dir / "telemetry" / "metrics.jsonl") if latest_run_out_dir is not None else Path("__none__"),
            (run_dir_by_id / "telemetry" / "metrics.jsonl") if run_dir_by_id is not None else Path("__none__"),
        ]
    )
    if metrics_path is None:
        return _fail("metrics.jsonl missing")
    metrics_rows = _read_jsonl(metrics_path)
    cycle_metrics = None
    for row in metrics_rows:
        if str(row.get("event", "")) == "CYCLE_METRICS":
            cycle_metrics = row
            break
    if cycle_metrics is None:
        return _fail("metrics.jsonl missing CYCLE_METRICS")
    metrics_payload = cycle_metrics.get("payload", {})
    if not isinstance(metrics_payload, dict):
        return _fail("CYCLE_METRICS payload is not dict")
    for key in ("active_risk_profile", "requested_risk_profile", "risk_profile_overrides_hash"):
        if key not in metrics_payload:
            return _fail(f"CYCLE_METRICS payload missing {key}")

    # D) trade_history fields on written rows
    trade_history_path = _first_existing(
        [
            out_dir / "trade_history.jsonl",
            (latest_run_out_dir / "trade_history.jsonl") if latest_run_out_dir is not None else Path("__none__"),
            (run_dir_by_id / "trade_history.jsonl") if run_dir_by_id is not None else Path("__none__"),
        ]
    )
    if trade_history_path is None:
        return _fail("trade_history.jsonl has no rows; expected at least one new trade row")
    trade_rows = _read_jsonl(trade_history_path)
    if not trade_rows:
        return _fail("trade_history.jsonl has no rows; expected at least one new trade row")
    required_trade_fields = [
        "risk_profile",
        "risk_profile_overrides_hash",
        "risk_profile_template_version",
    ]
    for idx, row in enumerate(trade_rows):
        missing = [k for k in required_trade_fields if k not in row]
        if missing:
            return _fail(f"trade row {idx} missing fields: {missing}")

    # E) daily report fields (optional in this test flow)
    daily_dir = _first_existing(
        [
            out_dir / "Daily Report",
            test_base_out / "Daily Report",
            (latest_run_out_dir / "Daily Report") if latest_run_out_dir is not None else Path("__none__"),
        ]
    )
    daily_rows = []
    if isinstance(daily_dir, Path) and daily_dir.exists():
        for p in daily_dir.rglob("*.json"):
            payload = safe_read_json(str(p), retries=2, sleep_ms=15)
            if isinstance(payload, dict):
                daily_rows.append((p, payload))
    daily_report_status = "SKIP: daily report not generated in this dryrun flow"
    if daily_rows:
        # validate first report
        p, report = daily_rows[0]
        missing_daily = []
        for key in ("risk_profile", "risk_profile_template_version", "risk_profile_overrides_hash"):
            if key not in report:
                missing_daily.append(key)
        if missing_daily:
            return _fail(f"daily report {p} missing fields: {missing_daily}")
        daily_report_status = f"PASS: daily report meta present ({p})"

    print("[PASS] risk_profile_artifacts_fields")
    print(
        "[INFO] snapshot active/requested="
        f"{snap.get('active_risk_profile')}/{snap.get('requested_risk_profile')} "
        f"source={snap.get('source')}"
    )
    print(f"[INFO] matched_event={matched[0]} total_events={len(events_rows)}")
    print(f"[INFO] metrics_rows={len(metrics_rows)} trade_rows={len(trade_rows)}")
    print(f"[INFO] daily_report={daily_report_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
