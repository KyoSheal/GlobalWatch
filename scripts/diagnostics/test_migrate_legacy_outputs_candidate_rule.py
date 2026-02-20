#!/usr/bin/env python3
"""Verify migrate_legacy_outputs only imports candidate run dirs."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, atomic_write_text
from scripts.diagnostics.migrate_legacy_outputs import migrate_legacy_outputs


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


def main() -> int:
    root = ROOT / "outputs" / "test_migrate_legacy_outputs_candidate_rule"
    legacy_root = root / "legacy"
    base_out_dir = root / "base"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    legacy_root.mkdir(parents=True, exist_ok=True)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # live candidate (has trade_history.jsonl)
    live_dir = legacy_root / "live_run_a"
    live_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        str(live_dir / "trade_history.jsonl"),
        json.dumps({"run_id": "LIVE-RUN-A", "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n",
    )
    atomic_write_json(
        str(live_dir / "run_summary.json"),
        {
            "schema_version": 1,
            "run_id": "LIVE-RUN-A",
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "risk_profile": "mid",
        },
        indent=2,
    )

    # diagnostics-like dir (telemetry only) -> should not import
    diag_dir = legacy_root / "diag_only_b"
    (diag_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        str(diag_dir / "telemetry" / "events.jsonl"),
        json.dumps({"event": "PRICE_FETCH", "ts_utc": datetime.now(timezone.utc).isoformat()}) + "\n",
    )

    # empty/noisy test dir -> should not import
    (legacy_root / "tmp_test_c").mkdir(parents=True, exist_ok=True)

    result = migrate_legacy_outputs(
        legacy_root=str(legacy_root),
        base_out_dir=str(base_out_dir),
        mode="copy",
        dry_run=False,
        update_latest=False,
        build_month_summaries=False,
    )

    if int(result.get("imported_count", -1)) != 1:
        print(f"[FAIL] imported_count={result.get('imported_count')} expected=1")
        return 1

    registry_rows = _read_jsonl(base_out_dir / "registry.jsonl")
    if len(registry_rows) != 2:
        print(f"[FAIL] registry row count={len(registry_rows)} expected=2 (start+end)")
        return 1
    for row in registry_rows:
        if str(row.get("run_kind", "")).strip().lower() != "live":
            print(f"[FAIL] registry run_kind invalid: {row.get('run_kind')!r}")
            return 1
        if str(row.get("run_id", "")).strip() != "LIVE-RUN-A":
            print(f"[FAIL] unexpected run_id in registry: {row.get('run_id')!r}")
            return 1

    print("[PASS] migrate_legacy_outputs_candidate_rule")
    print(f"[INFO] imported_count={result.get('imported_count')} registry_rows={len(registry_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

