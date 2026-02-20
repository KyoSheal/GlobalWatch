#!/usr/bin/env python3
"""Verify summarize_range defaults to live-only and can include dryrun."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json
from run_analytics import summarize_range


def main() -> int:
    base = ROOT / "outputs" / "test_summarize_runs_default_live_only"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    month_dir = base / month
    live_dir = month_dir / "run-live-1"
    dry_dir = month_dir / "run-dry-1"
    live_dir.mkdir(parents=True, exist_ok=True)
    dry_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        str(live_dir / "run_summary.json"),
        {
            "schema_version": 1,
            "run_id": "run-live-1",
            "ended_at_utc": (now - timedelta(hours=1)).isoformat(),
            "final_equity": 101.0,
            "pnl": 1.0,
            "total_return": 0.01,
            "risk_profile": "mid",
            "run_kind": "live",
        },
        indent=2,
    )
    atomic_write_json(
        str(dry_dir / "run_summary.json"),
        {
            "schema_version": 1,
            "run_id": "run-dry-1",
            "ended_at_utc": (now - timedelta(hours=2)).isoformat(),
            "final_equity": 99.0,
            "pnl": -1.0,
            "total_return": -0.01,
            "risk_profile": "mid",
            "run_kind": "dryrun",
        },
        indent=2,
    )

    res_live_only = summarize_range(str(base), "1M", now_utc=now)
    if int(res_live_only.get("run_count", -1)) != 1:
        print(f"[FAIL] default run_count={res_live_only.get('run_count')} expected=1")
        return 1
    by_kind_live = res_live_only.get("run_count_by_kind", {}) or {}
    if int(by_kind_live.get("live", 0)) != 1:
        print(f"[FAIL] default run_count_by_kind={by_kind_live!r}")
        return 1

    res_with_dry = summarize_range(str(base), "1M", now_utc=now, kinds={"live", "dryrun"})
    if int(res_with_dry.get("run_count", -1)) != 2:
        print(f"[FAIL] include dryrun run_count={res_with_dry.get('run_count')} expected=2")
        return 1
    by_kind_with_dry = res_with_dry.get("run_count_by_kind", {}) or {}
    if int(by_kind_with_dry.get("live", 0)) != 1 or int(by_kind_with_dry.get("dryrun", 0)) != 1:
        print(f"[FAIL] include dryrun run_count_by_kind={by_kind_with_dry!r}")
        return 1

    print("[PASS] summarize_runs_default_live_only")
    print(
        f"[INFO] default={res_live_only.get('run_count')} "
        f"include_dryrun={res_with_dry.get('run_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

