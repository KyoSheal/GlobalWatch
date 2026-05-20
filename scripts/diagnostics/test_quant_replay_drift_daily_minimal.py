#!/usr/bin/env python3
"""T45: minimal replay drift -> daily report + index integration test."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as _f: _f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def _write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _run_a12(daily_base: Path, date_str: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a12_attach_replay_drift_to_daily.py"),
        "--daily-base",
        str(daily_base),
        "--date",
        date_str,
        "--no-strict",
        "--no-fail-on-drift",
        "--verbose",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def _run_a7(daily_base: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
        "--daily-base",
        str(daily_base),
        "--lookback-days",
        "90",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_replay_drift_daily_min_"))
    try:
        date_str = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        daily_base = tmp / "outputs" / "Daily Report"
        report_path = daily_base / f"{date_str}.json"
        _write_json(report_path, {"date": date_str, "title": "Daily"})

        replay_dir = daily_base / "quant_packs" / date_str / "replay_window" / "auto"
        per_cycle = replay_dir / "per_cycle" / "20"
        per_cycle.mkdir(parents=True, exist_ok=True)

        _write_json(
            replay_dir / "replay_window_manifest.json",
            {"schema_version": 1, "status": "ok", "warnings": []},
        )
        _write_csv(
            replay_dir / "replay_window_summary.csv",
            [
                "cycle",
                "time_utc",
                "price_rows",
                "num_trades",
                "target_hash",
                "trades_hash",
                "gate_fail",
                "warnings_count",
                "ref_status",
                "attribution_tags",
                "weights_l1",
                "trades_notional_delta",
                "diff_path",
                "decision_path",
            ],
            [
                {
                    "cycle": 20,
                    "time_utc": f"{date_str}T15:00:00+00:00",
                    "price_rows": 3,
                    "num_trades": 2,
                    "target_hash": "th",
                    "trades_hash": "tr",
                    "gate_fail": "False",
                    "warnings_count": 0,
                    "ref_status": "ok",
                    "attribution_tags": "",
                    "weights_l1": "0.000000",
                    "trades_notional_delta": "0",
                    "diff_path": "per_cycle/20/diff.json",
                    "decision_path": "per_cycle/20/replay_decision.md",
                }
            ],
        )
        _write_json(
            per_cycle / "diff.json",
            {
                "weights_diff": {"weights_l1": 0.0, "top_deltas": []},
                "trades_diff": {"notional_delta": 0.0},
                "attribution_tags": [],
            },
        )
        _write_json(per_cycle / "replay_manifest.json", {"snapshot": {"total_equity": 100000.0}})

        p1 = _run_a12(daily_base, date_str)
        if p1.returncode not in (0, 1):
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a12 first run rc expected 0/1 got={p1.returncode}")

        obj1 = json.loads(report_path.read_text(encoding="utf-8"))
        qp1 = obj1.get("quant_pack") if isinstance(obj1.get("quant_pack"), dict) else {}
        rd1 = qp1.get("replay_drift") if isinstance(qp1.get("replay_drift"), dict) else None
        if not isinstance(rd1, dict) or not rd1.get("status"):
            return _fail("quant_pack.replay_drift missing after first run")

        bak = report_path.with_name(report_path.name + ".bak")
        if not bak.exists():
            return _fail("daily report backup .bak missing after attach")

        p2 = _run_a12(daily_base, date_str)
        if p2.returncode not in (0, 1):
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a12 second run rc expected 0/1 got={p2.returncode}")

        obj2 = json.loads(report_path.read_text(encoding="utf-8"))
        qp2 = obj2.get("quant_pack") if isinstance(obj2.get("quant_pack"), dict) else {}
        rd2 = qp2.get("replay_drift") if isinstance(qp2.get("replay_drift"), dict) else None
        if not isinstance(rd2, dict):
            return _fail("quant_pack.replay_drift missing after second run")
        if sorted(rd1.keys()) != sorted(rd2.keys()):
            return _fail("replay_drift structure changed unexpectedly on rerun")

        p3 = _run_a7(daily_base)
        if p3.returncode != 0:
            print(p3.stdout)
            print(p3.stderr)
            return _fail(f"a7 update index rc expected 0 got={p3.returncode}")

        idx_path = daily_base / "daily_reports_index.json"
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        rows = idx.get("reports") if isinstance(idx.get("reports"), list) else []
        row = next((r for r in rows if isinstance(r, dict) and str(r.get("date")) == date_str), None)
        if not isinstance(row, dict):
            return _fail("index missing target date row")
        quant = row.get("quant") if isinstance(row.get("quant"), dict) else {}
        if not isinstance(quant.get("replay_drift"), dict):
            return _fail("index quant.replay_drift missing")

        print("[PASS] quant_replay_drift_daily_minimal")
        print(f"[INFO] daily_base={daily_base}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
