#!/usr/bin/env python3
"""T46: CI demo for replay drift gate (nonstrict should not fail)."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ci_replay_drift_demo_"))
    try:
        artifacts_dir = (ROOT / "outputs" / "ci_artifacts" / "T46").resolve()
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir, ignore_errors=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        replay_dir = tmp / "replay_window" / "demo"
        per_cycle = replay_dir / "per_cycle" / "30"
        per_cycle.mkdir(parents=True, exist_ok=True)

        _write_json(
            replay_dir / "replay_window_manifest.json",
            {"schema_version": 1, "status": "ok", "warnings": ["cycle_30:macro_not_frozen"]},
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
                    "cycle": 30,
                    "time_utc": "2026-02-20T12:00:00+00:00",
                    "price_rows": 4,
                    "num_trades": 1,
                    "target_hash": "th30",
                    "trades_hash": "tr30",
                    "gate_fail": "False",
                    "warnings_count": 1,
                    "ref_status": "ok",
                    "attribution_tags": "NONDETERMINISM_WARNING|macro_not_frozen",
                    "weights_l1": "0.000000",
                    "trades_notional_delta": "0",
                    "diff_path": "per_cycle/30/diff.json",
                    "decision_path": "per_cycle/30/replay_decision.md",
                }
            ],
        )
        _write_json(
            per_cycle / "diff.json",
            {
                "weights_diff": {"weights_l1": 0.0, "top_deltas": []},
                "trades_diff": {"notional_delta": 0.0},
                "attribution_tags": ["NONDETERMINISM_WARNING", "macro_not_frozen"],
            },
        )
        _write_json(per_cycle / "replay_manifest.json", {"snapshot": {"total_equity": 100000.0}})

        out_dir = artifacts_dir
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a11_replay_drift_gate.py"),
            "--replay-window-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--no-strict",
            "--no-fail-on-drift",
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"expected rc in (0,1), got {proc.returncode}")

        result_path = out_dir / "drift_gate_result.json"
        report_path = out_dir / "drift_gate_report.md"
        if not result_path.exists() or not report_path.exists():
            return _fail("missing drift gate outputs")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = str(result.get("status", "")).upper()
        if status == "FAIL":
            return _fail("nonstrict demo status must not be FAIL")

        report_text = report_path.read_text(encoding="utf-8")
        if "Tag Counts (Top5)" not in report_text:
            return _fail("report missing tag section")
        if "NONDETERMINISM_WARNING" not in report_text and "macro_not_frozen" not in report_text:
            return _fail("report missing expected tag details")

        print("[PASS] ci_replay_drift_gate_demo")
        print(f"[INFO] replay_dir={replay_dir}")
        print(f"[INFO] artifacts_dir={artifacts_dir}")
        print(f"[INFO] status={status}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
