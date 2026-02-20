#!/usr/bin/env python3
"""T56: minimal regression for A4-10 index timeseries builder."""

from __future__ import annotations

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


def _build_index_obj(base: Path) -> dict:
    # stored in descending order intentionally; a19 should output ascending by date
    return {
        "updated_at": "2026-02-20T00:00:00+00:00",
        "report_dir": str(base),
        "reports": [
            {
                "date": "2026-02-19",
                "quant": {
                    "total_return": 0.01,
                    "sharpe": 1.2,
                    "max_drawdown": -0.03,
                    "trades_total": 2,
                    "gate_status": "PASS",
                    "reconcile": {
                        "status": "OK",
                        "return_gap_live_minus_backtest": -0.002,
                        "turnover_gap": -100.0,
                        "cost_gap": -1.0,
                        "warnings_count": 1,
                        "evidence_summary": {
                            "gate_status": "PASS",
                            "replay_drift_status": "PASS",
                            "gating_top3": [{"reason": "attempt_cooldown", "count": 1}],
                        },
                    },
                },
            },
            {
                "date": "2026-02-18",
                "quant": {
                    "total_return": -0.004,
                    "sharpe": 0.4,
                    "max_drawdown": -0.05,
                    "trades_total": 0,
                    "gate_status": "FAIL",
                    "reconcile": {
                        "status": "OK",
                        "return_gap_live_minus_backtest": -0.006,
                        "turnover_gap": 0.0,
                        "cost_gap": 0.0,
                        "warnings_count": 2,
                        "evidence_summary": {
                            "gate_status": "FAIL",
                            "replay_drift_status": "MISSING",
                            "gating_top3": [{"reason": "market_closed", "count": 2}],
                        },
                    },
                },
            },
        ],
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="index_timeseries_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)
        idx = _build_index_obj(daily_base)
        (daily_base / "daily_reports_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a19_build_index_timeseries.py"),
            "--daily-base",
            str(daily_base),
            "--lookback-days",
            "3650",
        ]
        p1 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a19 first run failed rc={p1.returncode}")
        csv_path = daily_base / "index_timeseries.csv"
        json_path = daily_base / "index_timeseries.json"
        if not csv_path.exists() or not json_path.exists():
            return _fail("missing index_timeseries outputs")
        csv_text_1 = csv_path.read_text(encoding="utf-8")
        json_text_1 = json_path.read_text(encoding="utf-8")

        p2 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a19 second run failed rc={p2.returncode}")
        csv_text_2 = csv_path.read_text(encoding="utf-8")
        json_text_2 = json_path.read_text(encoding="utf-8")
        if csv_text_1 != csv_text_2 or json_text_1 != json_text_2:
            return _fail("a19 outputs are not idempotent")

        obj = json.loads(json_text_1)
        rows = obj.get("rows") if isinstance(obj.get("rows"), list) else []
        if len(rows) != 2:
            return _fail(f"unexpected row count: {len(rows)}")
        if [r.get("date") for r in rows] != ["2026-02-18", "2026-02-19"]:
            return _fail("rows not sorted ascending by date")
        required = {
            "date",
            "total_return",
            "sharpe",
            "max_drawdown",
            "trades_total",
            "gate_status",
            "replay_drift_status",
            "reconcile_return_gap",
            "reconcile_turnover_gap",
            "reconcile_cost_gap",
            "gating_top1",
            "warnings_count",
        }
        if not required.issubset(set(obj.get("columns", []))):
            return _fail("missing required columns in index_timeseries.json")

        print("[PASS] index_timeseries_minimal")
        print(f"[INFO] rows={len(rows)} csv={csv_path}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

