#!/usr/bin/env python3
"""T52: minimal regression for A4-6 daily live-vs-backtest reconcile."""

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


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="reconcile_live_bt_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)
        date_str = "2026-02-18"
        daily_json = daily_base / f"{date_str}.json"
        report_obj = {
            "date": date_str,
            "summary": {},
            "quant_pack": {
                "summary": {
                    "total_return": 0.012,
                    "max_drawdown": -0.02,
                    "trades_total": 0,
                    "gate_status": "FAIL",
                },
                "backtest_from_run": {
                    "status": "OK",
                    "generated_utc": "2026-02-20T10:00:00+00:00",
                    "total_return": 0.021,
                    "max_drawdown": -0.015,
                    "trade_rows": 3,
                    "turnover_notional": 120000.0,
                    "total_cost": 80.0,
                    "days": 1,
                },
                "replay_drift": {"status": "FAIL"},
            },
        }
        with daily_json.open("w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a18_reconcile_live_vs_backtest.py"),
            "--daily-base",
            str(daily_base),
            "--date",
            date_str,
            "--verbose",
        ]
        p1 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p1.returncode not in (0, 1):
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a18 first run failed rc={p1.returncode}")
        p2 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p2.returncode not in (0, 1):
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a18 second run failed rc={p2.returncode}")

        obj = json.load(open(daily_json, "r", encoding="utf-8"))
        qp = obj.get("quant_pack") if isinstance(obj.get("quant_pack"), dict) else {}
        rec = qp.get("reconcile") if isinstance(qp.get("reconcile"), dict) else {}
        if not rec:
            return _fail("missing quant_pack.reconcile")
        gaps = rec.get("gaps") if isinstance(rec.get("gaps"), dict) else {}
        for key in ("return_gap_live_minus_backtest", "drawdown_gap", "turnover_gap", "cost_gap"):
            if key not in gaps:
                return _fail(f"missing gap key: {key}")
        attr = rec.get("attribution") if isinstance(rec.get("attribution"), dict) else {}
        if "likely_driver_top3" not in attr:
            return _fail("missing attribution.likely_driver_top3")

        # update index and assert projection
        p3 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "365",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p3.returncode != 0:
            print(p3.stdout)
            print(p3.stderr)
            return _fail(f"a7 update index failed rc={p3.returncode}")

        index_obj = json.load(open(daily_base / "daily_reports_index.json", "r", encoding="utf-8"))
        reports = index_obj.get("reports") if isinstance(index_obj.get("reports"), list) else []
        row = None
        for r in reports:
            if isinstance(r, dict) and str(r.get("date", "")) == date_str:
                row = r
                break
        if not isinstance(row, dict):
            return _fail("index row for date not found")
        q = row.get("quant") if isinstance(row.get("quant"), dict) else {}
        qrec = q.get("reconcile") if isinstance(q.get("reconcile"), dict) else {}
        if "return_gap_live_minus_backtest" not in qrec:
            return _fail("index quant.reconcile.return_gap_live_minus_backtest missing")

        print("[PASS] reconcile_live_vs_backtest_minimal")
        print(f"[INFO] daily_json={daily_json}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

