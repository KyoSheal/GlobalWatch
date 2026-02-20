#!/usr/bin/env python3
"""T53: A4-7 reconcile should fill live metrics and compute gaps deterministically."""

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
    tmp = Path(tempfile.mkdtemp(prefix="reconcile_fill_live_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        date_str = "2026-02-20"
        daily_base.mkdir(parents=True, exist_ok=True)
        daily_json = daily_base / f"{date_str}.json"

        # daily json intentionally lacks live return/dd in summary so reconcile must fill from metrics.json
        report_obj = {
            "date": date_str,
            "summary": {},
            "quant_pack": {
                "summary": {
                    "trades_total": 0,
                    "gate_status": "PASS",
                },
                "backtest_from_run": {
                    "status": "OK",
                    "generated_utc": "2026-02-20T10:00:00+00:00",
                    "total_return": 0.01,
                    "max_drawdown": -0.02,
                    "trade_rows": 3,
                    "turnover_notional": 12000.0,
                    "total_cost": 12.0,
                },
            },
        }
        with daily_json.open("w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)

        metrics_dir = (daily_base / "quant_packs" / date_str / "metrics").resolve()
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_obj = {
            "meta": {"run_id": "demo_run"},
            "performance": {"total_return": 0.015},
            "risk": {"max_drawdown": -0.03},
            "trading": {"trades_total": 0},
        }
        with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2)

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

        live = rec.get("live") if isinstance(rec.get("live"), dict) else {}
        if live.get("total_return") is None:
            return _fail("live.total_return should be filled")
        if live.get("max_drawdown") is None:
            return _fail("live.max_drawdown should be filled")
        if float(live.get("turnover_notional", -1.0)) != 0.0:
            return _fail("live.turnover_notional should be 0.0 when trades_total == 0")
        if float(live.get("total_cost", -1.0)) != 0.0:
            return _fail("live.total_cost should be 0.0 when trades_total == 0")

        gaps = rec.get("gaps") if isinstance(rec.get("gaps"), dict) else {}
        if gaps.get("return_gap_live_minus_backtest") is None:
            return _fail("return gap should be computable")

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
        rows = index_obj.get("reports") if isinstance(index_obj.get("reports"), list) else []
        row = None
        for item in rows:
            if isinstance(item, dict) and str(item.get("date", "")) == date_str:
                row = item
                break
        if not isinstance(row, dict):
            return _fail("index row not found")
        q = row.get("quant") if isinstance(row.get("quant"), dict) else {}
        qrec = q.get("reconcile") if isinstance(q.get("reconcile"), dict) else {}
        if "warnings_count" not in qrec:
            return _fail("index quant.reconcile.warnings_count missing")

        print("[PASS] reconcile_live_vs_backtest_fill_live_minimal")
        print(f"[INFO] daily_json={daily_json}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

