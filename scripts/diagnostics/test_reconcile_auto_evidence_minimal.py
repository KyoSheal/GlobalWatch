#!/usr/bin/env python3
"""T54: A4-8 reconcile auto-evidence should fill evidence_summary deterministically."""

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
    tmp = Path(tempfile.mkdtemp(prefix="reconcile_auto_ev_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        date_str = "2026-02-21"
        daily_base.mkdir(parents=True, exist_ok=True)
        daily_json = daily_base / f"{date_str}.json"

        report_obj = {
            "date": date_str,
            "summary": {},
            "quant_pack": {
                "summary": {
                    "trades_total": 0,
                    "total_return": 0.01,
                    "max_drawdown": -0.02,
                },
                "backtest_from_run": {
                    "status": "OK",
                    "total_return": 0.02,
                    "max_drawdown": -0.03,
                    "trade_rows": 4,
                    "turnover_notional": 15000.0,
                    "total_cost": 15.0,
                },
            },
        }
        with daily_json.open("w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)

        metrics_dir = (daily_base / "quant_packs" / date_str / "metrics").resolve()
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_obj = {
            "performance": {"total_return": 0.01},
            "risk": {"max_drawdown": -0.02},
            "trading": {"trades_total": 0},
            "gating": {
                "summary": {
                    "top3": [
                        {"reason": "attempt_cooldown", "count": 5},
                        {"reason": "market_closed", "count": 3},
                    ]
                }
            },
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
            "--auto-evidence",
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
        rec = ((obj.get("quant_pack") or {}).get("reconcile")) if isinstance((obj.get("quant_pack") or {}), dict) else None
        if not isinstance(rec, dict):
            return _fail("missing quant_pack.reconcile")
        ev = rec.get("evidence_summary") if isinstance(rec.get("evidence_summary"), dict) else {}
        if not ev:
            return _fail("missing evidence_summary")

        gate_status = str(ev.get("gate_status", "") or "")
        replay_status = str(ev.get("replay_drift_status", "") or "")
        if gate_status in ("", "NA"):
            return _fail(f"gate_status not filled: {gate_status!r}")
        if replay_status in ("", "NA"):
            return _fail(f"replay_drift_status not filled: {replay_status!r}")
        gt = ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else []
        if not gt:
            return _fail("gating_top3 should not be empty with metrics fallback")

        # update index and verify reconcile evidence projection
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

        idx = json.load(open(daily_base / "daily_reports_index.json", "r", encoding="utf-8"))
        rows = idx.get("reports") if isinstance(idx.get("reports"), list) else []
        row = None
        for r in rows:
            if isinstance(r, dict) and str(r.get("date", "")) == date_str:
                row = r
                break
        if not isinstance(row, dict):
            return _fail("index row missing")
        qrec = (((row.get("quant") or {}).get("reconcile")) if isinstance((row.get("quant") or {}), dict) else None)
        if not isinstance(qrec, dict):
            return _fail("index quant.reconcile missing")
        if str(qrec.get("gate_status", "") or "") in ("", "NA"):
            return _fail("index gate_status not projected")
        if str(qrec.get("replay_drift_status", "") or "") in ("", "NA"):
            return _fail("index replay_drift_status not projected")

        print("[PASS] reconcile_auto_evidence_minimal")
        print(f"[INFO] daily_json={daily_json}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

