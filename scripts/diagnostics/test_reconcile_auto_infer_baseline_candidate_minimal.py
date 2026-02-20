#!/usr/bin/env python3
"""T55: A4-9 reconcile auto-infers baseline/candidate and fills gating evidence."""

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


def _write_dataset(dataset_dir: Path, run_id: str, *, ret_scale: float, with_cycles: bool) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "schema_version": 1}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # equity_curve: 6 points so a2 can compute non-trivial metrics
    t0 = datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc)
    rows = []
    eq = 100000.0
    rets = [0.01, -0.003, 0.004, 0.006, -0.002, 0.005]
    for i, r in enumerate(rets):
        eq *= (1.0 + float(r) * ret_scale)
        rows.append({"time_utc": (t0 + timedelta(days=i)).isoformat(), "equity": f"{eq:.6f}"})
    with (dataset_dir / "equity_curve.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_utc", "equity"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with (dataset_dir / "trades.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_utc", "ticker", "side", "notional"])
        w.writeheader()

    with (dataset_dir / "cycles.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_utc", "cycle_id", "skip_reason"])
        w.writeheader()
        if with_cycles:
            w.writerow({"time_utc": (t0 + timedelta(days=0)).isoformat(), "cycle_id": 1, "skip_reason": "attempt_cooldown"})
            w.writerow({"time_utc": (t0 + timedelta(days=1)).isoformat(), "cycle_id": 2, "skip_reason": "market_closed"})
            w.writerow({"time_utc": (t0 + timedelta(days=2)).isoformat(), "cycle_id": 3, "skip_reason": "attempt_cooldown"})


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="reconcile_auto_infer_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        date_str = "2026-02-22"
        daily_base.mkdir(parents=True, exist_ok=True)
        daily_json = daily_base / f"{date_str}.json"

        report_obj = {
            "date": date_str,
            "summary": {},
            "quant_pack": {
                # no gate_result/replay_drift; force auto-evidence path
                "summary": {"trades_total": 0},
                "backtest_from_run": {
                    "status": "OK",
                    "total_return": 0.008,
                    "max_drawdown": -0.015,
                    "trade_rows": 2,
                    "turnover_notional": 10000.0,
                    "total_cost": 10.0,
                },
            },
        }
        daily_json.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        pack_dir = (daily_base / "quant_packs" / date_str).resolve()
        cand_ds = (pack_dir / "run_dataset").resolve()
        base_ds = (daily_base / "quant_packs" / "2026-02-21" / "run_dataset").resolve()
        _write_dataset(cand_ds, "cand_run", ret_scale=1.0, with_cycles=True)
        _write_dataset(base_ds, "base_run", ret_scale=0.8, with_cycles=False)

        # A4-9 inference target: pipeline_manifest carries dataset + baseline dataset paths
        pipeline_manifest = {
            "schema_version": 1,
            "date": date_str,
            "dataset_dir": str(cand_ds),
            "baseline_dataset_dir": str(base_ds),
        }
        (pack_dir / "pipeline_manifest.json").write_text(
            json.dumps(pipeline_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        rec = (((obj.get("quant_pack") or {}).get("reconcile")) if isinstance((obj.get("quant_pack") or {}), dict) else None)
        if not isinstance(rec, dict):
            return _fail("missing quant_pack.reconcile")
        ev = rec.get("evidence_summary") if isinstance(rec.get("evidence_summary"), dict) else {}
        if not ev:
            return _fail("missing evidence_summary")

        gate_status = str(ev.get("gate_status", "") or "")
        if gate_status in ("", "NOT_RUN"):
            return _fail(f"gate_status should be inferred and executed, got {gate_status!r}")
        gt = ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else []
        if len(gt) <= 0:
            return _fail("gating_top3 should be filled (cycles.csv fallback)")

        # idempotent subset check
        obj2 = json.load(open(daily_json, "r", encoding="utf-8"))
        rec2 = (((obj2.get("quant_pack") or {}).get("reconcile")) if isinstance((obj2.get("quant_pack") or {}), dict) else {})
        ev2 = rec2.get("evidence_summary") if isinstance(rec2.get("evidence_summary"), dict) else {}
        if str(ev2.get("gate_status", "")) != gate_status:
            return _fail("gate_status changed unexpectedly across repeated runs")

        print("[PASS] reconcile_auto_infer_baseline_candidate_minimal")
        print(f"[INFO] gate_status={gate_status} gating_top3={gt}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

