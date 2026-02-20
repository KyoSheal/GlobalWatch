#!/usr/bin/env python3
"""T41: minimal regression test for A2-2 daily quant pipeline."""

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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _mk_dataset(path: Path, run_id: str, equities: list[float], skip_reasons: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "manifest.json", {"run_id": run_id})

    t0 = datetime(2026, 2, 18, 14, 30, tzinfo=timezone.utc)
    eq_rows = []
    cycles = []
    trades = []
    for i, eq in enumerate(equities):
        ts = (t0 + timedelta(hours=i)).isoformat()
        eq_rows.append({"time_utc": ts, "equity": f"{eq:.6f}", "cash": "1000", "positions_value": f"{eq - 1000:.6f}"})
        reason = skip_reasons[i] if i < len(skip_reasons) else ""
        cycles.append(
            {
                "cycle_id": str(i + 1),
                "time_utc": ts,
                "session_state": "open",
                "regime_state": "neutral",
                "cash_target": "0.1",
                "total_equity": f"{eq:.6f}",
                "cash": "1000",
                "positions_value": f"{eq - 1000:.6f}",
                "skip_reason": reason,
                "decision_path": "",
                "cov_gate_reason": "",
                "cov_gate_max_rc": "",
                "rc_limit": "",
                "turnover_used_total": "",
            }
        )
        if i % 2 == 0:
            trades.append(
                {
                    "time_utc": ts,
                    "cycle_id": str(i + 1),
                    "ticker": "QQQ" if i % 4 == 0 else "SPY",
                    "side": "BUY" if i % 4 == 0 else "SELL",
                    "qty": "1",
                    "price": "100",
                    "notional": "100",
                    "is_forced": "False",
                    "force_reason": "",
                    "status": "ok",
                    "reason": "",
                }
            )

    _write_csv(path / "equity_curve.csv", ["time_utc", "equity", "cash", "positions_value"], eq_rows)
    _write_csv(
        path / "cycles.csv",
        [
            "cycle_id",
            "time_utc",
            "session_state",
            "regime_state",
            "cash_target",
            "total_equity",
            "cash",
            "positions_value",
            "skip_reason",
            "decision_path",
            "cov_gate_reason",
            "cov_gate_max_rc",
            "rc_limit",
            "turnover_used_total",
        ],
        cycles,
    )
    _write_csv(
        path / "trades.csv",
        ["time_utc", "cycle_id", "ticker", "side", "qty", "price", "notional", "is_forced", "force_reason", "status", "reason"],
        trades,
    )


def _run_pipeline(*, daily_base: Path, date_str: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a8_run_daily_quant_pipeline.py"),
        "--daily-base",
        str(daily_base),
        "--date",
        str(date_str),
        "--lookback-days",
        "30",
        "--verbose",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="daily_quant_pipeline_min_"))
    try:
        daily_base = tmp_root / "outputs" / "Daily Report"
        prev_date = "2026-02-18"
        date_str = "2026-02-19"

        _write_json(daily_base / f"{prev_date}.json", {"date": prev_date, "title": "Prev Day"})
        _write_json(daily_base / f"{date_str}.json", {"date": date_str, "title": "Today"})

        _mk_dataset(
            daily_base / "quant_packs" / prev_date / "run_dataset",
            "run-prev",
            [100000, 100100, 100200, 100240, 100260, 100300],
            ["", "", "attempt_cooldown", "", "", ""],
        )
        _mk_dataset(
            daily_base / "quant_packs" / date_str / "run_dataset",
            "run-today",
            [100000, 99980, 100050, 100120, 100090, 100180],
            ["market_closed", "", "", "", "", ""],
        )

        p1 = _run_pipeline(daily_base=daily_base, date_str=date_str)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first pipeline run expected rc=0 got={p1.returncode}")

        report_path = daily_base / f"{date_str}.json"
        idx_path = daily_base / "daily_reports_index.json"
        report_bak = daily_base / f"{date_str}.json.bak"
        idx_bak = daily_base / "daily_reports_index.json.bak"
        manifest_path = daily_base / "quant_packs" / date_str / "pipeline_manifest.json"

        if not report_path.exists() or not idx_path.exists() or not manifest_path.exists():
            return _fail("expected report/index/pipeline_manifest to exist")
        if not report_bak.exists() or not idx_bak.exists():
            return _fail("expected .bak files for report and index")

        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        if "quant_pack" not in report_obj or not isinstance(report_obj.get("quant_pack"), dict):
            return _fail("quant_pack missing after first pipeline run")

        idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
        rows = idx_obj.get("reports", [])
        by_date = {str(r.get("date")): r for r in rows if isinstance(r, dict)}
        today_row = by_date.get(date_str)
        if not isinstance(today_row, dict):
            return _fail("index missing today row")
        if not isinstance(today_row.get("quant"), dict):
            return _fail("index today row missing quant")

        pipeline_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        steps_ok = pipeline_obj.get("steps_ok", {})
        if not (steps_ok.get("build_pack") and steps_ok.get("embed") and steps_ok.get("update_index")):
            return _fail(f"steps_ok not all true: {steps_ok}")

        # second run for idempotency
        p2 = _run_pipeline(daily_base=daily_base, date_str=date_str)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second pipeline run expected rc=0 got={p2.returncode}")

        report_obj2 = json.loads(report_path.read_text(encoding="utf-8"))
        if "quant_pack" not in report_obj2 or not isinstance(report_obj2.get("quant_pack"), dict):
            return _fail("quant_pack missing after second pipeline run")

        idx_obj2 = json.loads(idx_path.read_text(encoding="utf-8"))
        rows2 = idx_obj2.get("reports", [])
        if len(rows2) != len(rows):
            return _fail("index row count changed unexpectedly on second run")

        print("[PASS] daily_quant_pipeline_minimal")
        print(f"[INFO] daily_base={daily_base}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
