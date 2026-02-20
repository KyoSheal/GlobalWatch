#!/usr/bin/env python3
"""T59: minimal regression for A4-12 attach backtest sweep to daily."""

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
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from price_store import compute_returns, save_prices, save_returns


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="backtest_sweep_attach_daily_min_"))
    try:
        outputs_base = (tmp / "outputs").resolve()
        daily_base = (outputs_base / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)
        date_str = "2026-02-18"
        report_path = daily_base / f"{date_str}.json"
        report_path.write_text(
            json.dumps({"date": date_str, "schema_version": 1, "summary": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        run_dir = outputs_base / "2026-02" / "20260218-1900-demo01"
        run_dir.mkdir(parents=True, exist_ok=True)
        snaps = run_dir / "portfolio_snapshots.jsonl"
        snap_rows = [
            {"time_utc": "2026-02-17T20:00:00+00:00", "target_weights": {"AAA": 0.7, "BBB": 0.2}},
            {"time_utc": "2026-02-18T20:00:00+00:00", "target_weights": {"AAA": 0.4, "BBB": 0.5}},
        ]
        with snaps.open("w", encoding="utf-8", newline="\n") as f:
            for row in snap_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        price_store = outputs_base / "price_store" / "cache_demo"
        prices = [
            {"date": "2026-02-17", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2026-02-18", "ticker": "AAA", "adj_close": 101.0},
            {"date": "2026-02-17", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-18", "ticker": "BBB", "adj_close": 49.5},
        ]
        save_prices(
            prices,
            price_store,
            source="csv",
            tickers=["AAA", "BBB"],
            request={"hash": "t59-demo", "start": "2026-02-17", "end": "2026-02-18"},
        )
        save_returns(compute_returns(prices), price_store)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a20_attach_backtest_sweep_to_daily.py"),
            "--daily-base",
            str(daily_base),
            "--date",
            date_str,
            "--run-dir",
            str(run_dir),
            "--price-store",
            str(price_store.parent),
            "--verbose",
        ]
        p1 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a20_attach first run failed rc={p1.returncode}")
        p2 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a20_attach second run failed rc={p2.returncode}")

        report_obj = json.load(open(report_path, "r", encoding="utf-8"))
        qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
        bsw = qp.get("backtest_sweep") if isinstance(qp.get("backtest_sweep"), dict) else {}
        if not bsw:
            return _fail("missing quant_pack.backtest_sweep")
        cbl = bsw.get("cost_bps_list") if isinstance(bsw.get("cost_bps_list"), list) else []
        if len(cbl) != 4:
            return _fail(f"expected cost_bps_list len=4, got {len(cbl)}")
        for key in ("break_even_cost_bps", "sensitivity_per_1bp", "status"):
            if key not in bsw:
                return _fail(f"missing summary key: {key}")

        sweep_report = daily_base / "quant_packs" / date_str / "backtest_sweep" / "sweep_report.md"
        if not sweep_report.exists():
            return _fail("missing sweep_report.md")
        txt = sweep_report.read_text(encoding="utf-8")
        if "- date_range: `` -> ``" in txt:
            return _fail("sweep_report has empty date_range")

        # index sync check
        p_idx = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "3650",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p_idx.returncode != 0:
            print(p_idx.stdout)
            print(p_idx.stderr)
            return _fail(f"a7 update index failed rc={p_idx.returncode}")
        idx_obj = json.load(open(daily_base / "daily_reports_index.json", "r", encoding="utf-8"))
        rows = idx_obj.get("reports") if isinstance(idx_obj.get("reports"), list) else []
        row = next((r for r in rows if isinstance(r, dict) and str(r.get("date")) == date_str), None)
        if not row:
            return _fail("index missing date row")
        q = row.get("quant") if isinstance(row.get("quant"), dict) else {}
        qbs = q.get("backtest_sweep") if isinstance(q.get("backtest_sweep"), dict) else {}
        if not qbs:
            return _fail("index missing quant.backtest_sweep")

        print("[PASS] backtest_sweep_attach_daily_minimal")
        print(f"[INFO] daily_json={report_path}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

