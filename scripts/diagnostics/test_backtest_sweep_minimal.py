#!/usr/bin/env python3
"""T58: minimal regression test for A4-11 backtest sweep."""

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


def _write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="backtest_sweep_min_"))
    try:
        price_store = tmp / "price_store" / "cache_demo"
        prices = [
            {"date": "2026-02-10", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2026-02-11", "ticker": "AAA", "adj_close": 101.0},
            {"date": "2026-02-12", "ticker": "AAA", "adj_close": 102.0},
            {"date": "2026-02-13", "ticker": "AAA", "adj_close": 103.0},
            {"date": "2026-02-14", "ticker": "AAA", "adj_close": 102.0},
            {"date": "2026-02-10", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-11", "ticker": "BBB", "adj_close": 49.0},
            {"date": "2026-02-12", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-13", "ticker": "BBB", "adj_close": 51.0},
            {"date": "2026-02-14", "ticker": "BBB", "adj_close": 52.0},
        ]
        save_prices(
            prices,
            price_store,
            source="csv",
            tickers=["AAA", "BBB"],
            request={"hash": "t58-demo", "start": "2026-02-10", "end": "2026-02-14", "tickers": ["AAA", "BBB"]},
        )
        save_returns(compute_returns(prices), price_store)

        weights_csv = tmp / "weights.csv"
        weights_rows = [
            {"date": "2026-02-10", "ticker": "AAA", "weight": 0.80},
            {"date": "2026-02-10", "ticker": "CASH", "weight": 0.20},
            {"date": "2026-02-12", "ticker": "BBB", "weight": 0.80},
            {"date": "2026-02-12", "ticker": "CASH", "weight": 0.20},
        ]
        _write_csv(weights_csv, ["date", "ticker", "weight"], weights_rows)

        out_dir = tmp / "sweep_out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a19_backtest_sweep.py"),
            "--weights-csv",
            str(weights_csv),
            "--price-store",
            str(price_store.parent),
            "--cost-bps-list",
            "0,5,20",
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a19_backtest_sweep failed rc={proc.returncode}")

        results_csv = out_dir / "sweep_results.csv"
        results_json = out_dir / "sweep_results.json"
        report_md = out_dir / "sweep_report.md"
        manifest_json = out_dir / "sweep_manifest.json"
        for p in (results_csv, results_json, report_md, manifest_json):
            if not p.exists():
                return _fail(f"missing output: {p}")

        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 3:
            return _fail(f"expected 3 scenario rows, got {len(rows)}")
        costs = [float(r.get("cost_bps", "nan")) for r in rows]
        if costs != sorted(costs):
            return _fail(f"cost_bps not sorted ascending: {costs}")

        eq0 = float(rows[0].get("end_equity", "0") or 0.0)
        eq2 = float(rows[-1].get("end_equity", "0") or 0.0)
        if eq2 > eq0 + 1e-6:
            return _fail(f"end_equity should not increase with higher cost: eq0={eq0} eq20={eq2}")

        md_text = report_md.read_text(encoding="utf-8")
        if "# Backtest Sweep Report" not in md_text:
            return _fail("sweep_report missing expected heading")

        js = json.load(open(results_json, "r", encoding="utf-8"))
        if not isinstance(js.get("rows"), list):
            return _fail("sweep_results.json missing rows list")

        print("[PASS] backtest_sweep_minimal")
        print(f"[INFO] out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

