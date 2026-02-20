#!/usr/bin/env python3
"""T50: minimal regression test for A4-4 run_dir offline backtest pipeline."""

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
    tmp = Path(tempfile.mkdtemp(prefix="backtest_from_run_min_"))
    try:
        run_dir = tmp / "outputs" / "2026-02" / "20260219-1200-abcd12"
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshots = run_dir / "portfolio_snapshots.jsonl"
        snap_rows = [
            {"time_utc": "2026-02-18T15:00:00+00:00", "target_weights": {"AAA": 0.6, "BBB": 0.2}},
            {"time_utc": "2026-02-18T20:00:00+00:00", "target_weights": {"AAA": 0.7, "BBB": 0.1}},
            {"time_utc": "2026-02-19T20:00:00+00:00", "target_weights": {"AAA": 0.2, "BBB": 0.7}},
        ]
        with snapshots.open("w", encoding="utf-8", newline="\n") as f:
            for row in snap_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        price_store_root = tmp / "outputs" / "price_store"
        cache_dir = price_store_root / "cache_demo"
        prices = [
            {"date": "2026-02-17", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2026-02-18", "ticker": "AAA", "adj_close": 101.0},
            {"date": "2026-02-19", "ticker": "AAA", "adj_close": 103.0},
            {"date": "2026-02-17", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-18", "ticker": "BBB", "adj_close": 49.0},
            {"date": "2026-02-19", "ticker": "BBB", "adj_close": 49.5},
        ]
        save_prices(
            prices,
            cache_dir,
            source="csv",
            tickers=["AAA", "BBB"],
            request={"hash": "t50-demo", "start": "2026-02-17", "end": "2026-02-19", "tickers": ["AAA", "BBB"]},
        )
        save_returns(compute_returns(prices), cache_dir)

        out_dir = tmp / "bt_from_run_out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a16_run_backtest_from_run.py"),
            "--run-dir",
            str(run_dir),
            "--price-store",
            str(price_store_root),
            "--out-dir",
            str(out_dir),
            "--report-tz",
            "America/New_York",
            "--cost-bps",
            "5",
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a16 failed rc={proc.returncode}")

        report_path = out_dir / "backtest" / "backtest_report.md"
        equity_path = out_dir / "backtest" / "backtest_equity.csv"
        manifest_path = out_dir / "backtest_from_run_manifest.json"
        for p in (report_path, equity_path, manifest_path):
            if not p.exists():
                return _fail(f"missing output {p}")

        report_text = report_path.read_text(encoding="utf-8")
        for token in ("total_return", "max_drawdown", "days"):
            if token not in report_text:
                return _fail(f"report missing token {token}")

        with equity_path.open("r", encoding="utf-8", newline="") as f:
            eq_rows = list(csv.DictReader(f))
        if len(eq_rows) != 2:
            return _fail(f"expected 2 equity rows, got {len(eq_rows)}")

        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        for key in (
            "schema_version",
            "generated_utc",
            "run_dir",
            "weights_path",
            "price_store_path",
            "backtest_out_dir",
            "date_range",
            "tickers_count",
            "warnings",
            "hash",
        ):
            if key not in manifest:
                return _fail(f"manifest missing key: {key}")

        print("[PASS] backtest_from_run_minimal")
        print(f"[INFO] out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

