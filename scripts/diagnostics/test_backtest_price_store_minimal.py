#!/usr/bin/env python3
"""T47: minimal regression test for A4-1 backtest price store/cache."""

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

from price_store import compute_returns, load_prices


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
    tmp = Path(tempfile.mkdtemp(prefix="backtest_price_store_min_"))
    try:
        cache_base = tmp / "outputs" / "backtest_cache"
        source_csv = tmp / "source_prices.csv"

        rows = [
            {"date": "2026-02-10", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2026-02-11", "ticker": "AAA", "adj_close": 101.0},
            {"date": "2026-02-12", "ticker": "AAA", "adj_close": 99.0},
            {"date": "2026-02-13", "ticker": "AAA", "adj_close": 102.0},
            {"date": "2026-02-14", "ticker": "AAA", "adj_close": 103.0},
            {"date": "2026-02-10", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-11", "ticker": "BBB", "adj_close": 50.5},
            {"date": "2026-02-12", "ticker": "BBB", "adj_close": 51.0},
            {"date": "2026-02-13", "ticker": "BBB", "adj_close": 49.0},
            {"date": "2026-02-14", "ticker": "BBB", "adj_close": 49.5},
        ]
        _write_csv(source_csv, ["date", "ticker", "adj_close"], rows)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a13_prepare_backtest_prices.py"),
            "--tickers",
            "AAA,BBB",
            "--start",
            "2026-02-10",
            "--end",
            "2026-02-14",
            "--cache-base",
            str(cache_base),
            "--source",
            "csv",
            "--csv-path",
            str(source_csv),
            "--verbose",
        ]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            return _fail(f"a13 failed rc={p.returncode}")

        cache_dirs = [x for x in cache_base.iterdir() if x.is_dir()]
        if len(cache_dirs) != 1:
            return _fail(f"expected 1 cache dir, got {len(cache_dirs)}")
        cache_dir = cache_dirs[0]

        prices_path = cache_dir / "prices_daily.csv"
        returns_path = cache_dir / "returns_daily.csv"
        manifest_path = cache_dir / "manifest.json"
        for pth in (prices_path, returns_path, manifest_path):
            if not pth.exists():
                return _fail(f"missing expected output: {pth}")

        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        for key in ("schema_version", "generated_utc", "source", "rows", "tickers", "date_range", "validation", "request", "returns"):
            if key not in manifest:
                return _fail(f"manifest missing key: {key}")

        loaded_prices = load_prices(cache_dir)
        if len(loaded_prices) != 10:
            return _fail(f"expected 10 prices rows, got {len(loaded_prices)}")

        computed_rets = compute_returns(loaded_prices)
        if len(computed_rets) != 8:
            return _fail(f"expected 8 return rows, got {len(computed_rets)}")

        # quick schema checks
        with returns_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            if cols != ["date", "ticker", "ret"]:
                return _fail(f"returns columns mismatch: {cols}")
            rows_out = list(reader)
        if len(rows_out) != 8:
            return _fail(f"returns csv row count mismatch: {len(rows_out)}")

        print("[PASS] backtest_price_store_minimal")
        print(f"[INFO] cache_dir={cache_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
