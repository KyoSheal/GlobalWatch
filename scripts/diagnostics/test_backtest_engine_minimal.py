#!/usr/bin/env python3
"""T48: minimal regression test for A4-2 backtest engine core."""

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
    tmp = Path(tempfile.mkdtemp(prefix="backtest_engine_min_"))
    try:
        cache_dir = tmp / "cache"
        returns_path = cache_dir / "returns_daily.csv"
        weights_path = tmp / "weights.csv"

        rets = [
            # AAA
            {"date": "2026-02-10", "ticker": "AAA", "ret": 0.01},
            {"date": "2026-02-11", "ticker": "AAA", "ret": 0.02},
            {"date": "2026-02-12", "ticker": "AAA", "ret": 0.00},
            {"date": "2026-02-13", "ticker": "AAA", "ret": 0.00},
            {"date": "2026-02-14", "ticker": "AAA", "ret": -0.01},
            # BBB
            {"date": "2026-02-10", "ticker": "BBB", "ret": 0.00},
            {"date": "2026-02-11", "ticker": "BBB", "ret": -0.01},
            {"date": "2026-02-12", "ticker": "BBB", "ret": 0.03},
            {"date": "2026-02-13", "ticker": "BBB", "ret": 0.01},
            {"date": "2026-02-14", "ticker": "BBB", "ret": 0.02},
        ]
        _write_csv(returns_path, ["date", "ticker", "ret"], rets)
        json.dump(
            {"request": {"hash": "demo_hash_for_t48"}, "schema_version": 1},
            open(cache_dir / "manifest.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

        weights = [
            {"date": "2026-02-10", "ticker": "AAA", "weight": 1.0},
            {"date": "2026-02-12", "ticker": "BBB", "weight": 1.0},
        ]
        _write_csv(weights_path, ["date", "ticker", "weight"], weights)

        out_dir = tmp / "bt_out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a14_run_backtest.py"),
            "--cache-dir",
            str(cache_dir),
            "--weights",
            str(weights_path),
            "--out-dir",
            str(out_dir),
            "--initial-equity",
            "100000",
            "--cost-bps",
            "0",
            "--rebalance",
            "daily",
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a14 failed rc={proc.returncode}")

        eq_path = out_dir / "backtest_equity.csv"
        tr_path = out_dir / "backtest_trades.csv"
        mf_path = out_dir / "backtest_manifest.json"
        md_path = out_dir / "backtest_report.md"
        for p in (eq_path, tr_path, mf_path, md_path):
            if not p.exists():
                return _fail(f"missing output {p}")

        # validate equity result
        with eq_path.open("r", encoding="utf-8", newline="") as f:
            eq_rows = list(csv.DictReader(f))
        if len(eq_rows) != 5:
            return _fail(f"expected 5 equity rows, got {len(eq_rows)}")
        final_equity = float(eq_rows[-1]["equity"])
        expected_final = 109315.14012
        if abs(final_equity - expected_final) > 1e-4:
            return _fail(f"final equity mismatch: got={final_equity} expected={expected_final}")

        with tr_path.open("r", encoding="utf-8", newline="") as f:
            tr_rows = list(csv.DictReader(f))
        if len(tr_rows) < 3:
            return _fail(f"expected >=3 trade rows, got {len(tr_rows)}")
        trade_dates = sorted(set([r.get("date", "") for r in tr_rows]))
        if "2026-02-10" not in trade_dates or "2026-02-12" not in trade_dates:
            return _fail(f"missing rebalance trade dates, got={trade_dates}")

        manifest = json.load(open(mf_path, "r", encoding="utf-8"))
        for key in ("schema_version", "generated_utc", "inputs", "params", "cost_summary", "warnings"):
            if key not in manifest:
                return _fail(f"manifest missing key: {key}")

        print("[PASS] backtest_engine_minimal")
        print(f"[INFO] out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
