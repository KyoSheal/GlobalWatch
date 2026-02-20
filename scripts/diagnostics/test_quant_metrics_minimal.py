#!/usr/bin/env python3
"""T33: minimal regression test for A1-2 quant metrics engine."""

from __future__ import annotations

import csv
import json
import math
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
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_metrics_min_"))
    try:
        dataset_dir = tmp_root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        t0 = datetime(2026, 2, 10, 14, 30, tzinfo=timezone.utc)
        equities = [100.0, 110.0, 105.0, 120.0, 90.0, 130.0]
        eq_rows = []
        for i, e in enumerate(equities):
            ts = (t0 + timedelta(hours=24 * i)).isoformat()
            eq_rows.append(
                {
                    "time_utc": ts,
                    "equity": f"{e:.2f}",
                    "cash": "0.0",
                    "positions_value": f"{e:.2f}",
                }
            )

        _write_json(dataset_dir / "manifest.json", {"run_id": "demo"})
        _write_csv(dataset_dir / "equity_curve.csv", ["time_utc", "equity", "cash", "positions_value"], eq_rows)
        _write_csv(
            dataset_dir / "cycles.csv",
            ["cycle_id", "time_utc", "skip_reason", "decision_path", "cov_gate_reason"],
            [
                {"cycle_id": "1", "time_utc": eq_rows[0]["time_utc"], "skip_reason": "", "decision_path": "", "cov_gate_reason": ""},
                {"cycle_id": "2", "time_utc": eq_rows[1]["time_utc"], "skip_reason": "attempt_cooldown", "decision_path": "", "cov_gate_reason": ""},
                {"cycle_id": "3", "time_utc": eq_rows[2]["time_utc"], "skip_reason": "portfolio_cov_rc_limit", "decision_path": "", "cov_gate_reason": ""},
            ],
        )
        _write_csv(
            dataset_dir / "trades.csv",
            ["time_utc", "cycle_id", "ticker", "side", "qty", "price", "notional", "is_forced", "force_reason", "status", "reason"],
            [
                {"time_utc": eq_rows[1]["time_utc"], "cycle_id": "2", "ticker": "AAA", "side": "BUY", "qty": "1", "price": "100", "notional": "100", "is_forced": "False", "force_reason": "", "status": "ok", "reason": ""},
                {"time_utc": eq_rows[2]["time_utc"], "cycle_id": "3", "ticker": "BBB", "side": "BUY", "qty": "2", "price": "100", "notional": "200", "is_forced": "False", "force_reason": "", "status": "ok", "reason": ""},
                {"time_utc": eq_rows[3]["time_utc"], "cycle_id": "4", "ticker": "AAA", "side": "SELL", "qty": "1", "price": "120", "notional": "120", "is_forced": "False", "force_reason": "", "status": "ok", "reason": ""},
            ],
        )

        out_dir = tmp_root / "metrics_out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a2_compute_metrics failed rc={proc.returncode}")

        metrics_json = out_dir / "metrics.json"
        metrics_md = out_dir / "metrics.md"
        daily_returns = out_dir / "daily_returns.csv"
        for p in (metrics_json, metrics_md, daily_returns):
            if not p.exists():
                return _fail(f"missing output: {p}")

        with metrics_json.open("r", encoding="utf-8") as f:
            m = json.load(f)

        total_return = float(m["performance"]["total_return"])
        expected_total_return = 0.30
        if abs(total_return - expected_total_return) > 1e-9:
            return _fail(f"total_return mismatch got={total_return} expected={expected_total_return}")

        max_dd = float(m["risk"]["max_drawdown"])
        if not (-0.251 <= max_dd <= -0.249):
            return _fail(f"max_drawdown out of expected range: {max_dd}")

        trades_total = int(m["trading"]["trades_total"])
        if trades_total != 3:
            return _fail(f"trades_total mismatch: {trades_total}")
        unique_tickers = int(m["trading"]["unique_tickers"])
        if unique_tickers != 2:
            return _fail(f"unique_tickers mismatch: {unique_tickers}")

        md_text = metrics_md.read_text(encoding="utf-8")
        for needle in ("Total Return", "Max DD", "Sharpe", "Trades"):
            if needle not in md_text:
                return _fail(f"metrics.md missing keyword: {needle}")

        print("[PASS] quant_metrics_minimal")
        print(
            "[INFO] "
            f"total_return={total_return:.6f} max_dd={max_dd:.6f} "
            f"trades_total={trades_total} unique_tickers={unique_tickers}"
        )
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

