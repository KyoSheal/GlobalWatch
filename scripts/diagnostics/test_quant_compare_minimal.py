#!/usr/bin/env python3
"""T34: minimal regression test for A1-3 run-to-run compare."""

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
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _make_dataset(path: Path, *, run_id: str, equities: list[float], trades_count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 2, 10, 14, 30, tzinfo=timezone.utc)
    eq_rows = []
    for i, e in enumerate(equities):
        ts = (t0 + timedelta(days=i)).isoformat()
        eq_rows.append({"time_utc": ts, "equity": f"{e:.4f}", "cash": "0", "positions_value": f"{e:.4f}"})

    _write_json(path / "manifest.json", {"run_id": run_id})
    _write_csv(path / "equity_curve.csv", ["time_utc", "equity", "cash", "positions_value"], eq_rows)
    _write_csv(
        path / "cycles.csv",
        ["cycle_id", "time_utc", "skip_reason", "decision_path", "cov_gate_reason"],
        [
            {"cycle_id": "1", "time_utc": eq_rows[0]["time_utc"], "skip_reason": "attempt_cooldown", "decision_path": "", "cov_gate_reason": ""},
            {"cycle_id": "2", "time_utc": eq_rows[1]["time_utc"], "skip_reason": "", "decision_path": "", "cov_gate_reason": ""},
            {"cycle_id": "3", "time_utc": eq_rows[2]["time_utc"], "skip_reason": "market_closed", "decision_path": "", "cov_gate_reason": ""},
        ],
    )

    trade_rows = []
    for i in range(trades_count):
        ts = (t0 + timedelta(days=i)).isoformat()
        trade_rows.append(
            {
                "time_utc": ts,
                "cycle_id": str(i + 1),
                "ticker": f"T{i%2+1}",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "qty": "1",
                "price": "100",
                "notional": "100",
                "is_forced": "False",
                "force_reason": "",
                "status": "ok",
                "reason": "",
            }
        )
    _write_csv(
        path / "trades.csv",
        ["time_utc", "cycle_id", "ticker", "side", "qty", "price", "notional", "is_forced", "force_reason", "status", "reason"],
        trade_rows,
    )


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_compare_min_"))
    try:
        ds_a = tmp_root / "dataset_a"
        ds_b = tmp_root / "dataset_b"
        out_dir = tmp_root / "compare_out"

        # A: weaker return, deeper DD
        _make_dataset(ds_a, run_id="run-a", equities=[100, 95, 90, 98, 102, 105], trades_count=5)
        # B: stronger return, milder DD
        _make_dataset(ds_b, run_id="run-b", equities=[100, 99, 101, 108, 112, 118], trades_count=4)

        for ds in (ds_a, ds_b):
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
                "--dataset-dir",
                str(ds),
            ]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                return _fail(f"a2 failed for {ds} rc={proc.returncode}")

        cmd_compare = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a3_compare_runs.py"),
            "--dataset-a",
            str(ds_a),
            "--dataset-b",
            str(ds_b),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        proc_c = subprocess.run(cmd_compare, cwd=str(ROOT), capture_output=True, text=True)
        if proc_c.returncode != 0:
            print(proc_c.stdout)
            print(proc_c.stderr)
            return _fail(f"a3 compare failed rc={proc_c.returncode}")

        compare_json = out_dir / "compare.json"
        compare_md = out_dir / "compare.md"
        delta_csv = out_dir / "delta_daily_returns.csv"
        for p in (compare_json, compare_md, delta_csv):
            if not p.exists():
                return _fail(f"missing output file: {p}")

        with compare_json.open("r", encoding="utf-8") as f:
            cmp_obj = json.load(f)
        winner = str((cmp_obj.get("headline") or {}).get("winner", ""))
        if winner != "B":
            return _fail(f"expected winner B, got {winner!r}")
        overlap_days = int(((cmp_obj.get("daily_returns_compare") or {}).get("overlap_days", 0) or 0))
        if overlap_days <= 0:
            return _fail(f"expected overlap_days > 0, got {overlap_days}")

        md_text = compare_md.read_text(encoding="utf-8")
        for needle in ("Sharpe", "Max Drawdown", "Trades"):
            if needle not in md_text:
                return _fail(f"compare.md missing keyword: {needle}")

        print("[PASS] quant_compare_minimal")
        print(f"[INFO] winner={winner} overlap_days={overlap_days} out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

