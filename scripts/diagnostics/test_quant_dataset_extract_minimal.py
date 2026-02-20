#!/usr/bin/env python3
"""T32: minimal regression test for A1 run dataset extractor."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_extract_min_"))
    try:
        outputs_dir = tmp_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        base_ts = datetime(2026, 2, 18, 19, 49, 1, tzinfo=timezone.utc)
        snapshots = []
        for i in range(3):
            ts = (base_ts.replace(minute=49 + i)).isoformat()
            snapshots.append(
                {
                    "cycle_id": i + 1,
                    "time_utc": ts,
                    "cash": 30000.0 - (i * 100.0),
                    "positions_value": i * 100.0,
                    "total_equity": 30000.0,
                    "regime_state": "risk_off" if i == 0 else "neutral",
                    "cash_target": 0.2,
                    "skip_reason": "",
                }
            )
        trades = [
            {
                "time_utc": base_ts.isoformat(),
                "cycle_id": 1,
                "ticker": "TLT",
                "side": "BUY",
                "quantity": 10,
                "price": 90.0,
                "notional": 900.0,
            },
            {
                "time_utc": base_ts.replace(minute=50).isoformat(),
                "cycle_id": 2,
                "ticker": "DE",
                "side": "SELL",
                "quantity": 2,
                "price": 400.0,
                "notional": 800.0,
            },
        ]

        _write_jsonl(outputs_dir / "portfolio_snapshots.jsonl", snapshots)
        _write_jsonl(outputs_dir / "trade_history.jsonl", trades)
        _write_json(
            outputs_dir / "snapshot_live.json",
            {
                "run_id": "test-run-001",
                "trade_history_path": str((outputs_dir / "trade_history.jsonl").resolve()),
            },
        )

        out_dir = tmp_root / "out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a1_extract_run_dataset.py"),
            "--base-out-dir",
            str(outputs_dir),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"extractor failed rc={proc.returncode}")

        manifest = out_dir / "manifest.json"
        equity_csv = out_dir / "equity_curve.csv"
        cycles_csv = out_dir / "cycles.csv"
        trades_csv = out_dir / "trades.csv"
        for p in (manifest, equity_csv, cycles_csv, trades_csv):
            if not p.exists():
                return _fail(f"missing output file: {p}")

        with manifest.open("r", encoding="utf-8") as f:
            manifest_obj = json.load(f)
        if str(manifest_obj.get("run_id")) != "test-run-001":
            return _fail(f"manifest run_id mismatch: {manifest_obj.get('run_id')!r}")

        cycles_rows, cycles_cols = _read_csv_rows(cycles_csv)
        if len(cycles_rows) < 3:
            return _fail(f"expected >=3 cycle rows, got {len(cycles_rows)}")
        for col in ("cycle_id", "time_utc", "total_equity"):
            if col not in cycles_cols:
                return _fail(f"cycles.csv missing column {col}")

        trades_rows, _ = _read_csv_rows(trades_csv)
        if len(trades_rows) < 2:
            return _fail(f"expected >=2 trade rows, got {len(trades_rows)}")

        for row in cycles_rows:
            if "+00:00" not in str(row.get("time_utc", "")):
                return _fail(f"cycle row time_utc not UTC ISO offset: {row.get('time_utc')!r}")
        for row in trades_rows:
            if "+00:00" not in str(row.get("time_utc", "")):
                return _fail(f"trade row time_utc not UTC ISO offset: {row.get('time_utc')!r}")

        print("[PASS] quant_dataset_extract_minimal")
        print(
            "[INFO] "
            f"cycles={len(cycles_rows)} trades={len(trades_rows)} out_dir={out_dir}"
        )
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

