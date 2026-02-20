#!/usr/bin/env python3
"""T35: minimal regression test for A1-4 leaderboard builder."""

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


def _write_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _mk_dataset(base_dir: Path, date_str: str, run_id: str, metrics_obj: dict, cycles_rows: list[dict]) -> Path:
    ds = base_dir / date_str / "run_dataset"
    ds.mkdir(parents=True, exist_ok=True)
    _write_json(ds / "manifest.json", {"run_id": run_id})
    _write_json(ds / "metrics" / "metrics.json", metrics_obj)
    # minimal daily returns (6 points)
    d0 = datetime(2026, 2, 10, tzinfo=timezone.utc)
    daily_rows = []
    close = 100.0
    for i in range(6):
        close = close * (1.0 + (0.01 if i % 2 == 0 else -0.005))
        daily_rows.append(
            {
                "date_local": (d0 + timedelta(days=i)).date().isoformat(),
                "close_equity": f"{close:.6f}",
                "daily_return": "0.01" if i % 2 == 0 else "-0.005",
            }
        )
    _write_csv(ds / "metrics" / "daily_returns.csv", ["date_local", "close_equity", "daily_return"], daily_rows)
    _write_csv(ds / "cycles.csv", ["cycle_id", "time_utc", "skip_reason", "decision_path", "cov_gate_reason"], cycles_rows)
    return ds


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_leaderboard_min_"))
    try:
        base_dir = tmp_root / "Daily Report"
        out_dir = tmp_root / "leaderboard_out"

        # A
        _mk_dataset(
            base_dir,
            "2026-02-18",
            "run-A",
            {
                "meta": {"run_id": "run-A"},
                "performance": {"total_return": 0.05, "cagr": 0.15},
                "risk": {"vol_annualized": 0.20, "sharpe": 0.5, "sortino": 0.7, "max_drawdown": -0.20, "calmar": 0.75},
                "trading": {"trades_total": 10, "turnover_ratio": 0.10, "buys": 6, "sells": 4, "unique_tickers": 5},
                "data_quality": {"insufficient_points": False, "missing_files": [], "parse_warnings": {"a": 0}},
                "gating": {"summary": {"counts": {"attempt_cooldown": 2}, "top3": [{"reason": "attempt_cooldown", "count": 2}]}},
            },
            [
                {"cycle_id": 1, "time_utc": "2026-02-18T15:00:00+00:00", "skip_reason": "attempt_cooldown", "decision_path": "", "cov_gate_reason": ""},
                {"cycle_id": 2, "time_utc": "2026-02-18T16:00:00+00:00", "skip_reason": "", "decision_path": "", "cov_gate_reason": ""},
            ],
        )
        # B (expected top)
        _mk_dataset(
            base_dir,
            "2026-02-19",
            "run-B",
            {
                "meta": {"run_id": "run-B"},
                "performance": {"total_return": 0.03, "cagr": 0.12},
                "risk": {"vol_annualized": 0.18, "sharpe": 1.2, "sortino": 1.5, "max_drawdown": -0.10, "calmar": 1.20},
                "trading": {"trades_total": 8, "turnover_ratio": 0.15, "buys": 4, "sells": 4, "unique_tickers": 4},
                "data_quality": {"insufficient_points": False, "missing_files": [], "parse_warnings": {"a": 0}},
                "gating": {"summary": {"counts": {"market_closed": 1}, "top3": [{"reason": "market_closed", "count": 1}]}},
            },
            [
                {"cycle_id": 1, "time_utc": "2026-02-19T15:00:00+00:00", "skip_reason": "market_closed", "decision_path": "", "cov_gate_reason": ""},
            ],
        )
        # C
        _mk_dataset(
            base_dir,
            "2026-02-20",
            "run-C",
            {
                "meta": {"run_id": "run-C"},
                "performance": {"total_return": -0.02, "cagr": -0.08},
                "risk": {"vol_annualized": 0.25, "sharpe": -0.2, "sortino": -0.1, "max_drawdown": -0.35, "calmar": -0.2},
                "trading": {"trades_total": 5, "turnover_ratio": 0.05, "buys": 2, "sells": 3, "unique_tickers": 3},
                "data_quality": {"insufficient_points": False, "missing_files": [], "parse_warnings": {"a": 1}},
                "gating": {"summary": {"counts": {"risk_gate_abort": 3}, "top3": [{"reason": "risk_gate_abort", "count": 3}]}},
            },
            [
                {"cycle_id": 1, "time_utc": "2026-02-20T15:00:00+00:00", "skip_reason": "risk_gate_abort", "decision_path": "", "cov_gate_reason": ""},
            ],
        )

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a4_build_leaderboard.py"),
            "--base-dir",
            str(base_dir),
            "--out-dir",
            str(out_dir),
            "--sort-by",
            "composite",
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a4_build_leaderboard failed rc={proc.returncode}")

        leaderboard_csv = out_dir / "leaderboard.csv"
        leaderboard_json = out_dir / "leaderboard.json"
        leaderboard_md = out_dir / "leaderboard.md"
        gating_csv = out_dir / "gating_summary.csv"
        manifest_lb = out_dir / "manifest_leaderboard.json"
        for p in (leaderboard_csv, leaderboard_json, leaderboard_md, gating_csv, manifest_lb):
            if not p.exists():
                return _fail(f"missing output file: {p}")

        with leaderboard_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            cols = list(reader.fieldnames or [])
        required_cols = {"run_id", "total_return", "sharpe", "max_drawdown", "turnover_ratio", "composite_score", "rank"}
        if not required_cols.issubset(set(cols)):
            return _fail("leaderboard.csv missing required columns")
        if not rows:
            return _fail("leaderboard.csv has no rows")
        top_run = rows[0].get("run_id", "")
        if top_run != "run-B":
            return _fail(f"expected top run run-B, got {top_run!r}")

        with gating_csv.open("r", encoding="utf-8", newline="") as f:
            g_rows = list(csv.DictReader(f))
        reasons = {str(r.get("reason", "")).strip().lower() for r in g_rows}
        if "attempt_cooldown" not in reasons:
            return _fail("gating_summary.csv missing attempt_cooldown")

        md = leaderboard_md.read_text(encoding="utf-8")
        for needle in ("Top 10", "Bottom 5"):
            if needle not in md:
                return _fail(f"leaderboard.md missing section {needle!r}")

        print("[PASS] quant_leaderboard_minimal")
        print(f"[INFO] top_run={top_run} rows={len(rows)} out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

