#!/usr/bin/env python3
"""T36: minimal regression test for A1-5 quant gate."""

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


def _mk_dataset(path: Path, run_id: str, sharpe: float, max_dd: float, total_return: float, trades_total: int, gating_ratio: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "manifest.json", {"run_id": run_id})
    _write_json(
        path / "metrics" / "metrics.json",
        {
            "meta": {"run_id": run_id},
            "performance": {"total_return": total_return, "cagr": total_return},
            "risk": {"sharpe": sharpe, "max_drawdown": max_dd, "calmar": 0.5, "vol_annualized": 0.2},
            "trading": {"trades_total": trades_total, "turnover_ratio": 0.2, "buys": trades_total // 2, "sells": trades_total // 2, "unique_tickers": 3},
            "data_quality": {"insufficient_points": False, "missing_files": [], "parse_warnings": {"x": 0}},
            "gating": {"summary": {"counts": {"attempt_cooldown": 1}, "top3": [{"reason": "attempt_cooldown", "count": 1}]}},
        },
    )
    d0 = datetime(2026, 2, 10, tzinfo=timezone.utc)
    daily_rows = []
    close = 100.0
    for i in range(6):
        r = 0.01 if i % 2 == 0 else -0.005
        close *= (1.0 + r)
        daily_rows.append({"date_local": (d0 + timedelta(days=i)).date().isoformat(), "close_equity": f"{close:.6f}", "daily_return": f"{r:.6f}"})
    _write_csv(path / "metrics" / "daily_returns.csv", ["date_local", "close_equity", "daily_return"], daily_rows)

    total_cycles = 10
    gated_cycles = int(round(gating_ratio * total_cycles))
    cycles = []
    for i in range(total_cycles):
        reason = "attempt_cooldown" if i < gated_cycles else ""
        cycles.append(
            {
                "cycle_id": i + 1,
                "time_utc": (d0 + timedelta(hours=i)).isoformat(),
                "skip_reason": reason,
                "decision_path": "",
                "cov_gate_reason": "",
            }
        )
    _write_csv(path / "cycles.csv", ["cycle_id", "time_utc", "skip_reason", "decision_path", "cov_gate_reason"], cycles)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_gate_min_"))
    try:
        baseline = tmp_root / "baseline" / "run_dataset"
        cand_good = tmp_root / "cand_good" / "run_dataset"
        cand_bad = tmp_root / "cand_bad" / "run_dataset"
        out_dir = tmp_root / "gate_out"

        _mk_dataset(baseline, "baseline", sharpe=1.0, max_dd=-0.10, total_return=0.10, trades_total=10, gating_ratio=0.2)
        _mk_dataset(cand_good, "cand_good", sharpe=0.9, max_dd=-0.12, total_return=0.09, trades_total=11, gating_ratio=0.3)
        _mk_dataset(cand_bad, "cand_bad", sharpe=0.4, max_dd=-0.25, total_return=0.01, trades_total=30, gating_ratio=0.8)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a5_quant_gate.py"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(cand_good),
            "--candidate",
            str(cand_bad),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 3:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"expected exit code 3 when one candidate fails, got {proc.returncode}")

        summary_path = out_dir / "gate_summary.json"
        if not summary_path.exists():
            return _fail("missing gate_summary.json")
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        items = summary.get("candidates", []) or []
        status_map = {Path(i.get("candidate_dir", "")).parts[-2] if i.get("candidate_dir") else "": i.get("status") for i in items}

        # find by suffix to avoid path separator differences
        statuses = {str(i.get("candidate_dir", "")): str(i.get("status", "")) for i in items}
        good_ok = any(k.endswith(str(cand_good)) and v == "PASS" for k, v in statuses.items())
        bad_fail = any(k.endswith(str(cand_bad)) and v == "FAIL" for k, v in statuses.items())
        if not good_ok:
            return _fail("cand_good expected PASS")
        if not bad_fail:
            return _fail("cand_bad expected FAIL")

        bad_report = out_dir / f"02_{cand_bad.name}" / "gate_report.md"
        if not bad_report.exists():
            return _fail("missing bad candidate gate_report.md")
        report_text = bad_report.read_text(encoding="utf-8")
        if "sharpe_drop_max" not in report_text:
            return _fail("gate_report.md missing sharpe_drop_max fail rule")

        print("[PASS] quant_gate_minimal")
        print(f"[INFO] summary_candidates={len(items)} out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

