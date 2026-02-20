#!/usr/bin/env python3
"""T37: minimal regression test for A1-6 daily quant pack."""

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


def _mk_dataset(path: Path, run_id: str, equities: list[float], skip_reasons: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "manifest.json", {"run_id": run_id})

    t0 = datetime(2026, 2, 18, 14, 30, tzinfo=timezone.utc)
    eq_rows = []
    cycles = []
    trades = []
    for i, eq in enumerate(equities):
        ts = (t0 + timedelta(hours=i)).isoformat()
        eq_rows.append({"time_utc": ts, "equity": f"{eq:.6f}", "cash": "1000", "positions_value": f"{eq - 1000:.6f}"})
        reason = skip_reasons[i] if i < len(skip_reasons) else ""
        cycles.append(
            {
                "cycle_id": str(i + 1),
                "time_utc": ts,
                "session_state": "open",
                "regime_state": "neutral",
                "cash_target": "0.1",
                "total_equity": f"{eq:.6f}",
                "cash": "1000",
                "positions_value": f"{eq - 1000:.6f}",
                "skip_reason": reason,
                "decision_path": "",
                "cov_gate_reason": "",
                "cov_gate_max_rc": "",
                "rc_limit": "",
                "turnover_used_total": "",
            }
        )
        if i % 2 == 0:
            trades.append(
                {
                    "time_utc": ts,
                    "cycle_id": str(i + 1),
                    "ticker": "QQQ" if i % 4 == 0 else "SPY",
                    "side": "BUY" if i % 4 == 0 else "SELL",
                    "qty": "1",
                    "price": "100",
                    "notional": "100",
                    "is_forced": "False",
                    "force_reason": "",
                    "status": "ok",
                    "reason": "",
                }
            )

    _write_csv(path / "equity_curve.csv", ["time_utc", "equity", "cash", "positions_value"], eq_rows)
    _write_csv(
        path / "cycles.csv",
        [
            "cycle_id",
            "time_utc",
            "session_state",
            "regime_state",
            "cash_target",
            "total_equity",
            "cash",
            "positions_value",
            "skip_reason",
            "decision_path",
            "cov_gate_reason",
            "cov_gate_max_rc",
            "rc_limit",
            "turnover_used_total",
        ],
        cycles,
    )
    _write_csv(
        path / "trades.csv",
        ["time_utc", "cycle_id", "ticker", "side", "qty", "price", "notional", "is_forced", "force_reason", "status", "reason"],
        trades,
    )


def _run_a6(*, daily_dir: Path, base_dir: Path, out_dir: Path, strict: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a6_build_daily_quant_pack.py"),
        "--daily-dir",
        str(daily_dir),
        "--base-dir",
        str(base_dir),
        "--out-dir",
        str(out_dir),
        "--auto-metrics",
        "--auto-gate",
        "--auto-leaderboard",
        "--lookback-days",
        "14",
        "--verbose",
    ]
    if strict:
        cmd.append("--strict")
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def _run_a6_flat(*, daily_base: Path, date_str: str, strict: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a6_build_daily_quant_pack.py"),
        "--daily-base",
        str(daily_base),
        "--date",
        str(date_str),
        "--auto-metrics",
        "--auto-gate",
        "--auto-leaderboard",
        "--lookback-days",
        "14",
        "--verbose",
    ]
    if strict:
        cmd.append("--strict")
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_daily_pack_min_"))
    try:
        base_dir = tmp_root / "Daily Report"
        prev_day = base_dir / "2026-02-18"
        today = base_dir / "2026-02-19"
        lonely = base_dir / "2026-02-25"

        _mk_dataset(prev_day / "run_dataset", "run-prev", [100000, 100200, 100300, 100250, 100400, 100450], ["", "attempt_cooldown", "", "", "", ""])
        _mk_dataset(today / "run_dataset", "run-today", [100000, 99900, 100050, 100020, 100200, 100150], ["market_closed", "", "", "attempt_cooldown", "", ""])
        _mk_dataset(lonely / "run_dataset", "run-lonely", [100000, 100050, 100070, 100060, 100100, 100120], ["", "", "", "", "", ""])

        # Case 1: baseline exists via prev_day
        out_ok = today / "quant"
        proc_ok = _run_a6(daily_dir=today, base_dir=base_dir, out_dir=out_ok, strict=False)
        if proc_ok.returncode != 0:
            print(proc_ok.stdout)
            print(proc_ok.stderr)
            return _fail(f"a6 non-strict expected rc=0, got rc={proc_ok.returncode}")
        report_path = out_ok / "daily_quant_report.md"
        manifest_path = out_ok / "pack_manifest.json"
        if not report_path.exists() or not manifest_path.exists():
            return _fail("missing daily_quant_report.md or pack_manifest.json")
        report_text = report_path.read_text(encoding="utf-8")
        if "Gate:" not in report_text:
            return _fail("daily_quant_report.md missing Gate section")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_obj = json.load(f)
        baseline_reason = str((manifest_obj.get("baseline") or {}).get("reason", ""))
        if not baseline_reason:
            return _fail("pack_manifest.json missing baseline reason")

        # Case 2: baseline missing, strict=false => rc=0
        out_missing = lonely / "quant_non_strict"
        proc_missing = _run_a6(daily_dir=lonely, base_dir=base_dir, out_dir=out_missing, strict=False)
        if proc_missing.returncode != 0:
            print(proc_missing.stdout)
            print(proc_missing.stderr)
            return _fail(f"baseline missing strict=false expected rc=0, got rc={proc_missing.returncode}")

        # Case 3: baseline missing, strict=true => rc=3
        out_strict = lonely / "quant_strict"
        proc_strict = _run_a6(daily_dir=lonely, base_dir=base_dir, out_dir=out_strict, strict=True)
        if proc_strict.returncode != 3:
            print(proc_strict.stdout)
            print(proc_strict.stderr)
            return _fail(f"baseline missing strict=true expected rc=3, got rc={proc_strict.returncode}")

        # Case 4: flat JSON mode (--daily-base + --date) works
        _write_json(base_dir / "2026-02-18.json", {"date": "2026-02-18"})
        _write_json(base_dir / "2026-02-19.json", {"date": "2026-02-19"})
        _mk_dataset(base_dir / "quant_packs" / "2026-02-18" / "run_dataset", "run-prev-flat", [100000, 100100, 100200, 100300, 100280, 100350], ["", "", "", "", "", ""])
        _mk_dataset(base_dir / "quant_packs" / "2026-02-19" / "run_dataset", "run-today-flat", [100000, 100050, 100000, 100120, 100140, 100180], ["", "attempt_cooldown", "", "", "", ""])
        proc_flat = _run_a6_flat(daily_base=base_dir, date_str="2026-02-19", strict=False)
        if proc_flat.returncode != 0:
            print(proc_flat.stdout)
            print(proc_flat.stderr)
            return _fail(f"flat mode expected rc=0, got rc={proc_flat.returncode}")
        flat_out = base_dir / "quant_packs" / "2026-02-19"
        if not (flat_out / "daily_quant_report.md").exists():
            return _fail("flat mode missing daily_quant_report.md")
        if not (flat_out / "pack_manifest.json").exists():
            return _fail("flat mode missing pack_manifest.json")

        print("[PASS] quant_daily_pack_minimal")
        print(f"[INFO] baseline_reason={baseline_reason} out_dir={out_ok}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
