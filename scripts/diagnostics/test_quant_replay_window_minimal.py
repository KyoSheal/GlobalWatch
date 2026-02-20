#!/usr/bin/env python3
"""T43: minimal replay window + attribution determinism test."""

from __future__ import annotations

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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _run_window(*, run_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a10_replay_window.py"),
        "--run-dir",
        str(run_dir),
        "--cycles",
        "101:103:1",
        "--out-dir",
        str(out_dir),
        "--strict",
        "--compare-ref",
        "--verbose",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_replay_window_min_"))
    try:
        run_dir = tmp / "outputs" / "2026-02" / "20260218-1948-9246a7"
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshots = [
            {
                "run_id": "20260218-1948-9246a7",
                "cycle": 101,
                "time_utc": "2026-02-18T19:48:00+00:00",
                "total_equity": 100000.0,
                "cash": 20000.0,
                "positions": {
                    "AAA": {"value": 30000.0},
                    "BBB": {"value": 50000.0},
                },
                "target_weights": {"AAA": 0.45, "BBB": 0.35, "CASH": 0.20},
                "price_debug": {
                    "AAA": {"price": 100.0, "price_ts": "2026-02-18T19:48:00+00:00", "status": "LIVE"},
                    "BBB": {"price": 100.0, "price_ts": "2026-02-18T19:48:00+00:00", "status": "LIVE"},
                },
            },
            {
                "run_id": "20260218-1948-9246a7",
                "cycle": 102,
                "time_utc": "2026-02-18T20:08:00+00:00",
                "total_equity": 100100.0,
                "cash": 20020.0,
                "positions": {
                    "AAA": {"value": 30030.0},
                    "BBB": {"value": 50050.0},
                },
                "target_weights": {"AAA": 0.45, "BBB": 0.35, "CASH": 0.20},
                "price_debug": {
                    "AAA": {"price": 101.0, "price_ts": "2026-02-18T20:08:00+00:00", "status": "LIVE"},
                    "BBB": {"price": 99.0, "price_ts": "2026-02-18T20:08:00+00:00", "status": "LIVE"},
                },
            },
            {
                "run_id": "20260218-1948-9246a7",
                "cycle": 103,
                "time_utc": "2026-02-18T20:28:00+00:00",
                "total_equity": 100200.0,
                "cash": 20040.0,
                "positions": {
                    "AAA": {"value": 30060.0},
                    "BBB": {"value": 50100.0},
                },
                "target_weights": {"AAA": 0.45, "BBB": 0.35, "CASH": 0.20},
                "price_debug": {
                    "AAA": {"price": 102.0, "price_ts": "2026-02-18T20:28:00+00:00", "status": "LIVE"},
                    "BBB": {"price": 98.5, "price_ts": "2026-02-18T20:28:00+00:00", "status": "LIVE"},
                },
            },
        ]

        with (run_dir / "cycle_snapshots.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in snapshots:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # reference: cycle 102 intentionally differs to trigger attribution INPUT_DRIFT
        refs = {
            "101": {
                "target_weights": {"AAA": 0.45, "BBB": 0.35, "CASH": 0.20},
                "planned_trades": [
                    {"ticker": "AAA", "side": "BUY", "desired_trade_value": 15000.0},
                    {"ticker": "BBB", "side": "SELL", "desired_trade_value": -15000.0},
                ],
                "gate": {"gate_fail": False, "reason": ""},
                "price_source": "snapshot.price_debug",
            },
            "102": {
                "target_weights": {"AAA": 0.47, "BBB": 0.33, "CASH": 0.20},
                "planned_trades": [
                    {"ticker": "AAA", "side": "BUY", "desired_trade_value": 17000.0},
                    {"ticker": "BBB", "side": "SELL", "desired_trade_value": -17000.0},
                ],
                "gate": {"gate_fail": False, "reason": ""},
                "price_source": "snapshot.price_debug",
            },
            "103": {
                "target_weights": {"AAA": 0.45, "BBB": 0.35, "CASH": 0.20},
                "planned_trades": [
                    {"ticker": "AAA", "side": "BUY", "desired_trade_value": 15000.0},
                    {"ticker": "BBB", "side": "SELL", "desired_trade_value": -15000.0},
                ],
                "gate": {"gate_fail": False, "reason": ""},
                "price_source": "snapshot.price_debug",
            },
        }
        _write_json(run_dir / "references_by_cycle.json", refs)

        out1 = run_dir / "replay_window" / "r1"
        out2 = run_dir / "replay_window" / "r2"

        p1 = _run_window(run_dir=run_dir, out_dir=out1)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first window replay expected rc=0 got={p1.returncode}")

        p2 = _run_window(run_dir=run_dir, out_dir=out2)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second window replay expected rc=0 got={p2.returncode}")

        s1 = (out1 / "replay_window_summary.csv")
        s2 = (out2 / "replay_window_summary.csv")
        if not s1.exists() or not s2.exists():
            return _fail("summary csv missing")
        if s1.read_text(encoding="utf-8") != s2.read_text(encoding="utf-8"):
            return _fail("summary csv is not deterministic across two runs")

        report = out1 / "replay_window_report.md"
        if not report.exists():
            return _fail("replay_window_report.md missing")
        report_text = report.read_text(encoding="utf-8")
        if "Attribution Tags (Top)" not in report_text:
            return _fail("report missing attribution section")

        diff_102 = out1 / "per_cycle" / "102" / "diff.json"
        if not diff_102.exists():
            return _fail("cycle 102 diff.json missing")
        diff_obj = json.loads(diff_102.read_text(encoding="utf-8"))
        l1 = float(((diff_obj.get("weights_diff") or {}).get("weights_l1") or 0.0))
        tags = list(diff_obj.get("attribution_tags") or [])
        if l1 <= 0:
            return _fail("expected positive weights_l1 for cycle 102")
        if not any(t in tags for t in ("INPUT_DRIFT", "SOURCE_MISMATCH", "REF_MISSING")):
            return _fail(f"unexpected attribution tags for cycle 102: {tags}")

        print("[PASS] quant_replay_window_minimal")
        print(f"[INFO] report={report}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
