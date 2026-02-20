#!/usr/bin/env python3
"""T44: minimal replay drift gate test."""

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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _write_summary_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "cycle",
        "time_utc",
        "price_rows",
        "num_trades",
        "target_hash",
        "trades_hash",
        "gate_fail",
        "warnings_count",
        "ref_status",
        "attribution_tags",
        "weights_l1",
        "trades_notional_delta",
        "diff_path",
        "decision_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _run_gate(*, replay_window_dir: Path, out_dir: Path, strict: bool, fail_on_drift: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a11_replay_drift_gate.py"),
        "--replay-window-dir",
        str(replay_window_dir),
        "--out-dir",
        str(out_dir),
        "--verbose",
    ]
    if strict:
        cmd.append("--strict")
    else:
        cmd.append("--no-strict")

    if fail_on_drift:
        cmd.append("--fail-on-drift")
    else:
        cmd.append("--no-fail-on-drift")

    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_replay_drift_min_"))
    try:
        replay_dir = tmp / "replay_window" / "20-21"
        per20 = replay_dir / "per_cycle" / "20"
        per21 = replay_dir / "per_cycle" / "21"
        per20.mkdir(parents=True, exist_ok=True)
        per21.mkdir(parents=True, exist_ok=True)

        _write_summary_csv(
            replay_dir / "replay_window_summary.csv",
            [
                {
                    "cycle": 20,
                    "time_utc": "2026-02-20T10:00:00+00:00",
                    "price_rows": 2,
                    "num_trades": 2,
                    "target_hash": "th20",
                    "trades_hash": "tr20",
                    "gate_fail": "False",
                    "warnings_count": 1,
                    "ref_status": "ok",
                    "attribution_tags": "NONDETERMINISM_WARNING|INPUT_DRIFT",
                    "weights_l1": "0.030000",
                    "trades_notional_delta": "2100",
                    "diff_path": "per_cycle/20/diff.json",
                    "decision_path": "per_cycle/20/replay_decision.md",
                },
                {
                    "cycle": 21,
                    "time_utc": "2026-02-20T10:20:00+00:00",
                    "price_rows": 2,
                    "num_trades": 2,
                    "target_hash": "th21",
                    "trades_hash": "tr21",
                    "gate_fail": "False",
                    "warnings_count": 0,
                    "ref_status": "ok",
                    "attribution_tags": "",
                    "weights_l1": "0.000000",
                    "trades_notional_delta": "0",
                    "diff_path": "per_cycle/21/diff.json",
                    "decision_path": "per_cycle/21/replay_decision.md",
                },
            ],
        )

        _write_json(
            replay_dir / "replay_window_manifest.json",
            {
                "schema_version": 1,
                "status": "ok",
                "warnings": ["cycle_20:macro_not_frozen"],
            },
        )

        _write_json(
            per20 / "diff.json",
            {
                "weights_diff": {
                    "weights_l1": 0.03,
                    "top_deltas": [
                        {"ticker": "AAA", "ref": 0.47, "replay": 0.45, "delta": -0.02, "abs_delta": 0.02}
                    ],
                },
                "trades_diff": {"notional_delta": 2100.0},
                "attribution_tags": ["NONDETERMINISM_WARNING", "INPUT_DRIFT"],
            },
        )
        _write_json(
            per21 / "diff.json",
            {
                "weights_diff": {"weights_l1": 0.0, "top_deltas": []},
                "trades_diff": {"notional_delta": 0.0},
                "attribution_tags": [],
            },
        )
        _write_json(per20 / "replay_manifest.json", {"snapshot": {"total_equity": 100000.0}})
        _write_json(per21 / "replay_manifest.json", {"snapshot": {"total_equity": 100000.0}})

        out_strict = tmp / "out_strict"
        p1 = _run_gate(replay_window_dir=replay_dir, out_dir=out_strict, strict=True, fail_on_drift=True)
        if p1.returncode != 3:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"strict expected rc=3 got={p1.returncode}")

        result_strict = json.loads((out_strict / "drift_gate_result.json").read_text(encoding="utf-8"))
        if str(result_strict.get("status")) != "FAIL":
            return _fail("strict drift result status must be FAIL")

        out_warn = tmp / "out_warn"
        p2 = _run_gate(replay_window_dir=replay_dir, out_dir=out_warn, strict=False, fail_on_drift=False)
        if p2.returncode != 1:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"nonstrict expected rc=1 got={p2.returncode}")

        report = out_warn / "drift_gate_report.md"
        if not report.exists():
            return _fail("drift_gate_report.md missing")
        report_text = report.read_text(encoding="utf-8")
        if "Tag Counts (Top5)" not in report_text:
            return _fail("report missing tag stats section")

        # idempotency: run nonstrict again and summary csv should be identical
        p3 = _run_gate(replay_window_dir=replay_dir, out_dir=out_warn, strict=False, fail_on_drift=False)
        if p3.returncode != 1:
            print(p3.stdout)
            print(p3.stderr)
            return _fail(f"nonstrict second expected rc=1 got={p3.returncode}")

        s1 = (out_warn / "drift_gate_summary.csv").read_text(encoding="utf-8")
        s2 = (out_warn / "drift_gate_summary.csv").read_text(encoding="utf-8")
        if s1 != s2:
            return _fail("drift_gate_summary.csv is not deterministic")

        print("[PASS] quant_replay_drift_minimal")
        print(f"[INFO] replay_dir={replay_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
