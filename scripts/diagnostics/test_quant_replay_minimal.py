#!/usr/bin/env python3
"""T42: minimal deterministic replay test for A3-1."""

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


def _run_replay(*, run_dir: Path, cycle: int, out_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a9_replay_cycle.py"),
        "--run-dir",
        str(run_dir),
        "--cycle",
        str(cycle),
        "--out-dir",
        str(out_dir),
        "--strict",
        "--verbose",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_replay_min_"))
    try:
        run_dir = tmp_root / "outputs" / "2026-02" / "20260218-1948-9246a7"
        cycle = 277
        snapshot = {
            "run_id": "20260218-1948-9246a7",
            "cycle": cycle,
            "total_equity": 100000.0,
            "cash": 20000.0,
            "positions": {
                "AAA": {"quantity": 300, "price": 100.0, "value": 30000.0},
                "BBB": {"quantity": 500, "price": 100.0, "value": 50000.0},
            },
            "target_weights": {
                "AAA": 0.45,
                "BBB": 0.35,
                "CASH": 0.20,
            },
            "skip_reason": "",
            "price_debug": {
                "AAA": {"price": 100.0, "price_ts": "2026-02-18T19:48:00+00:00", "status": "LIVE", "source": "test"},
                "BBB": {"price": 100.0, "price_ts": "2026-02-18T19:48:00+00:00", "status": "LIVE", "source": "test"},
            },
        }
        _write_json(run_dir / "snapshot_live.json", snapshot)

        out1 = run_dir / "replay" / "r1"
        out2 = run_dir / "replay" / "r2"

        p1 = _run_replay(run_dir=run_dir, cycle=cycle, out_dir=out1)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first replay expected rc=0 got={p1.returncode}")

        p2 = _run_replay(run_dir=run_dir, cycle=cycle, out_dir=out2)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second replay expected rc=0 got={p2.returncode}")

        trades1 = (out1 / "replay_planned_trades.csv")
        trades2 = (out2 / "replay_planned_trades.csv")
        if not trades1.exists() or not trades2.exists():
            return _fail("replay_planned_trades.csv missing")

        text1 = trades1.read_text(encoding="utf-8")
        text2 = trades2.read_text(encoding="utf-8")
        if text1 != text2:
            return _fail("planned trades differ between two replay runs")

        manifest = json.loads((out1 / "replay_manifest.json").read_text(encoding="utf-8"))
        steps_ok = manifest.get("steps_ok", {})
        if not all(bool(steps_ok.get(k)) for k in ("snapshot_loaded", "price_loaded", "macro_frozen", "planned_trades_built")):
            return _fail(f"steps_ok not all true: {steps_ok}")

        decision_md = (out1 / "replay_decision.md")
        if not decision_md.exists():
            return _fail("replay_decision.md missing")

        print("[PASS] quant_replay_minimal")
        print(f"[INFO] run_dir={run_dir} cycle={cycle} trades_rows={max(0, len(text1.splitlines())-1)}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
