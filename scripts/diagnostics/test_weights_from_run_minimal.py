#!/usr/bin/env python3
"""T49: minimal regression test for A4-3 weights extraction from run artifacts."""

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


def _approx(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= eps


def _load_weights(path: Path) -> dict:
    by_date = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        d = str(r.get("date") or "")
        t = str(r.get("ticker") or "")
        w = float(r.get("weight") or 0.0)
        by_date.setdefault(d, {})[t] = w
    return by_date


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="weights_from_run_min_"))
    try:
        run_dir = tmp / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshots = run_dir / "portfolio_snapshots.jsonl"
        rows = [
            {"time_utc": "2026-02-18T15:00:00+00:00", "target_weights": {"AAA": 0.50}},
            {"time_utc": "2026-02-18T20:00:00+00:00", "target_weights": {"AAA": 0.70, "BBB": 0.10}},
            {"time_utc": "2026-02-19T15:00:00+00:00", "target_weights": {"AAA": 0.40, "BBB": 0.10}},
            {"time_utc": "2026-02-19T20:00:00+00:00", "target_weights": {"AAA": 0.70, "BBB": 0.60}},
        ]
        with snapshots.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        out_dir = tmp / "weights_out"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a15_extract_weights_from_run.py"),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--report-tz",
            "America/New_York",
            "--verbose",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return _fail(f"a15 failed rc={proc.returncode}")

        weights_csv = out_dir / "weights.csv"
        manifest_json = out_dir / "weights_manifest.json"
        if not weights_csv.exists() or not manifest_json.exists():
            return _fail("missing weights.csv or weights_manifest.json")

        by_date = _load_weights(weights_csv)
        if sorted(by_date.keys()) != ["2026-02-18", "2026-02-19"]:
            return _fail(f"unexpected dates: {sorted(by_date.keys())}")

        d1 = by_date["2026-02-18"]
        if not _approx(d1.get("AAA", 0.0), 0.70):
            return _fail(f"day1 AAA mismatch: {d1.get('AAA')}")
        if not _approx(d1.get("BBB", 0.0), 0.10):
            return _fail(f"day1 BBB mismatch: {d1.get('BBB')}")
        if not _approx(d1.get("CASH", 0.0), 0.20):
            return _fail(f"day1 CASH mismatch: {d1.get('CASH')}")
        if not _approx(sum(d1.values()), 1.0):
            return _fail(f"day1 sum != 1: {sum(d1.values())}")

        d2 = by_date["2026-02-19"]
        if not _approx(d2.get("AAA", 0.0), 0.70 / 1.30):
            return _fail(f"day2 AAA mismatch: {d2.get('AAA')}")
        if not _approx(d2.get("BBB", 0.0), 0.60 / 1.30):
            return _fail(f"day2 BBB mismatch: {d2.get('BBB')}")
        if not _approx(d2.get("CASH", 0.0), 0.0):
            return _fail(f"day2 CASH mismatch: {d2.get('CASH')}")
        if not _approx(sum(d2.values()), 1.0):
            return _fail(f"day2 sum != 1: {sum(d2.values())}")

        manifest = json.load(open(manifest_json, "r", encoding="utf-8"))
        for key in ("schema_version", "generated_utc", "run_dir", "source_file", "days", "rows", "warnings", "hash"):
            if key not in manifest:
                return _fail(f"manifest missing key: {key}")

        print("[PASS] weights_from_run_minimal")
        print(f"[INFO] out_dir={out_dir}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

