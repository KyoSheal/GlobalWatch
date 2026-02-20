#!/usr/bin/env python3
"""A1-3: run-to-run compare CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_compare import (
    compare_two_runs,
    load_metrics_and_daily,
    render_compare_markdown,
    write_delta_daily_csv,
)

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _metrics_file(dataset_dir: Path) -> Path:
    return dataset_dir / "metrics" / "metrics.json"


def _daily_file(dataset_dir: Path) -> Path:
    return dataset_dir / "metrics" / "daily_returns.csv"


def _ensure_metrics_generated(
    dataset_dir: Path,
    *,
    report_tz: str,
    annualization: int,
    rf: float,
    verbose: bool,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str((dataset_dir / "metrics").resolve()),
        "--report-tz",
        str(report_tz),
        "--annualization",
        str(int(annualization)),
        "--rf",
        str(float(rf)),
    ]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare two run datasets (A1-3).")
    p.add_argument("--dataset-a", required=True)
    p.add_argument("--dataset-b", required=True)
    p.add_argument("--out-dir", default="", help="Output directory for compare outputs.")
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--annualization", type=int, default=252)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--auto-metrics", action="store_true", help="Generate metrics.json if missing.")
    p.add_argument("--fail-on", action="append", default=[], help="Optional fail rules (repeatable).")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dataset_a = Path(args.dataset_a).resolve()
    dataset_b = Path(args.dataset_b).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (dataset_b / "compare").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_metrics:
        for ds in (dataset_a, dataset_b):
            if (not _metrics_file(ds).exists()) or (not _daily_file(ds).exists()):
                if args.verbose:
                    print(f"[INFO] auto-metrics generating for {ds}")
                rc = _ensure_metrics_generated(
                    ds,
                    report_tz=str(args.report_tz),
                    annualization=int(args.annualization),
                    rf=float(args.rf),
                    verbose=bool(args.verbose),
                )
                if rc != 0:
                    print(f"[ERROR] auto-metrics failed for {ds} rc={rc}", file=sys.stderr)
                    return rc

    metrics_a, daily_a, quality_a = load_metrics_and_daily(
        dataset_a,
        report_tz=str(args.report_tz),
        annualization=int(args.annualization),
        rf=float(args.rf),
    )
    metrics_b, daily_b, quality_b = load_metrics_and_daily(
        dataset_b,
        report_tz=str(args.report_tz),
        annualization=int(args.annualization),
        rf=float(args.rf),
    )

    compare, daily_delta_rows = compare_two_runs(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        daily_a=daily_a,
        daily_b=daily_b,
        quality_a=quality_a,
        quality_b=quality_b,
        report_tz=str(args.report_tz),
        annualization=int(args.annualization),
        rf=float(args.rf),
        fail_rules=[str(x) for x in (args.fail_on or [])],
    )

    compare_json_path = out_dir / "compare.json"
    compare_md_path = out_dir / "compare.md"
    delta_csv_path = out_dir / "delta_daily_returns.csv"

    _write_json(compare_json_path, compare)
    _write_text(compare_md_path, render_compare_markdown(compare))
    write_delta_daily_csv(delta_csv_path, daily_delta_rows)

    if args.verbose:
        head = compare.get("headline", {}) or {}
        dcmp = compare.get("daily_returns_compare", {}) or {}
        print(f"[INFO] dataset_a={dataset_a}")
        print(f"[INFO] dataset_b={dataset_b}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            "[INFO] "
            f"winner={head.get('winner')} "
            f"reason={head.get('winner_reason')} "
            f"overlap_days={dcmp.get('overlap_days', 0)}"
        )
        print("[PASS] a3_compare_runs")

    fail_block = compare.get("fail_rules", {}) or {}
    if bool(fail_block.get("enabled", False)) and not bool(fail_block.get("ok", True)):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

