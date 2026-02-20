#!/usr/bin/env python3
"""A1-4: Build multi-run leaderboard from run_dataset directories."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_io_utils import to_iso_utc
from quant_leaderboard import (
    assemble_rows,
    build_gating_summary,
    discover_datasets,
    ensure_metrics,
    rank_rows,
    render_leaderboard_md,
    write_outputs,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build leaderboard across multiple run datasets.")
    p.add_argument("--base-dir", default="outputs/Daily Report")
    p.add_argument("--pattern", default="**/run_dataset")
    p.add_argument("--out-dir", default="")
    p.add_argument("--auto-metrics", action="store_true")
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--annualization", type=int, default=252)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--min-days", type=int, default=5)
    p.add_argument(
        "--sort-by",
        default="composite",
        choices=[
            "composite",
            "sharpe",
            "total_return",
            "max_drawdown",
            "calmar",
            "vol_annualized",
            "turnover_ratio",
            "trades_total",
        ],
    )
    p.add_argument("--descending", dest="descending", action="store_true", default=True)
    p.add_argument("--ascending", dest="descending", action="store_false")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--include-raw", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (base_dir / "_leaderboard").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(base_dir, pattern=str(args.pattern))
    metrics_actions: List[Dict[str, Any]] = []
    auto_generated = 0
    for ds in datasets:
        action = ensure_metrics(
            ds,
            auto_metrics=bool(args.auto_metrics),
            report_tz=str(args.report_tz),
            annualization=int(args.annualization),
            rf=float(args.rf),
            min_points=int(args.min_points),
            verbose=bool(args.verbose),
        )
        action["dataset_dir"] = str(ds)
        metrics_actions.append(action)
        if bool(action.get("generated", False)):
            auto_generated += 1

    sort_by = str(args.sort_by or "composite")
    descending = bool(args.descending)
    if sort_by == "max_drawdown":
        descending = False

    rows, scan_summary, per_run_gating = assemble_rows(
        datasets,
        sort_by=sort_by,
        descending=descending,
        include_raw=bool(args.include_raw),
        min_days=int(args.min_days),
    )
    # re-rank once with effective direction in case overrides changed
    rows = rank_rows(rows, sort_by=sort_by, descending=descending)

    gating_summary_rows = build_gating_summary(per_run_gating)
    scan_summary = dict(scan_summary)
    scan_summary["datasets_discovered"] = len(datasets)
    scan_summary["auto_metrics_generated"] = int(auto_generated)
    scan_summary["sort_by"] = sort_by
    scan_summary["descending"] = bool(descending)

    leaderboard_json = {
        "schema_version": 1,
        "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
        "base_dir": str(base_dir),
        "pattern": str(args.pattern),
        "scan_summary": scan_summary,
        "rows_total": len(rows),
        "rows_top": rows[: max(0, int(args.top))],
        "rows": rows,
        "gating_summary": gating_summary_rows,
    }

    leaderboard_md = render_leaderboard_md(
        rows,
        scan_summary=scan_summary,
        gating_summary_rows=gating_summary_rows,
        sort_by=sort_by,
        top_n=int(args.top),
    )

    manifest_leaderboard = {
        "schema_version": 1,
        "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "pattern": str(args.pattern),
        "datasets_discovered": [str(x) for x in datasets],
        "metrics_actions": metrics_actions,
        "scan_summary": scan_summary,
    }

    write_outputs(
        out_dir,
        rows=rows,
        leaderboard_json=leaderboard_json,
        leaderboard_md=leaderboard_md,
        gating_summary_rows=gating_summary_rows,
        manifest_leaderboard=manifest_leaderboard,
    )

    if args.verbose:
        print(f"[INFO] base_dir={base_dir}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            "[INFO] "
            f"datasets={len(datasets)} rows={len(rows)} "
            f"auto_metrics_generated={auto_generated} sort_by={sort_by}"
        )
        if rows:
            top = rows[0]
            print(
                "[INFO] top="
                f"run_id={top.get('run_id')} rank={top.get('rank')} "
                f"score={top.get('composite_score')}"
            )
        print("[PASS] a4_build_leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

