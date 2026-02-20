#!/usr/bin/env python3
"""A1-6: Daily Quant Pack CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_daily_pack import build_daily_pack


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build daily quant pack from A1-1..A1-5 artifacts.")
    p.add_argument("--daily-dir", default="", help="Legacy daily directory mode, e.g. outputs/Daily Report/2026-02-18")
    p.add_argument("--daily-base", default="outputs/Daily Report", help="Base folder for flat JSON daily reports.")
    p.add_argument("--date", default="", help="Date string YYYY-MM-DD for flat JSON daily report mode.")
    p.add_argument("--dataset-dir", default="")
    p.add_argument("--baseline-dataset", default="")
    p.add_argument("--baseline-mode", default="prev_day", choices=["prev_day", "best_recent"])
    p.add_argument("--base-dir", default="outputs/Daily Report")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--auto-extract", action="store_true", default=False)
    p.add_argument("--base-out-dir", default="outputs")
    p.add_argument("--auto-metrics", action="store_true", default=True)
    p.add_argument("--auto-gate", action="store_true", default=True)
    p.add_argument("--auto-leaderboard", action="store_true", default=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--annualization", type=int, default=252)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_dir_arg = str(args.daily_dir or "").strip()
    date_arg = str(args.date or "").strip()
    if not daily_dir_arg and not date_arg:
        print("[ERROR] provide either --daily-dir or --date (with --daily-base)", file=sys.stderr)
        return 2

    daily_base = Path(args.daily_base).resolve()
    daily_dir = Path(daily_dir_arg).resolve() if daily_dir_arg else (daily_base / date_arg).resolve()
    base_dir = Path(args.base_dir).resolve()
    if (not daily_dir_arg) and (str(args.base_dir).strip() == "outputs/Daily Report"):
        base_dir = daily_base
    if str(args.out_dir or "").strip():
        out_dir = Path(args.out_dir).resolve()
    else:
        if daily_dir_arg:
            out_dir = (daily_dir / "quant").resolve()
        else:
            out_dir = (daily_base / "quant_packs" / date_arg).resolve()
    base_out_dir = Path(args.base_out_dir).resolve()

    code, manifest = build_daily_pack(
        daily_dir_arg=daily_dir_arg,
        daily_base=daily_base,
        date_str=date_arg,
        dataset_dir_arg=str(args.dataset_dir or ""),
        baseline_dataset_arg=str(args.baseline_dataset or ""),
        baseline_mode=str(args.baseline_mode or "prev_day"),
        base_dir=base_dir,
        lookback_days=int(args.lookback_days),
        auto_extract=bool(args.auto_extract),
        base_out_dir=base_out_dir,
        auto_metrics=bool(args.auto_metrics),
        auto_gate=bool(args.auto_gate),
        auto_leaderboard=bool(args.auto_leaderboard),
        out_dir=out_dir,
        strict=bool(args.strict),
        report_tz=str(args.report_tz),
        annualization=int(args.annualization),
        rf=float(args.rf),
        min_points=int(args.min_points),
        verbose=bool(args.verbose),
    )

    if args.verbose:
        baseline = manifest.get("baseline", {}) if isinstance(manifest, dict) else {}
        print(f"[INFO] daily_dir={daily_dir}")
        print(f"[INFO] daily_base={daily_base} date={date_arg or daily_dir.name}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            "[INFO] "
            f"status={manifest.get('status')} "
            f"baseline_reason={baseline.get('reason')} "
            f"gate_status={manifest.get('gate_status', '')}"
        )
        print(f"[INFO] manifest={out_dir / 'pack_manifest.json'}")
        print(f"[INFO] report={out_dir / 'daily_quant_report.md'}")
        if code == 0:
            print("[PASS] a6_build_daily_quant_pack")
        else:
            print(f"[INFO] a6_exit_code={code}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
