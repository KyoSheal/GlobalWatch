#!/usr/bin/env python3
"""Attach execution blockers/no-trade summary to flat daily report JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_exec_blockers import (
    attach_exec_blockers_to_daily,
    discover_latest_date,
    parse_date_str,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach execution blockers and no-trade summary into daily JSON.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest under daily-base")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--auto-compute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2

    date_norm = parse_date_str(args.date) if str(args.date or "").strip() else discover_latest_date(daily_base)
    if not date_norm:
        print(f"[ERROR] no valid date found under {daily_base}")
        return 2

    rc, info = attach_exec_blockers_to_daily(
        daily_base=daily_base,
        date_str=date_norm,
        strict=bool(args.strict),
        auto_compute=bool(args.auto_compute),
        verbose=bool(args.verbose),
    )
    if int(rc) == 2:
        print(f"[ERROR] {info.get('error', 'attach failed')}")
    elif bool(args.verbose):
        print("[PASS] a20_attach_exec_blockers_to_daily")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

