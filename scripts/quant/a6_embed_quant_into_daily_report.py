#!/usr/bin/env python3
"""CLI: embed A1-6 quant report into daily report file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_daily_embed import discover_quant_md, embed_quant_into_daily_report, write_embed_manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed quant pack markdown into daily report.")
    p.add_argument("--daily-dir", default="", help="Legacy daily directory path mode.")
    p.add_argument("--daily-base", default="outputs/Daily Report", help="Base path for flat JSON daily reports.")
    p.add_argument("--date", default="", help="Date string YYYY-MM-DD for flat JSON daily reports.")
    p.add_argument("--quant-md", default="")
    p.add_argument("--report-file", default="")
    p.add_argument("--mode", default="replace", choices=["append", "replace"])
    p.add_argument("--out-file", default="")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_dir_arg = str(args.daily_dir or "").strip()
    date_arg = str(args.date or "").strip()
    if not daily_dir_arg and not date_arg:
        print("[ERROR] provide either --daily-dir or --date (with --daily-base)", file=sys.stderr)
        return 2

    daily_dir = Path(daily_dir_arg).resolve() if daily_dir_arg else None
    daily_base = Path(args.daily_base).resolve()
    quant_md = Path(args.quant_md).resolve() if str(args.quant_md or "").strip() else None
    if quant_md is None:
        quant_md = discover_quant_md(
            daily_dir=daily_dir,
            daily_base=daily_base,
            date_str=date_arg,
            quant_md_arg="",
        )
    report_file = Path(args.report_file).resolve() if str(args.report_file or "").strip() else None
    out_file = Path(args.out_file).resolve() if str(args.out_file or "").strip() else None

    result = embed_quant_into_daily_report(
        daily_dir=daily_dir,
        daily_base=daily_base,
        date_str=date_arg,
        quant_md=quant_md,
        report_file=report_file,
        mode=str(args.mode or "replace"),
        out_file=out_file,
        strict=bool(args.strict),
    )
    manifest_path = write_embed_manifest(
        daily_dir=daily_dir,
        daily_base=daily_base,
        date_str=date_arg,
        result=result,
    )

    if args.verbose:
        print(f"[INFO] daily_dir={daily_dir}")
        print(f"[INFO] daily_base={daily_base} date={date_arg}")
        print(f"[INFO] quant_md={quant_md}")
        print(f"[INFO] report_file={result.daily_report_in}")
        print(f"[INFO] out_file={result.daily_report_out}")
        print(f"[INFO] mode={result.mode} created_fallback={result.created_fallback}")
        print(f"[INFO] is_json={result.is_json}")
        if result.warnings:
            print(f"[WARN] {result.warnings}")
        if result.notes:
            print(f"[INFO] notes={result.notes}")
        print(f"[INFO] embed_manifest={manifest_path}")
        if result.exit_code == 0:
            print("[PASS] a6_embed_quant_into_daily_report")
        else:
            print(f"[INFO] a6_embed_exit_code={result.exit_code}")

    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
