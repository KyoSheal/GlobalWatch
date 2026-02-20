#!/usr/bin/env python3
"""CLI wrapper for run_analytics.summarize_range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_analytics import ALLOWED_RANGES, ALLOWED_RUN_KINDS, summarize_range  # noqa: E402


def _resolve_kinds(args) -> set[str]:
    if isinstance(args.kinds, str) and args.kinds.strip():
        out = set()
        for part in args.kinds.split(","):
            key = str(part or "").strip().lower()
            if key in ALLOWED_RUN_KINDS:
                out.add(key)
        return out or {"live"}
    out = {"live"}
    if bool(args.include_dryrun):
        out.add("dryrun")
    if bool(args.include_diagnostics):
        out.add("diagnostics")
    if bool(args.include_test):
        out.add("test")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize runs from month cache + run summaries.")
    parser.add_argument("--base-out-dir", type=str, default="outputs", help="Base output root (default: outputs)")
    parser.add_argument("--range", type=str, default="1M", choices=sorted(ALLOWED_RANGES), help="Range: 1M|3M|6M|1Y|YTD")
    parser.add_argument("--kinds", type=str, default="", help="Comma-separated run kinds (live,dryrun,diagnostics,test).")
    parser.add_argument("--include-dryrun", action="store_true", help="Include dryrun runs.")
    parser.add_argument("--include-diagnostics", action="store_true", help="Include diagnostics runs.")
    parser.add_argument("--include-test", action="store_true", help="Include test runs.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    kinds = _resolve_kinds(args)

    result = summarize_range(base_out_dir=args.base_out_dir, range_key=args.range, kinds=sorted(kinds))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("Run Summary")
    print(f"  Base Out Dir: {result['base_out_dir']}")
    print(f"  Range: {result['range']} ({result['range_start_utc']} -> {result['range_end_utc']})")
    print(f"  Kinds: {','.join(result.get('kinds', []))}")
    print(f"  used_month_cache_count: {result.get('used_month_cache_count', 0)}")
    print(f"  fallback_month_scan_count: {result.get('fallback_month_scan_count', 0)}")
    print(f"  Run Count: {result['run_count']}")
    by_kind = result.get("run_count_by_kind", {})
    if isinstance(by_kind, dict) and by_kind:
        print("  Run Count By Kind:")
        for k in sorted(by_kind.keys()):
            print(f"    - {k}: {int(by_kind.get(k, 0) or 0)}")
    print(f"  Total PnL: {result['total_pnl']:.2f}")
    print(f"  Start Equity: {result['start_equity']}")
    print(f"  End Equity: {result['end_equity']}")
    print(f"  Max Drawdown: {result['max_drawdown']}")
    print("  By Risk Profile:")
    by_risk = result.get("by_risk_profile", {})
    if isinstance(by_risk, dict) and by_risk:
        for profile in sorted(by_risk.keys()):
            item = by_risk[profile]
            print(
                f"    - {profile}: runs={int(item.get('run_count', 0))} "
                f"pnl={float(item.get('total_pnl', 0.0)):.2f} "
                f"avg_ret={float(item.get('avg_total_return', 0.0)):.4f}"
            )
    else:
        print("    - (none)")
    missing = result.get("missing_run_ids", [])
    print(f"  Missing Summaries: {len(missing) if isinstance(missing, list) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
