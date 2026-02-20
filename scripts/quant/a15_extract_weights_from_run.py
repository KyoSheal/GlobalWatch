#!/usr/bin/env python3
"""A4-3 CLI: extract deterministic daily target weights from run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from weights_from_run import build_daily_weights, write_weights


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract daily target weights from run artifacts.")
    p.add_argument("--run-dir", required=True, help="Run directory or root containing portfolio_snapshots.jsonl")
    p.add_argument("--out-dir", default="", help="default outputs/backtest_weights/<hash>")
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--date-start", default="", help="Optional local date inclusive YYYY-MM-DD")
    p.add_argument("--date-end", default="", help="Optional local date inclusive YYYY-MM-DD")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"[ERROR] run dir not found: {run_dir}", file=sys.stderr)
        return 2

    try:
        rows, manifest = build_daily_weights(
            run_dir,
            report_tz=str(args.report_tz),
            date_start=str(args.date_start or ""),
            date_end=str(args.date_end or ""),
        )
    except Exception as exc:
        print(f"[ERROR] failed to extract weights: {exc}", file=sys.stderr)
        return 2

    run_hash = _canonical_hash(
        {
            "run_dir": str(run_dir),
            "source_file": manifest.get("source_file"),
            "source_kind": manifest.get("source_kind"),
            "report_tz": manifest.get("report_tz"),
            "date_start": manifest.get("date_start"),
            "date_end": manifest.get("date_end"),
            "rows_hash": manifest.get("hash"),
        }
    )
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (ROOT / "outputs" / "backtest_weights" / run_hash).resolve()

    manifest = dict(manifest)
    manifest["extract_hash"] = run_hash
    write_info: Dict[str, str] = write_weights(out_dir, rows, manifest)
    if args.verbose:
        print(f"[INFO] run_dir={run_dir}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            f"[INFO] source={manifest.get('source_kind')} days={manifest.get('days')} rows={manifest.get('rows')} warnings={len(manifest.get('warnings') or [])}"
        )
        print(f"[INFO] weights_csv={write_info.get('weights_csv')}")
        print("[PASS] a15_extract_weights_from_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

