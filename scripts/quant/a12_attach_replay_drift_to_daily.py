#!/usr/bin/env python3
"""A3-4 CLI: attach replay drift summary into flat daily report json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_replay_drift_daily import attach_replay_drift_to_daily


def _load_rules_json(path: str) -> Optional[Dict[str, Any]]:
    p = str(path or "").strip()
    if not p:
        return None
    rp = Path(p).resolve()
    if not rp.exists():
        raise FileNotFoundError(f"rules-json not found: {rp}")
    obj = json.load(open(rp, "r", encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("rules-json must be a JSON object")
    return obj


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach replay drift gate summary to flat daily report json.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest date json under daily-base")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fail-on-drift", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rules-json", default="")
    p.add_argument("--out-base", default="", help="default: <daily-base>/quant_packs/<date>/replay_drift")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[A12] ERROR: daily base not found: {daily_base}", file=sys.stderr)
        return 2

    try:
        rules = _load_rules_json(args.rules_json)
    except Exception as exc:
        print(f"[A12] ERROR: {exc}", file=sys.stderr)
        return 2

    out_base = Path(args.out_base).resolve() if str(args.out_base or "").strip() else None
    rc, manifest = attach_replay_drift_to_daily(
        daily_base=daily_base,
        date_str=str(args.date or ""),
        strict=bool(args.strict),
        fail_on_drift=args.fail_on_drift,
        rules=rules,
        out_base=out_base,
        verbose=bool(args.verbose),
    )

    if args.verbose:
        print(f"[A12] daily_base={daily_base}")
        print(f"[A12] date={manifest.get('date')}")
        print(f"[A12] replay_window_dir={manifest.get('replay_window_dir')}")
        print(f"[A12] drift_gate_out_dir={manifest.get('drift_gate_out_dir')}")
        print(f"[A12] status={manifest.get('status')} rc={rc}")
        print(f"[A12] manifest={(Path(manifest.get('drift_gate_out_dir')) / 'replay_drift_manifest.json') if manifest.get('drift_gate_out_dir') else '-'}")
        warns = manifest.get("warnings") if isinstance(manifest.get("warnings"), list) else []
        if warns:
            print(f"[A12] warnings={warns}")

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
