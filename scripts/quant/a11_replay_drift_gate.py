#!/usr/bin/env python3
"""A3-3 CLI: Replay Window Drift Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_replay_drift import DEFAULT_RULES, run_drift_gate


def _parse_forbid_tags(text: str) -> Optional[list[str]]:
    s = str(text or "").strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return parts if parts else None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate drift gate from replay window outputs.")
    p.add_argument("--replay-window-dir", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-on-drift", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rules-json", default="")
    p.add_argument("--max-weights-l1", type=float, default=None)
    p.add_argument("--max-abs-weight-delta", type=float, default=None)
    p.add_argument("--max-trade-delta-ratio", type=float, default=None)
    p.add_argument("--forbid-tags", default="")
    p.add_argument("--verbose", action="store_true")
    return p


def _merge_rules(args) -> Dict[str, Any]:
    rules = dict(DEFAULT_RULES)

    rules_json = str(args.rules_json or "").strip()
    if rules_json:
        p = Path(rules_json).resolve()
        if not p.exists():
            raise FileNotFoundError(f"rules json not found: {p}")
        obj = json.load(open(p, "r", encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("rules-json must be a json object")
        rules.update(obj)

    if args.max_weights_l1 is not None:
        rules["max_weights_l1"] = float(args.max_weights_l1)
    if args.max_abs_weight_delta is not None:
        rules["max_abs_weight_delta"] = float(args.max_abs_weight_delta)
    if args.max_trade_delta_ratio is not None:
        rules["max_trade_delta_ratio"] = float(args.max_trade_delta_ratio)

    forbid = _parse_forbid_tags(args.forbid_tags)
    if forbid is not None:
        rules["forbid_tags"] = forbid
    else:
        if bool(args.strict):
            rules["forbid_tags"] = list(rules.get("forbid_tags_strict", DEFAULT_RULES["forbid_tags_strict"]))
        else:
            rules["forbid_tags"] = []

    return rules


def main() -> int:
    args = _build_parser().parse_args()
    replay_window_dir = Path(args.replay_window_dir).resolve()
    if not replay_window_dir.exists() or not replay_window_dir.is_dir():
        print(f"[ERROR] replay-window-dir not found: {replay_window_dir}", file=sys.stderr)
        return 2

    if str(args.out_dir or "").strip():
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (replay_window_dir / "drift_gate").resolve()

    strict = bool(args.strict)
    if args.fail_on_drift is None:
        fail_on_drift = bool(strict)
    else:
        fail_on_drift = bool(args.fail_on_drift)

    try:
        rules = _merge_rules(args)
    except Exception as exc:
        print(f"[ERROR] invalid rules: {exc}", file=sys.stderr)
        return 2

    rc, result = run_drift_gate(
        replay_window_dir=replay_window_dir,
        out_dir=out_dir,
        strict=strict,
        fail_on_drift=fail_on_drift,
        rules=rules,
    )

    if args.verbose:
        print(f"[A11] replay_window_dir={replay_window_dir}")
        print(f"[A11] out_dir={out_dir}")
        print(f"[A11] strict={strict} fail_on_drift={fail_on_drift}")
        print(f"[A11] status={result.get('status')} rc={rc}")
        print(f"[A11] result={out_dir / 'drift_gate_result.json'}")
        print(f"[A11] summary={out_dir / 'drift_gate_summary.csv'}")
        print(f"[A11] report={out_dir / 'drift_gate_report.md'}")

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
