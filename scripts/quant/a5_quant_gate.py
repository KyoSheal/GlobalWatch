#!/usr/bin/env python3
"""A1-5 Quant Regression Gate CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_gate import DEFAULT_RULES, run_gate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quant regression gate for candidate run datasets.")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--auto-metrics", action="store_true")
    p.add_argument("--out-dir", default="")
    p.add_argument("--rules", default="")
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--annualization", type=int, default=252)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def _load_rules(path: str) -> Dict[str, Any]:
    rules = dict(DEFAULT_RULES)
    p = str(path or "").strip()
    if not p:
        return rules
    fp = Path(p)
    if not fp.exists():
        raise FileNotFoundError(f"rules file not found: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("rules json must be object")
    for k, v in obj.items():
        rules[str(k)] = v
    return rules


def main() -> int:
    args = _build_parser().parse_args()
    baseline_dir = Path(args.baseline).resolve()
    candidate_dirs = [Path(x).resolve() for x in (args.candidate or [])]
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else None

    try:
        rules = _load_rules(str(args.rules or ""))
    except Exception as exc:
        print(f"[ERROR] failed loading rules: {exc}", file=sys.stderr)
        return 2

    code, summary = run_gate(
        baseline_dir=baseline_dir,
        candidate_dirs=candidate_dirs,
        out_dir=out_dir,
        auto_metrics=bool(args.auto_metrics),
        report_tz=str(args.report_tz),
        annualization=int(args.annualization),
        rf=float(args.rf),
        min_points=int(args.min_points),
        rules=rules,
        strict=bool(args.strict),
        verbose=bool(args.verbose),
    )

    if args.verbose:
        if code == 0:
            print("[PASS] quant_gate")
        else:
            print(f"[INFO] quant_gate_exit={code}")
        if isinstance(summary, dict):
            print(f"[INFO] summary_keys={sorted(summary.keys())}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

