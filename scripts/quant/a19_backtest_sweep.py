#!/usr/bin/env python3
"""A4-11 CLI: backtest robustness sweep for cost_bps scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from quant_backtest_sweep import parse_list_csv, run_sweep, write_outputs
from weights_from_run import build_daily_weights, write_weights


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic backtest sweep by cost_bps list.")
    p.add_argument("--weights-csv", default="")
    p.add_argument("--run-dir", default="")
    p.add_argument("--price-store", required=True)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--cost-bps-list", required=True, help='Comma list, e.g. "0,2.5,5,10"')
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--initial-equity", type=float, default=100000.0)
    p.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve_weights_path(args: argparse.Namespace, out_dir: Path) -> Tuple[Path, Dict[str, str]]:
    meta: Dict[str, str] = {}
    weights_csv = str(args.weights_csv or "").strip()
    run_dir = str(args.run_dir or "").strip()
    if weights_csv:
        p = Path(weights_csv).resolve()
        if not p.exists():
            raise FileNotFoundError(f"weights-csv not found: {p}")
        meta["weights_source"] = "weights_csv"
        return p, meta
    if not run_dir:
        raise ValueError("either --weights-csv or --run-dir is required")
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run-dir not found: {run_path}")
    rows, manifest = build_daily_weights(
        run_path,
        report_tz=str(args.report_tz or "America/New_York"),
        date_start=str(args.start or ""),
        date_end=str(args.end or ""),
    )
    weights_out = (out_dir / "weights").resolve()
    write_info = write_weights(weights_out, rows, manifest)
    p = Path(write_info["weights_csv"]).resolve()
    meta["weights_source"] = "run_dir"
    meta["weights_generated_from_run_dir"] = str(run_path)
    return p, meta


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cost_bps_list: List[float] = parse_list_csv(str(args.cost_bps_list))
    except Exception as exc:
        print(f"[ERROR] invalid --cost-bps-list: {exc}")
        return 2

    try:
        weights_path, meta = _resolve_weights_path(args, out_dir)
    except Exception as exc:
        print(f"[ERROR] unable to resolve weights source: {exc}")
        return 2

    try:
        rows, manifest = run_sweep(
            weights_csv=weights_path,
            price_store_dir=Path(args.price_store).resolve(),
            start=str(args.start or ""),
            end=str(args.end or ""),
            cost_bps_list=cost_bps_list,
            out_dir=out_dir,
            initial_equity=float(args.initial_equity),
            rebalance_rule=str(args.rebalance),
        )
    except Exception as exc:
        print(f"[ERROR] sweep failed: {exc}")
        return 2

    if meta:
        request = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
        request.update(meta)
        manifest["request"] = request

    write_info = write_outputs(out_dir, rows, manifest)

    if bool(args.verbose):
        print(f"[INFO] out_dir={out_dir}")
        print(f"[INFO] scenarios={len(rows)} cost_bps_list={','.join([str(x) for x in cost_bps_list])}")
        print(f"[INFO] report={write_info.get('report_md', '')}")
        print("[PASS] a19_backtest_sweep")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

