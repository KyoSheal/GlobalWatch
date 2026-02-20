#!/usr/bin/env python3
"""A3-1 CLI: replay single cycle decision offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_replay import (
    load_price_debug,
    load_snapshot,
    resolve_run_dir,
    run_single_cycle_replay,
    write_replay_outputs,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay one cycle decision from snapshot + price_debug.")
    p.add_argument("--base-out-dir", default="outputs")
    p.add_argument("--run-dir", default="")
    p.add_argument("--date", default="")
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-on-gate", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    base_out_dir = Path(args.base_out_dir).resolve()
    cycle = int(args.cycle) if int(args.cycle or 0) > 0 else None

    run_dir, run_reason = resolve_run_dir(base_out_dir, str(args.run_dir or ""), str(args.date or ""))
    if run_dir is None and str(args.run_dir or "").strip():
        print(f"[ERROR] run_dir not found: {args.run_dir}")
        return 2

    snapshot, snapshot_info = load_snapshot(base_out_dir, run_dir, cycle)
    if not isinstance(snapshot, dict):
        print("[ERROR] snapshot unavailable")
        return 2

    price_debug, price_info = load_price_debug(snapshot, run_dir)

    if str(args.out_dir or "").strip():
        out_dir = Path(args.out_dir).resolve()
    else:
        cycle_tag = str(cycle if cycle is not None else (snapshot_info.get("cycle") or "latest"))
        if run_dir is not None:
            out_dir = (run_dir / "replay" / f"cycle_{cycle_tag}").resolve()
        else:
            out_dir = (base_out_dir / "replay" / f"cycle_{cycle_tag}").resolve()

    result = run_single_cycle_replay(
        snapshot=snapshot,
        price_debug=price_debug,
        strict=bool(args.strict),
        fail_on_gate=bool(args.fail_on_gate),
    )

    outputs = write_replay_outputs(
        out_dir=out_dir,
        result=result,
        snapshot_source=str(snapshot_info.get("path") or ""),
        price_source=price_info,
        strict=bool(args.strict),
    )

    if args.verbose:
        print(f"[A9] base_out_dir={base_out_dir}")
        print(f"[A9] run_dir={run_dir} (reason={run_reason})")
        print(f"[A9] snapshot={snapshot_info.get('path')} cycle={snapshot_info.get('cycle')}")
        print(f"[A9] price_debug_count={price_info.get('count', 0)}")
        print(f"[A9] planned_trades={len(result.planned_trades)}")
        print(f"[A9] gate_fail={bool(result.gate.get('gate_fail', False))} reason={result.gate.get('reason', '')}")
        print(f"[A9] out_dir={out_dir}")
        print(f"[A9] replay_manifest={outputs.get('manifest')}")

    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
