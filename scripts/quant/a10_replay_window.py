#!/usr/bin/env python3
"""A3-2 CLI: replay multi-cycle window + attribution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_replay import resolve_run_dir
from quant_replay_window import run_replay_window


def _parse_date(s: str) -> Optional[str]:
    text = str(s or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _latest_date_in_daily_base(daily_base: Path) -> Optional[str]:
    dates = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        try:
            d = datetime.strptime(p.stem, "%Y-%m-%d").date().isoformat()
            dates.append(d)
        except Exception:
            continue
    if not dates:
        return None
    dates.sort()
    return dates[-1]


def _resolve_run_dir_from_daily(base_out_dir: Path, daily_base: Path, date_str: str) -> Optional[Path]:
    report_path = (daily_base / f"{date_str}.json").resolve()
    if report_path.exists():
        try:
            obj = json.load(open(report_path, "r", encoding="utf-8"))
        except Exception:
            obj = {}
        if isinstance(obj, dict):
            for key in ("run_dir", "runDir", "output_dir", "out_dir"):
                v = str(obj.get(key, "") or "").strip()
                if not v:
                    continue
                p = Path(v)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                if p.exists() and p.is_dir():
                    return p
            for key in ("run_id", "runId", "session_id"):
                rid = str(obj.get(key, "") or "").strip()
                if rid:
                    p = (base_out_dir / date_str[:7] / rid).resolve()
                    if p.exists() and p.is_dir():
                        return p
    p, _ = resolve_run_dir(base_out_dir, "", date_str)
    return p


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay window across cycles with attribution.")
    p.add_argument("--base-out-dir", default="outputs")
    p.add_argument("--run-dir", default="")
    p.add_argument("--daily-base", default="")
    p.add_argument("--date", default="")
    p.add_argument("--cycles", default="")
    p.add_argument("--start-cycle", type=int, default=0)
    p.add_argument("--end-cycle", type=int, default=0)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out-dir", default="")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compare-ref", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--fail-on-drift", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    base_out_dir = Path(args.base_out_dir).resolve()

    run_dir: Optional[Path] = None
    if str(args.run_dir or "").strip():
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"[ERROR] run_dir not found: {run_dir}")
            return 2
    else:
        daily_base = Path(args.daily_base).resolve() if str(args.daily_base or "").strip() else None
        date_str = _parse_date(args.date)
        if daily_base is not None:
            if date_str is None:
                date_str = _latest_date_in_daily_base(daily_base)
            if date_str is not None:
                run_dir = _resolve_run_dir_from_daily(base_out_dir, daily_base, date_str)
        if run_dir is None:
            run_dir, _ = resolve_run_dir(base_out_dir, "", date_str or "")

    if run_dir is None:
        print("[ERROR] failed to resolve run_dir")
        return 2

    start_cycle = int(args.start_cycle) if int(args.start_cycle or 0) > 0 else None
    end_cycle = int(args.end_cycle) if int(args.end_cycle or 0) > 0 else None

    if str(args.out_dir or "").strip():
        out_dir = Path(args.out_dir).resolve()
    else:
        if str(args.cycles or "").strip():
            tag = str(args.cycles).replace(":", "-")
        elif start_cycle is not None or end_cycle is not None:
            tag = f"{start_cycle or 'auto'}-{end_cycle or 'auto'}"
        else:
            tag = "auto"
        out_dir = (run_dir / "replay_window" / tag).resolve()

    rc, manifest = run_replay_window(
        run_dir=run_dir,
        cycles_spec=str(args.cycles or ""),
        start_cycle=start_cycle,
        end_cycle=end_cycle,
        step=max(1, int(args.step or 1)),
        out_dir=out_dir,
        strict=bool(args.strict),
        compare_ref=bool(args.compare_ref),
        max_cycles=max(1, int(args.max_cycles or 200)),
        fail_on_drift=bool(args.fail_on_drift),
        verbose=bool(args.verbose),
    )

    if args.verbose:
        print(f"[A10] out_dir={out_dir}")
        print(f"[A10] manifest={out_dir / 'replay_window_manifest.json'}")
        print(f"[A10] status={manifest.get('status')} rc={rc}")

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
