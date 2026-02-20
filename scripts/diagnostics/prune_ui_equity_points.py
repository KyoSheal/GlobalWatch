#!/usr/bin/env python3
"""Prune weekend/off-hours/blackout equity points from UI data sources."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, atomic_write_text, safe_read_json
from market_time_filter import sanitize_equity_rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune UI equity points with market-time sanitization.")
    p.add_argument("--base-out-dir", default="outputs", help="Base output directory (default: outputs)")
    p.add_argument(
        "--blackout-date",
        action="append",
        default=[],
        help="Blackout market date (YYYY-MM-DD). Can be repeated.",
    )
    p.add_argument("--market-tz", default="America/New_York")
    p.add_argument("--market-open-time-et", default="09:30")
    p.add_argument("--market-close-time-et", default="16:00")
    p.add_argument("--open-grace-min", type=int, default=15)
    p.add_argument("--close-grace-min", type=int, default=10)
    p.add_argument("--also-portfolio-snapshots", action="store_true")
    p.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    return p


def _sanitize_rows(rows, args):
    blackout_dates = args.blackout_date if args.blackout_date else ["2026-02-15"]
    return sanitize_equity_rows(
        rows,
        market_tz=str(args.market_tz),
        open_time_et=str(args.market_open_time_et),
        close_time_et=str(args.market_close_time_et),
        open_grace_min=int(args.open_grace_min),
        close_grace_min=int(args.close_grace_min),
        drop_weekends=True,
        drop_offhours=True,
        blackout_dates_market=blackout_dates,
    )


def _prune_snapshot_file(snapshot_path: Path, args) -> dict:
    result = {
        "path": str(snapshot_path),
        "kind": "snapshot_live",
        "processed": False,
        "changed": False,
        "stats": {},
    }
    payload = safe_read_json(str(snapshot_path), retries=3, sleep_ms=30)
    if not isinstance(payload, dict):
        return result
    rows = payload.get("equity_history", [])
    if not isinstance(rows, list):
        return result

    clean_rows, stats = _sanitize_rows(rows, args)
    changed = len(clean_rows) != len(rows)
    result["processed"] = True
    result["changed"] = bool(changed)
    result["stats"] = dict(stats)
    result["before"] = int(len(rows))
    result["after"] = int(len(clean_rows))

    if changed and bool(args.apply):
        backup_path = Path(str(snapshot_path) + ".bak")
        shutil.copy2(snapshot_path, backup_path)
        payload["equity_history"] = clean_rows
        atomic_write_json(str(snapshot_path), payload, indent=2)
        result["backup"] = str(backup_path)
    return result


def _prune_portfolio_snapshots_jsonl(path: Path, args) -> dict:
    result = {
        "path": str(path),
        "kind": "portfolio_snapshots_jsonl",
        "processed": False,
        "changed": False,
        "stats": {},
    }
    if not path.exists():
        return result

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return result

    parsed = []
    keep_lines = []
    for idx, line in enumerate(lines):
        text = str(line).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            # Keep non-parseable rows untouched.
            keep_lines.append(text)
            continue
        if not isinstance(obj, dict):
            keep_lines.append(text)
            continue
        parsed.append((idx, obj, text))

    probe_rows = [{"timestamp": obj.get("timestamp"), "__idx": idx} for idx, obj, _ in parsed]
    clean_probe_rows, stats = _sanitize_rows(probe_rows, args)
    keep_idx = {int(row.get("__idx")) for row in clean_probe_rows if "__idx" in row}

    filtered_lines = []
    for idx, obj, text in parsed:
        if idx in keep_idx:
            filtered_lines.append(text)

    changed = len(filtered_lines) != len(parsed)
    result["processed"] = True
    result["changed"] = bool(changed)
    result["stats"] = dict(stats)
    result["before"] = int(len(parsed))
    result["after"] = int(len(filtered_lines))

    if changed and bool(args.apply):
        backup_path = Path(str(path) + ".bak")
        shutil.copy2(path, backup_path)
        content = "\n".join(filtered_lines)
        if content:
            content += "\n"
        atomic_write_text(str(path), content)
        result["backup"] = str(backup_path)
    return result


def _print_result(row: dict) -> None:
    print(
        f"[PRUNE] kind={row.get('kind')} path={row.get('path')} processed={row.get('processed')} "
        f"changed={row.get('changed')} before={row.get('before', 0)} after={row.get('after', 0)} "
        f"stats={row.get('stats', {})}"
    )
    if row.get("backup"):
        print(f"[BACKUP] {row.get('backup')}")


def main() -> int:
    args = _build_parser().parse_args()
    base = Path(args.base_out_dir).resolve()
    snapshot_path = base / "snapshot_live.json"
    portfolio_path = base / "portfolio_snapshots.jsonl"

    print(f"[MODE] {'APPLY' if args.apply else 'DRY-RUN'} base_out_dir={base}")
    if not args.blackout_date:
        print("[BLACKOUT] default blackout date=2026-02-15")
    else:
        print(f"[BLACKOUT] blackout dates={args.blackout_date}")

    snapshot_result = _prune_snapshot_file(snapshot_path, args)
    _print_result(snapshot_result)

    if bool(args.also_portfolio_snapshots):
        portfolio_result = _prune_portfolio_snapshots_jsonl(portfolio_path, args)
        _print_result(portfolio_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
