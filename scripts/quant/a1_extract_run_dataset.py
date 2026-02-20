#!/usr/bin/env python3
"""A1-1 Run Dataset Extractor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_io_utils import ensure_dir, iter_jsonl, parse_iso_to_utc, safe_read_json, to_iso_utc

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


SCHEMA_VERSION = 1
CYCLE_COLUMNS = [
    "cycle_id",
    "time_utc",
    "session_state",
    "regime_state",
    "cash_target",
    "total_equity",
    "cash",
    "positions_value",
    "skip_reason",
    "decision_path",
    "cov_gate_reason",
    "cov_gate_max_rc",
    "rc_limit",
    "turnover_used_total",
]
EQUITY_COLUMNS = ["time_utc", "equity", "cash", "positions_value"]
TRADES_COLUMNS = [
    "time_utc",
    "cycle_id",
    "ticker",
    "side",
    "qty",
    "price",
    "notional",
    "is_forced",
    "force_reason",
    "status",
    "reason",
]


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _extract_time_utc(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in obj:
            dt = parse_iso_to_utc(obj.get(key))
            if dt is not None:
                return to_iso_utc(dt)
    return None


def _detect_run_id(snapshot_obj: Dict[str, Any], run_dir: Optional[Path]) -> str:
    run_id = str(snapshot_obj.get("run_id", "") or "").strip()
    if run_id:
        return run_id
    if run_dir is not None and run_dir.name.strip():
        return run_dir.name.strip()
    return "unknown_run"


def _find_latest_trade_history(base_out_dir: Path) -> Optional[Path]:
    latest_path: Optional[Path] = None
    latest_mtime: float = -1.0
    for candidate in base_out_dir.rglob("trade_history.jsonl"):
        if not candidate.is_file():
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = candidate
    return latest_path


def _select_cycle_source(
    run_dir: Optional[Path],
    base_out_dir: Path,
    prefer: str,
) -> Tuple[str, Optional[Path], Optional[Path]]:
    snapshots_path = (run_dir / "portfolio_snapshots.jsonl") if run_dir else (base_out_dir / "portfolio_snapshots.jsonl")
    snapshot_live_path = base_out_dir / "snapshot_live.json"

    has_snapshots = snapshots_path.exists()
    has_snapshot_live = snapshot_live_path.exists()

    if prefer == "snapshot_live":
        if has_snapshot_live:
            return "snapshot_live", snapshot_live_path, snapshots_path if has_snapshots else None
        if has_snapshots:
            return "portfolio_snapshots", snapshots_path, snapshot_live_path if has_snapshot_live else None
    else:
        if has_snapshots:
            return "portfolio_snapshots", snapshots_path, snapshot_live_path if has_snapshot_live else None
        if has_snapshot_live:
            return "snapshot_live", snapshot_live_path, snapshots_path if has_snapshots else None

    return "", None, snapshot_live_path if has_snapshot_live else None


def _read_portfolio_snapshots(
    path: Path,
    max_rows: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    cycles: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    bad_lines = 0

    for idx, (_, obj, err) in enumerate(iter_jsonl(path), start=1):
        if err or not isinstance(obj, dict):
            bad_lines += 1
            continue
        time_utc = _extract_time_utc(obj, ["time_utc", "ts", "time", "timestamp", "snapshot_time", "now_utc"])
        if not time_utc:
            bad_lines += 1
            continue

        cycle_id = _int_or_none(obj.get("cycle_id"))
        if cycle_id is None:
            cycle_id = _int_or_none(obj.get("cycle"))
        if cycle_id is None:
            cycle_id = idx

        total_equity = _num_or_none(obj.get("total_equity"))
        cash = _num_or_none(obj.get("cash"))
        positions_value = _num_or_none(obj.get("positions_value"))
        cycles.append(
            {
                "cycle_id": cycle_id,
                "time_utc": time_utc,
                "session_state": obj.get("session_state") or obj.get("session") or "",
                "regime_state": obj.get("regime_state") or "",
                "cash_target": _num_or_none(obj.get("cash_target")),
                "total_equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
                "skip_reason": obj.get("skip_reason") or "",
                "decision_path": obj.get("decision_path") or "",
                "cov_gate_reason": obj.get("cov_gate_reason") or obj.get("abort_reason") or "",
                "cov_gate_max_rc": _num_or_none(obj.get("cov_gate_max_rc") or obj.get("max_rc_fraction_cov")),
                "rc_limit": _num_or_none(obj.get("rc_limit")),
                "turnover_used_total": _num_or_none(obj.get("turnover_used_total") or obj.get("planner_turnover_used_total")),
            }
        )
        equity_rows.append(
            {
                "time_utc": time_utc,
                "equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
            }
        )

    cycles.sort(key=lambda r: (str(r.get("time_utc", "")), int(r.get("cycle_id") or 0)))
    equity_rows.sort(key=lambda r: str(r.get("time_utc", "")))
    if max_rows and max_rows > 0:
        cycles = cycles[-max_rows:]
        equity_rows = equity_rows[-max_rows:]
    return cycles, equity_rows, bad_lines


def _read_snapshot_live_equity_history(
    snapshot_obj: Dict[str, Any],
    max_rows: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    cycles: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    bad_lines = 0
    history = snapshot_obj.get("equity_history")
    if not isinstance(history, list):
        return cycles, equity_rows, bad_lines

    global_session_state = snapshot_obj.get("session_state") or snapshot_obj.get("status") or ""
    global_regime_state = snapshot_obj.get("regime_state") or ""
    global_cash_target = _num_or_none(snapshot_obj.get("cash_target"))
    global_skip_reason = snapshot_obj.get("skip_reason") or ""
    global_decision_path = snapshot_obj.get("decision_path") or ""
    global_cov_gate_reason = snapshot_obj.get("cov_gate_reason") or ""
    global_cov_gate_max_rc = _num_or_none(snapshot_obj.get("cov_gate_max_rc") or snapshot_obj.get("max_rc_fraction_cov"))
    global_rc_limit = _num_or_none(snapshot_obj.get("rc_limit"))
    global_turnover = _num_or_none(snapshot_obj.get("turnover_used_total") or snapshot_obj.get("planner_turnover_used_total"))

    for idx, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            bad_lines += 1
            continue
        time_utc = _extract_time_utc(item, ["time_utc", "ts", "time", "timestamp"])
        if not time_utc:
            bad_lines += 1
            continue
        total_equity = _num_or_none(item.get("equity") or item.get("total_equity"))
        cash = _num_or_none(item.get("cash") or snapshot_obj.get("cash"))
        positions_value = _num_or_none(item.get("positions_value") or snapshot_obj.get("positions_value"))
        cycle_id = _int_or_none(item.get("cycle_id"))
        if cycle_id is None:
            cycle_id = idx

        cycles.append(
            {
                "cycle_id": cycle_id,
                "time_utc": time_utc,
                "session_state": global_session_state,
                "regime_state": global_regime_state,
                "cash_target": global_cash_target,
                "total_equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
                "skip_reason": global_skip_reason,
                "decision_path": global_decision_path,
                "cov_gate_reason": global_cov_gate_reason,
                "cov_gate_max_rc": global_cov_gate_max_rc,
                "rc_limit": global_rc_limit,
                "turnover_used_total": global_turnover,
            }
        )
        equity_rows.append(
            {
                "time_utc": time_utc,
                "equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
            }
        )

    cycles.sort(key=lambda r: (str(r.get("time_utc", "")), int(r.get("cycle_id") or 0)))
    equity_rows.sort(key=lambda r: str(r.get("time_utc", "")))
    if max_rows and max_rows > 0:
        cycles = cycles[-max_rows:]
        equity_rows = equity_rows[-max_rows:]
    return cycles, equity_rows, bad_lines


def _read_trades(path: Path, max_rows: Optional[int]) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    bad_lines = 0
    for _, obj, err in iter_jsonl(path):
        if err or not isinstance(obj, dict):
            bad_lines += 1
            continue
        time_utc = _extract_time_utc(obj, ["time_utc", "ts", "time", "timestamp"])
        if not time_utc:
            bad_lines += 1
            continue
        cycle_id = _int_or_none(obj.get("cycle_id"))
        if cycle_id is None:
            cycle_id = _int_or_none(obj.get("cycle"))
        qty = _num_or_none(obj.get("qty"))
        if qty is None:
            qty = _num_or_none(obj.get("quantity"))
        if qty is None:
            qty = _num_or_none(obj.get("shares"))
        price = _num_or_none(obj.get("price"))
        notional = _num_or_none(obj.get("notional"))
        if notional is None and qty is not None and price is not None:
            notional = qty * price
        rows.append(
            {
                "time_utc": time_utc,
                "cycle_id": cycle_id,
                "ticker": str(obj.get("ticker", "") or "").strip(),
                "side": str(obj.get("side", "") or "").strip(),
                "qty": qty,
                "price": price,
                "notional": notional,
                "is_forced": bool(obj.get("hard") or obj.get("forced")),
                "force_reason": str(obj.get("hard_reason") or obj.get("force_reason") or "").strip(),
                "status": str(obj.get("status") or "").strip(),
                "reason": str(obj.get("reason") or obj.get("skip_reason") or "").strip(),
            }
        )
    rows.sort(key=lambda r: (str(r.get("time_utc", "")), str(r.get("ticker", ""))))
    if max_rows and max_rows > 0:
        rows = rows[-max_rows:]
    return rows, bad_lines


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), manifest, indent=2)
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract run artifacts into a stable quant dataset.")
    p.add_argument("--base-out-dir", default="outputs", help="Base outputs directory.")
    p.add_argument("--run-dir", default="", help="Optional run directory (preferred for run artifacts).")
    p.add_argument("--out-dir", default="", help="Output dataset directory.")
    p.add_argument("--prefer", default="portfolio_snapshots", choices=["portfolio_snapshots", "snapshot_live"])
    p.add_argument("--max-rows", type=int, default=0, help="Optional max rows to keep for each output table.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    base_out_dir = Path(args.base_out_dir).resolve()
    run_dir = Path(args.run_dir).resolve() if str(args.run_dir or "").strip() else None
    max_rows = int(args.max_rows) if int(args.max_rows or 0) > 0 else None

    default_out_dir = base_out_dir / "Daily Report" / datetime.now(timezone.utc).date().isoformat() / "run_dataset"
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else default_out_dir.resolve()
    ensure_dir(out_dir)

    source_kind, source_path, alt_snapshot_path = _select_cycle_source(run_dir, base_out_dir, args.prefer)
    snapshot_live_path = base_out_dir / "snapshot_live.json"
    snapshot_obj = safe_read_json(snapshot_live_path) or {}
    if not isinstance(snapshot_obj, dict):
        snapshot_obj = {}
    if not snapshot_obj and alt_snapshot_path and alt_snapshot_path.exists():
        snapshot_obj = safe_read_json(alt_snapshot_path) or {}
        if not isinstance(snapshot_obj, dict):
            snapshot_obj = {}

    if not source_kind or source_path is None or not source_path.exists():
        print(
            "[ERROR] No valid cycle source found. Missing both "
            "portfolio_snapshots.jsonl and snapshot_live.json.",
            file=sys.stderr,
        )
        return 2

    run_id = _detect_run_id(snapshot_obj, run_dir)

    if source_kind == "portfolio_snapshots":
        cycles_rows, equity_rows, bad_lines_cycles = _read_portfolio_snapshots(source_path, max_rows=max_rows)
    else:
        cycles_rows, equity_rows, bad_lines_cycles = _read_snapshot_live_equity_history(snapshot_obj, max_rows=max_rows)

    trade_history_from_snapshot = str(snapshot_obj.get("trade_history_path", "") or "").strip()
    trade_history_path: Optional[Path] = None
    if trade_history_from_snapshot:
        cand = Path(trade_history_from_snapshot)
        if not cand.is_absolute():
            cand = (ROOT / cand).resolve()
        if cand.exists():
            trade_history_path = cand
    if trade_history_path is None:
        trade_history_path = _find_latest_trade_history(base_out_dir)

    missing_trades = trade_history_path is None or not trade_history_path.exists()
    bad_lines_trades = 0
    if missing_trades:
        trades_rows: List[Dict[str, Any]] = []
    else:
        trades_rows, bad_lines_trades = _read_trades(trade_history_path, max_rows=max_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "base_out_dir": str(base_out_dir),
        "run_dir": str(run_dir) if run_dir else "",
        "created_at_utc": to_iso_utc(datetime.now(timezone.utc)),
        "chosen_sources": {
            "cycles_source_kind": source_kind,
            "cycles_source_path": str(source_path),
            "snapshot_live_path": str(snapshot_live_path),
            "trade_history_path": str(trade_history_path) if trade_history_path else "",
            "missing_trades": bool(missing_trades),
        },
        "rows": {
            "cycles": len(cycles_rows),
            "equity_curve": len(equity_rows),
            "trades": len(trades_rows),
        },
        "bad_lines": {
            "cycles_source": bad_lines_cycles,
            "trades_source": bad_lines_trades,
        },
        "max_rows": max_rows if max_rows else None,
    }

    _write_csv(out_dir / "equity_curve.csv", equity_rows, EQUITY_COLUMNS)
    _write_csv(out_dir / "cycles.csv", cycles_rows, CYCLE_COLUMNS)
    _write_csv(out_dir / "trades.csv", trades_rows, TRADES_COLUMNS)
    _write_manifest(out_dir / "manifest.json", manifest)

    if args.verbose:
        print(f"[INFO] run_id={run_id}")
        print(f"[INFO] cycles_source={source_kind} path={source_path}")
        print(f"[INFO] trade_history_path={trade_history_path if trade_history_path else '(missing)'}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            "[INFO] rows: "
            f"cycles={len(cycles_rows)} equity={len(equity_rows)} trades={len(trades_rows)} "
            f"bad_cycles={bad_lines_cycles} bad_trades={bad_lines_trades}"
        )
        print("[PASS] a1_extract_run_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

