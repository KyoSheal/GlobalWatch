#!/usr/bin/env python3
"""A4-10: Build dashboard-ready timeseries from daily_reports_index.json."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None

from quant_io_utils import safe_read_json
from a7_update_daily_reports_index import update_daily_reports_index


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _csv_row_from_index_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    quant = entry.get("quant") if isinstance(entry.get("quant"), dict) else {}
    rec = quant.get("reconcile") if isinstance(quant.get("reconcile"), dict) else {}
    ev = rec.get("evidence_summary") if isinstance(rec.get("evidence_summary"), dict) else {}
    if not ev:
        ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}

    # gating_top1 from evidence_summary.gating_top3 -> fallback reconcile.gating_top1
    gating_top1 = ""
    top3 = ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else []
    if top3 and isinstance(top3[0], dict):
        gating_top1 = str(top3[0].get("reason", "") or "").strip()
    if not gating_top1:
        g1 = rec.get("gating_top1") if isinstance(rec.get("gating_top1"), dict) else {}
        gating_top1 = str(g1.get("reason", "") or "").strip()

    turnover_gap = _num_or_none(rec.get("turnover_gap"))
    if turnover_gap is None:
        gaps = rec.get("gaps") if isinstance(rec.get("gaps"), dict) else {}
        turnover_gap = _num_or_none(gaps.get("turnover_gap"))

    cost_gap = _num_or_none(rec.get("cost_gap"))
    if cost_gap is None:
        gaps = rec.get("gaps") if isinstance(rec.get("gaps"), dict) else {}
        cost_gap = _num_or_none(gaps.get("cost_gap"))
    bsw = quant.get("backtest_sweep") if isinstance(quant.get("backtest_sweep"), dict) else {}
    no_trade_flag = bool(_boolish(quant.get("no_trade_flag")))

    return {
        "date": str(entry.get("date", "") or ""),
        "total_return": _num_or_none(quant.get("total_return")),
        "sharpe": _num_or_none(quant.get("sharpe")),
        "max_drawdown": _num_or_none(quant.get("max_drawdown")),
        "trades_total": int(_num_or_none(quant.get("trades_total")) or 0),
        "gate_status": str(
            rec.get("gate_status")
            if rec.get("gate_status") not in (None, "")
            else quant.get("gate_status", "NA")
        ),
        "replay_drift_status": str(rec.get("replay_drift_status", "") or "NA"),
        "reconcile_return_gap": _num_or_none(rec.get("return_gap_live_minus_backtest")),
        "reconcile_turnover_gap": turnover_gap,
        "reconcile_cost_gap": cost_gap,
        "gating_top1": gating_top1,
        "warnings_count": int(_num_or_none(rec.get("warnings_count")) or 0),
        "backtest_sweep_status": str(bsw.get("status", "") or "NA"),
        "break_even_cost_bps": _num_or_none(bsw.get("break_even_cost_bps")),
        "sensitivity_per_1bp": _num_or_none(bsw.get("sensitivity_per_1bp")),
        "return_at_10bps": _num_or_none(bsw.get("return_at_10bps")),
        "backtest_sweep_warnings_count": int(_num_or_none(bsw.get("warnings_count")) or 0),
        "exec_blocker_top1_reason": str(quant.get("exec_blocker_top1_reason", "") or ""),
        "exec_blocker_top1_ratio": _num_or_none(quant.get("exec_blocker_top1_ratio")),
        "exec_blocked_ratio": _num_or_none(quant.get("exec_blocked_ratio")),
        "no_trade_primary_reason": str(quant.get("no_trade_primary_reason", "") or ""),
        "no_trade_flag": int(1 if no_trade_flag else 0),
        "exec_no_trade_warnings_count": int(_num_or_none(quant.get("warnings_count")) or 0),
    }


def build_index_timeseries(daily_base: Path, lookback_days: int, verbose: bool = False) -> Dict[str, Any]:
    daily_base = daily_base.resolve()
    index_path = (daily_base / "daily_reports_index.json").resolve()
    if not index_path.exists():
        update_daily_reports_index(daily_base, lookback_days=max(lookback_days, 30), verbose=verbose)
    index_obj = safe_read_json(index_path)
    if not isinstance(index_obj, dict):
        raise RuntimeError(f"index missing/invalid: {index_path}")

    rows_raw = index_obj.get("reports") if isinstance(index_obj.get("reports"), list) else []
    today = datetime.now(timezone.utc).date()
    min_day = today - timedelta(days=max(1, int(lookback_days)))
    out_rows: List[Dict[str, Any]] = []
    for item in rows_raw:
        if not isinstance(item, dict):
            continue
        d = str(item.get("date", "") or "").strip()
        try:
            day = datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            continue
        if day < min_day:
            continue
        out_rows.append(_csv_row_from_index_entry(item))

    out_rows.sort(key=lambda r: str(r.get("date", "")))
    columns = [
        "date",
        "total_return",
        "sharpe",
        "max_drawdown",
        "trades_total",
        "gate_status",
        "replay_drift_status",
        "reconcile_return_gap",
        "reconcile_turnover_gap",
        "reconcile_cost_gap",
        "gating_top1",
        "warnings_count",
        "backtest_sweep_status",
        "break_even_cost_bps",
        "sensitivity_per_1bp",
        "return_at_10bps",
        "backtest_sweep_warnings_count",
        "exec_blocker_top1_reason",
        "exec_blocker_top1_ratio",
        "exec_blocked_ratio",
        "no_trade_primary_reason",
        "no_trade_flag",
        "exec_no_trade_warnings_count",
    ]

    csv_path = (daily_base / "index_timeseries.csv").resolve()
    json_path = (daily_base / "index_timeseries.json").resolve()

    # deterministic CSV rendering
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(csv_path.parent)) as tf:
        writer = csv.DictWriter(tf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
        tmp_csv = Path(tf.name)
    os.replace(tmp_csv, csv_path)

    prev_obj = safe_read_json(json_path) if json_path.exists() else {}
    generated_at = _now_utc_iso()
    if isinstance(prev_obj, dict):
        prev_cmp = {
            "schema_version": prev_obj.get("schema_version"),
            "daily_base": prev_obj.get("daily_base"),
            "lookback_days": prev_obj.get("lookback_days"),
            "columns": prev_obj.get("columns"),
            "rows": prev_obj.get("rows"),
            "source_index": prev_obj.get("source_index"),
        }
        cur_cmp = {
            "schema_version": 1,
            "daily_base": str(daily_base),
            "lookback_days": int(lookback_days),
            "columns": columns,
            "rows": out_rows,
            "source_index": str(index_path),
        }
        if prev_cmp == cur_cmp and prev_obj.get("generated_at_utc"):
            generated_at = str(prev_obj.get("generated_at_utc"))

    out_json = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "daily_base": str(daily_base),
        "lookback_days": int(lookback_days),
        "columns": columns,
        "rows": out_rows,
        "source_index": str(index_path),
    }
    _write_json_atomic(json_path, out_json)

    result = {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "rows": len(out_rows),
        "columns": columns,
    }
    if verbose:
        print(f"[A19] index={index_path}")
        print(f"[A19] rows={len(out_rows)} csv={csv_path}")
        print("[PASS] a19_build_index_timeseries")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build dashboard-ready timeseries from daily_reports_index.json.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2
    try:
        build_index_timeseries(daily_base, lookback_days=int(args.lookback_days), verbose=bool(args.verbose))
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
