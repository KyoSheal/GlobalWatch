#!/usr/bin/env python3
"""A3-1 replay helpers: deterministic single-cycle decision replay."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import parse_iso_to_utc, safe_read_json, to_iso_utc

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _normalize_ticker(t: Any) -> str:
    return str(t or "").strip().upper()


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in columns}
            for c in ("desired_trade_value", "priority"):
                if c in out and isinstance(out[c], float):
                    out[c] = f"{out[c]:.10f}"
            w.writerow(out)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (-abs(float(r.get("priority", 0.0) or 0.0)), str(r.get("ticker", ""))))


def resolve_run_dir(base_out_dir: Path, run_dir_arg: str, date_str: str) -> Tuple[Optional[Path], str]:
    if str(run_dir_arg or "").strip():
        p = Path(run_dir_arg).resolve()
        if p.exists() and p.is_dir():
            return p, "explicit"
        return None, "explicit_missing"

    date_norm = str(date_str or "").strip()
    if date_norm:
        date_compact = date_norm.replace("-", "")
        month_dir = (base_out_dir / date_norm[:7]).resolve()
        hits: List[Path] = []
        if month_dir.exists():
            for child in month_dir.iterdir():
                if child.is_dir() and child.name.startswith(f"{date_compact}-"):
                    hits.append(child.resolve())
        if hits:
            hits.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
            return hits[0], "date_prefix"

    # fallback latest month/run dir under base_out_dir
    hits_all: List[Path] = []
    for month in base_out_dir.glob("20??-??"):
        if not month.is_dir():
            continue
        for child in month.iterdir():
            if child.is_dir() and (child / "snapshot_live.json").exists():
                hits_all.append(child.resolve())
    if hits_all:
        hits_all.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
        return hits_all[0], "latest_run_with_snapshot"

    return None, "not_found"


def load_snapshot(base_out_dir: Path, run_dir: Optional[Path], cycle: Optional[int]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Path] = []
    if run_dir is not None:
        candidates.append((run_dir / "snapshot_live.json").resolve())
    candidates.append((base_out_dir / "snapshot_live.json").resolve())

    for path in candidates:
        snap = safe_read_json(path)
        if isinstance(snap, dict):
            info = {
                "path": str(path),
                "run_id": str(snap.get("run_id") or ""),
                "cycle": int(_num_or_none(snap.get("cycle") or snap.get("cycle_id") or 0) or 0),
                "selected_cycle": int(cycle) if cycle is not None else None,
            }
            return snap, info
    return None, {"path": "", "run_id": "", "cycle": 0, "selected_cycle": int(cycle) if cycle is not None else None}


def _extract_price_debug_from_obj(obj: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for tk, row in obj.items():
            ticker = _normalize_ticker(tk)
            if not ticker:
                continue
            if isinstance(row, dict):
                price = _num_or_none(row.get("price") or row.get("current_price"))
                ts = str(row.get("price_ts") or row.get("ts") or row.get("timestamp") or "")
                status = str(row.get("status") or "")
                source = str(row.get("source") or "")
                out[ticker] = {
                    "price": price,
                    "price_ts": ts,
                    "status": status,
                    "source": source,
                    "bar_interval": row.get("bar_interval"),
                    "tz_ok": row.get("tz_ok"),
                }
        return out
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            ticker = _normalize_ticker(row.get("ticker"))
            if not ticker:
                continue
            price = _num_or_none(row.get("price") or row.get("current_price"))
            ts = str(row.get("price_ts") or row.get("ts") or row.get("timestamp") or "")
            status = str(row.get("status") or "")
            source = str(row.get("source") or "")
            out[ticker] = {
                "price": price,
                "price_ts": ts,
                "status": status,
                "source": source,
                "bar_interval": row.get("bar_interval"),
                "tz_ok": row.get("tz_ok"),
            }
    return out


def load_price_debug(snapshot: Dict[str, Any], run_dir: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    sources_checked: List[str] = []
    merged: Dict[str, Dict[str, Any]] = {}

    # 1) snapshot embedded price_debug
    for key in ("price_debug", "current_price_debug", "price_debug_items"):
        obj = snapshot.get(key)
        if obj is None:
            continue
        src = f"snapshot.{key}"
        sources_checked.append(src)
        rows = _extract_price_debug_from_obj(obj)
        if rows:
            merged.update(rows)

    # 2) run_dir files
    if run_dir is not None:
        for name in (
            "price_debug.json",
            "price_debug_latest.json",
            "price_debug_save.json",
            "price_debug_items.json",
        ):
            p = (run_dir / name).resolve()
            if not p.exists():
                continue
            sources_checked.append(str(p))
            obj = safe_read_json(p)
            if isinstance(obj, dict):
                rows = _extract_price_debug_from_obj(obj.get("items") if isinstance(obj.get("items"), (list, dict)) else obj)
                if rows:
                    merged.update(rows)

    info = {
        "sources_checked": sources_checked,
        "count": len(merged),
    }
    return merged, info


def build_price_provider(price_debug: Dict[str, Dict[str, Any]]):
    cache = dict(price_debug)

    def _get_quote(ticker: str) -> Optional[Dict[str, Any]]:
        return cache.get(_normalize_ticker(ticker))

    return _get_quote


def _compute_current_weights(snapshot: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    total_equity = _num_or_none(snapshot.get("total_equity"))
    cash = _num_or_none(snapshot.get("cash"))
    positions = snapshot.get("positions") if isinstance(snapshot.get("positions"), dict) else {}

    if total_equity is None:
        pos_total = 0.0
        for _, row in positions.items():
            if not isinstance(row, dict):
                continue
            v = _num_or_none(row.get("value"))
            if v is None:
                qty = _num_or_none(row.get("quantity")) or 0.0
                px = _num_or_none(row.get("price")) or 0.0
                v = qty * px
            pos_total += max(0.0, float(v or 0.0))
        total_equity = pos_total + float(cash or 0.0)

    if total_equity is None or total_equity <= 0:
        total_equity = 1.0

    weights: Dict[str, float] = {}
    for tk, row in positions.items():
        ticker = _normalize_ticker(tk)
        if not ticker or ticker == "CASH" or not isinstance(row, dict):
            continue
        v = _num_or_none(row.get("value"))
        if v is None:
            qty = _num_or_none(row.get("quantity")) or 0.0
            px = _num_or_none(row.get("price")) or 0.0
            v = qty * px
        w = max(0.0, float(v or 0.0) / float(total_equity))
        if w > 0:
            weights[ticker] = w

    cash_w = float(cash or 0.0) / float(total_equity)
    weights["CASH"] = max(0.0, cash_w)

    total_w = sum(weights.values())
    if total_w > 0:
        for k in list(weights.keys()):
            weights[k] = float(weights[k] / total_w)
    return weights, float(total_equity)


def _normalize_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (weights or {}).items():
        ticker = _normalize_ticker(k)
        if not ticker:
            continue
        w = _num_or_none(v)
        if w is None:
            continue
        out[ticker] = max(0.0, float(w))
    if "CASH" not in out:
        out["CASH"] = 0.0
    total = sum(out.values())
    if total <= 0:
        return {"CASH": 1.0}
    for k in list(out.keys()):
        out[k] = float(out[k] / total)
    return out


def _extract_target_weights(snapshot: Dict[str, Any], current_weights: Dict[str, float]) -> Tuple[Dict[str, float], str]:
    for key in ("target_weights", "planned_target_weights", "target_allocations"):
        if isinstance(snapshot.get(key), dict):
            return _normalize_weights(snapshot.get(key)), f"snapshot.{key}"
    return _normalize_weights(current_weights), "fallback_current_weights"


def _gate_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(snapshot.get("skip_reason") or snapshot.get("abort_reason") or snapshot.get("cov_gate_reason") or "")
    detail = str(snapshot.get("skip_reason_detail") or snapshot.get("decision_path") or "")
    gate_fail = bool(reason)
    return {
        "gate_fail": gate_fail,
        "reason": reason,
        "detail": detail,
        "risk_gate_basis": snapshot.get("risk_gate_basis"),
    }


def _generate_planned_trades(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    total_equity: float,
    *,
    min_abs_delta_w: float = 1e-4,
) -> List[Dict[str, Any]]:
    tickers = sorted(set([k for k in current_weights.keys() if k != "CASH"]) | set([k for k in target_weights.keys() if k != "CASH"]))
    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        cw = float(current_weights.get(ticker, 0.0) or 0.0)
        tw = float(target_weights.get(ticker, 0.0) or 0.0)
        delta = tw - cw
        if abs(delta) < float(min_abs_delta_w):
            continue
        side = "BUY" if delta > 0 else "SELL"
        desired = float(delta * float(total_equity))
        rows.append(
            {
                "ticker": ticker,
                "side": side,
                "desired_trade_value": desired,
                "is_forced": False,
                "force_reason": "",
                "priority": abs(delta),
            }
        )
    return _sort_rows(rows)


@dataclass
class ReplayResult:
    exit_code: int
    warnings: List[str]
    snapshot_info: Dict[str, Any]
    price_info: Dict[str, Any]
    target_weights: Dict[str, float]
    planned_trades: List[Dict[str, Any]]
    gate: Dict[str, Any]
    steps_ok: Dict[str, bool]


def run_single_cycle_replay(
    *,
    snapshot: Dict[str, Any],
    price_debug: Dict[str, Dict[str, Any]],
    strict: bool,
    fail_on_gate: bool,
) -> ReplayResult:
    warnings: List[str] = []
    steps_ok = {
        "snapshot_loaded": isinstance(snapshot, dict),
        "price_loaded": len(price_debug) > 0,
        "macro_frozen": True,
        "planned_trades_built": False,
    }
    if not steps_ok["snapshot_loaded"]:
        return ReplayResult(2, ["snapshot_missing"], {}, {"count": 0}, {}, [], {}, steps_ok)

    if not steps_ok["price_loaded"]:
        warnings.append("missing_price_debug")
        if strict:
            return ReplayResult(2, warnings, {}, {"count": 0}, {}, [], {}, steps_ok)

    current_weights, total_equity = _compute_current_weights(snapshot)
    target_weights, target_src = _extract_target_weights(snapshot, current_weights)
    if target_src == "fallback_current_weights":
        warnings.append("macro_not_frozen")
        steps_ok["macro_frozen"] = False
        if strict:
            return ReplayResult(2, warnings, {}, {"count": len(price_debug)}, target_weights, [], {}, steps_ok)

    planned = _generate_planned_trades(current_weights, target_weights, total_equity)
    steps_ok["planned_trades_built"] = True
    gate = _gate_summary(snapshot)

    rc = 0
    if warnings and not strict:
        rc = 1
    if fail_on_gate and bool(gate.get("gate_fail")):
        rc = 3

    snap_info = {
        "cycle": int(_num_or_none(snapshot.get("cycle") or snapshot.get("cycle_id") or 0) or 0),
        "run_id": str(snapshot.get("run_id") or ""),
        "target_source": target_src,
        "total_equity": float(total_equity),
    }
    return ReplayResult(
        exit_code=rc,
        warnings=warnings,
        snapshot_info=snap_info,
        price_info={"count": len(price_debug)},
        target_weights=target_weights,
        planned_trades=planned,
        gate=gate,
        steps_ok=steps_ok,
    )


def _build_decision_md(result: ReplayResult, out_dir: Path) -> str:
    lines: List[str] = []
    lines.append("# Replay Decision")
    lines.append("")
    lines.append(f"- Generated: `{_now_utc_iso()}`")
    lines.append(f"- Run ID: `{result.snapshot_info.get('run_id', '')}`")
    lines.append(f"- Cycle: `{result.snapshot_info.get('cycle', '')}`")
    lines.append(f"- Target Source: `{result.snapshot_info.get('target_source', '')}`")
    lines.append(f"- Price Rows: `{result.price_info.get('count', 0)}`")
    lines.append("")

    lines.append("## Gate")
    lines.append(f"- gate_fail: `{bool(result.gate.get('gate_fail', False))}`")
    lines.append(f"- reason: `{result.gate.get('reason', '')}`")
    lines.append(f"- detail: `{result.gate.get('detail', '')}`")
    lines.append("")

    lines.append("## Planned Trades (Top)")
    if result.planned_trades:
        for row in result.planned_trades[:10]:
            lines.append(
                f"- {row.get('ticker')} {row.get('side')} value={float(row.get('desired_trade_value', 0.0)):.2f} "
                f"priority={float(row.get('priority', 0.0)):.6f}"
            )
    else:
        lines.append("- no planned trades")
    lines.append("")

    if result.warnings:
        lines.append("## Warnings")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Outputs")
    lines.append(f"- `replay_manifest.json`")
    lines.append(f"- `replay_target_weights.json`")
    lines.append(f"- `replay_planned_trades.csv`")
    lines.append(f"- `replay_decision.md`")
    return "\n".join(lines) + "\n"


def write_replay_outputs(
    *,
    out_dir: Path,
    result: ReplayResult,
    snapshot_source: str,
    price_source: Dict[str, Any],
    strict: bool,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "started_at_utc": _now_utc_iso(),
        "finished_at_utc": _now_utc_iso(),
        "strict": bool(strict),
        "exit_code": int(result.exit_code),
        "warnings": list(result.warnings),
        "snapshot": {
            "source_path": str(snapshot_source),
            **result.snapshot_info,
        },
        "price_debug": {
            "source": price_source,
            **result.price_info,
        },
        "gate": result.gate,
        "steps_ok": result.steps_ok,
        "rows": {
            "planned_trades": len(result.planned_trades),
            "target_weights": len(result.target_weights),
        },
    }

    target_obj = {
        "schema_version": 1,
        "run_id": result.snapshot_info.get("run_id", ""),
        "cycle": result.snapshot_info.get("cycle", 0),
        "target_weights": result.target_weights,
    }

    trades_csv = out_dir / "replay_planned_trades.csv"
    target_json = out_dir / "replay_target_weights.json"
    manifest_json = out_dir / "replay_manifest.json"
    decision_md = out_dir / "replay_decision.md"

    _write_csv(
        trades_csv,
        result.planned_trades,
        ["ticker", "side", "desired_trade_value", "is_forced", "force_reason", "priority"],
    )
    _write_json_atomic(target_json, target_obj)
    _write_json_atomic(manifest_json, manifest)
    _write_text(decision_md, _build_decision_md(result, out_dir))

    return {
        "manifest": manifest_json,
        "target_weights": target_json,
        "planned_trades": trades_csv,
        "decision": decision_md,
    }
