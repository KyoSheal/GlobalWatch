#!/usr/bin/env python3
"""A4-11: run deterministic backtest parameter sweeps."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backtest_engine import load_returns, load_weights, run_backtest, write_backtest


SWEEP_COLUMNS = [
    "scenario_id",
    "cost_bps",
    "start_equity",
    "end_equity",
    "total_return",
    "max_drawdown",
    "sharpe",
    "turnover_notional",
    "total_cost",
    "days",
    "trade_rows",
    "rebalance_count",
    "weights_csv",
    "returns_cache_dir",
    "report_md",
    "warnings",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


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
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _format_cost_bps(cost_bps: float) -> str:
    if abs(cost_bps - round(cost_bps)) < 1e-9:
        return str(int(round(cost_bps)))
    s = f"{float(cost_bps):.2f}".rstrip("0").rstrip(".")
    return s


def parse_list_csv(raw: str) -> List[float]:
    items = [s.strip() for s in str(raw or "").split(",")]
    vals: List[float] = []
    for item in items:
        if not item:
            continue
        try:
            vals.append(float(item))
        except Exception as exc:
            raise ValueError(f"invalid float in list: '{item}'") from exc
    if not vals:
        raise ValueError("empty list")
    return sorted(vals)


def _extract_weights_meta(weight_rows: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
    dates = sorted({str(r.get("date") or "") for r in weight_rows if str(r.get("date") or "")})
    tickers = sorted(
        {
            str(r.get("ticker") or "").upper()
            for r in weight_rows
            if str(r.get("ticker") or "").upper() not in ("", "CASH")
        }
    )
    if not dates:
        raise ValueError("weights has no date rows")
    if not tickers:
        raise ValueError("weights has no tradable tickers")
    return tickers, dates[0], dates[-1]


def _filter_returns_rows(
    rows: List[Dict[str, Any]],
    *,
    tickers: List[str],
    start: str,
    end: str,
) -> List[Dict[str, Any]]:
    ticker_set = {str(t).upper() for t in tickers}
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = str(row.get("date") or "")
        t = str(row.get("ticker") or "").upper()
        if t not in ticker_set:
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        rv = _as_float(row.get("ret"))
        if rv is None:
            continue
        out.append({"date": d, "ticker": t, "ret": float(rv)})
    out.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return out


def _resolve_returns_source(
    price_store_dir: Path,
    *,
    tickers: List[str],
    start: str,
    end: str,
) -> Tuple[Path, List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    root = Path(price_store_dir).resolve()
    candidates: List[Path] = []
    if (root / "returns_daily.csv").exists():
        candidates.append(root)
    for p in root.rglob("returns_daily.csv"):
        parent = p.parent.resolve()
        if parent not in candidates:
            candidates.append(parent)
    candidates.sort(key=lambda d: (float((d / "returns_daily.csv").stat().st_mtime), str(d)), reverse=True)
    for cache_dir in candidates:
        try:
            rows = load_returns(cache_dir)
        except Exception:
            continue
        filtered = _filter_returns_rows(rows, tickers=tickers, start=start, end=end)
        if not filtered:
            continue
        present = {str(r.get("ticker") or "").upper() for r in filtered}
        if all(t in present for t in tickers):
            return cache_dir, filtered, warnings
    raise FileNotFoundError(
        f"no returns_daily.csv under {root} covering tickers={','.join(tickers)} range={start}..{end}"
    )


def _summarize_single(
    *,
    scenario_id: str,
    cost_bps: float,
    manifest: Dict[str, Any],
    weights_csv: Path,
    returns_cache_dir: Path,
    report_md: Path,
) -> Dict[str, Any]:
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    cost_summary = manifest.get("cost_summary") if isinstance(manifest.get("cost_summary"), dict) else {}
    warnings = manifest.get("warnings") if isinstance(manifest.get("warnings"), list) else []
    row: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "cost_bps": float(cost_bps),
        "start_equity": _as_float(summary.get("start_equity"), 0.0),
        "end_equity": _as_float(summary.get("end_equity"), 0.0),
        "total_return": _as_float(summary.get("total_return"), 0.0),
        "max_drawdown": _as_float(summary.get("max_drawdown"), 0.0),
        "sharpe": None,
        "turnover_notional": _as_float(cost_summary.get("total_turnover_notional"), 0.0),
        "total_cost": _as_float(cost_summary.get("total_cost"), 0.0),
        "days": int(summary.get("days", 0) or 0),
        "trade_rows": int(cost_summary.get("trade_rows", 0) or 0),
        "rebalance_count": int(cost_summary.get("rebalance_count", 0) or 0),
        "weights_csv": str(weights_csv),
        "returns_cache_dir": str(returns_cache_dir),
        "report_md": str(report_md),
        "warnings": ";".join([str(w) for w in warnings]),
    }
    return row


def run_single_backtest(
    *,
    weights_csv: Path,
    price_store_dir: Path,
    start: str,
    end: str,
    cost_bps: float,
    out_dir: Path,
    initial_equity: float = 100000.0,
    rebalance_rule: str = "daily",
) -> Dict[str, Any]:
    weight_rows = load_weights(Path(weights_csv).resolve())
    tickers, inferred_start, inferred_end = _extract_weights_meta(weight_rows)
    start = str(start or inferred_start)
    end = str(end or inferred_end)
    returns_cache_dir, returns_rows = _resolve_returns_source(
        Path(price_store_dir),
        tickers=tickers,
        start=start,
        end=end,
    )[:2]
    returns_rows = _filter_returns_rows(returns_rows, tickers=tickers, start=start, end=end)

    scenario_id = f"costbps_{_format_cost_bps(float(cost_bps))}"
    scenario_out = Path(out_dir).resolve() / "scenarios" / scenario_id
    scenario_out.mkdir(parents=True, exist_ok=True)

    equity_rows, trades_rows, manifest = run_backtest(
        returns_rows,
        weight_rows,
        initial_equity=float(initial_equity),
        cost_bps=float(cost_bps),
        rebalance_rule=str(rebalance_rule),
    )
    manifest["inputs"] = {
        "weights_csv": str(Path(weights_csv).resolve()),
        "returns_cache_dir": str(returns_cache_dir),
        "date_start": start,
        "date_end": end,
        "tickers": tickers,
    }
    write_info = write_backtest(scenario_out, equity_rows, trades_rows, manifest)
    return _summarize_single(
        scenario_id=scenario_id,
        cost_bps=float(cost_bps),
        manifest=manifest,
        weights_csv=Path(weights_csv).resolve(),
        returns_cache_dir=returns_cache_dir,
        report_md=Path(write_info.get("report_md", "")),
    )


def run_sweep(
    *,
    weights_csv: Path,
    price_store_dir: Path,
    start: str,
    end: str,
    cost_bps_list: List[float],
    out_dir: Path,
    initial_equity: float = 100000.0,
    rebalance_rule: str = "daily",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    weight_rows = load_weights(Path(weights_csv).resolve())
    _, inferred_start, inferred_end = _extract_weights_meta(weight_rows)
    effective_start = str(start or inferred_start)
    effective_end = str(end or inferred_end)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for cost_bps in sorted([float(x) for x in cost_bps_list]):
        row = run_single_backtest(
            weights_csv=weights_csv,
            price_store_dir=price_store_dir,
            start=effective_start,
            end=effective_end,
            cost_bps=float(cost_bps),
            out_dir=out_dir,
            initial_equity=float(initial_equity),
            rebalance_rule=str(rebalance_rule),
        )
        rows.append(row)
        if row.get("warnings"):
            warnings.append(str(row.get("warnings")))
    rows.sort(key=lambda r: float(r.get("cost_bps", 0.0) or 0.0))
    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "request": {
            "weights_csv": str(Path(weights_csv).resolve()),
            "price_store_dir": str(Path(price_store_dir).resolve()),
            "start": effective_start,
            "end": effective_end,
            "cost_bps_list": [float(x) for x in sorted(cost_bps_list)],
            "initial_equity": float(initial_equity),
            "rebalance_rule": str(rebalance_rule),
        },
        "request_hash": _canonical_hash(
            {
                "weights_csv": str(Path(weights_csv).resolve()),
                "price_store_dir": str(Path(price_store_dir).resolve()),
                "start": effective_start,
                "end": effective_end,
                "cost_bps_list": [float(x) for x in sorted(cost_bps_list)],
                "initial_equity": float(initial_equity),
                "rebalance_rule": str(rebalance_rule),
            }
        ),
        "scenario_count": len(rows),
        "warnings": sorted(set([w for w in warnings if w])),
    }
    return rows, manifest


def render_sweep_report_md(rows: List[Dict[str, Any]], manifest: Dict[str, Any], top_k: int = 10) -> str:
    lines: List[str] = []
    lines.append("# Backtest Sweep Report")
    lines.append("")
    lines.append(f"- generated_utc: `{manifest.get('generated_utc', '')}`")
    lines.append(f"- request_hash: `{manifest.get('request_hash', '')}`")
    req = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
    lines.append(f"- weights_csv: `{req.get('weights_csv', '')}`")
    lines.append(f"- price_store_dir: `{req.get('price_store_dir', '')}`")
    lines.append(f"- date_range: `{req.get('start', '')}` -> `{req.get('end', '')}`")
    lines.append(f"- scenarios: `{manifest.get('scenario_count', 0)}`")
    lines.append("")
    lines.append("| scenario_id | cost_bps | end_equity | total_return | max_drawdown | total_cost | turnover_notional | days |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows[: max(1, int(top_k))]:
        lines.append(
            "| {sid} | {cb:.4f} | {end:.4f} | {ret:.6f} | {mdd:.6f} | {tc:.6f} | {to:.6f} | {days} |".format(
                sid=str(row.get("scenario_id", "")),
                cb=float(row.get("cost_bps", 0.0) or 0.0),
                end=float(row.get("end_equity", 0.0) or 0.0),
                ret=float(row.get("total_return", 0.0) or 0.0),
                mdd=float(row.get("max_drawdown", 0.0) or 0.0),
                tc=float(row.get("total_cost", 0.0) or 0.0),
                to=float(row.get("turnover_notional", 0.0) or 0.0),
                days=int(row.get("days", 0) or 0),
            )
        )
    lines.append("")
    warnings = manifest.get("warnings") if isinstance(manifest.get("warnings"), list) else []
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Dict[str, str]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict[str, Any]] = []
    for r in rows:
        csv_rows.append(
            {
                "scenario_id": str(r.get("scenario_id", "")),
                "cost_bps": f"{float(r.get('cost_bps', 0.0) or 0.0):.6f}",
                "start_equity": f"{float(r.get('start_equity', 0.0) or 0.0):.10f}",
                "end_equity": f"{float(r.get('end_equity', 0.0) or 0.0):.10f}",
                "total_return": f"{float(r.get('total_return', 0.0) or 0.0):.10f}",
                "max_drawdown": f"{float(r.get('max_drawdown', 0.0) or 0.0):.10f}",
                "sharpe": "" if r.get("sharpe") in (None, "") else f"{float(r.get('sharpe', 0.0) or 0.0):.10f}",
                "turnover_notional": f"{float(r.get('turnover_notional', 0.0) or 0.0):.10f}",
                "total_cost": f"{float(r.get('total_cost', 0.0) or 0.0):.10f}",
                "days": int(r.get("days", 0) or 0),
                "trade_rows": int(r.get("trade_rows", 0) or 0),
                "rebalance_count": int(r.get("rebalance_count", 0) or 0),
                "weights_csv": str(r.get("weights_csv", "")),
                "returns_cache_dir": str(r.get("returns_cache_dir", "")),
                "report_md": str(r.get("report_md", "")),
                "warnings": str(r.get("warnings", "")),
            }
        )

    results_csv = out / "sweep_results.csv"
    results_json = out / "sweep_results.json"
    report_md = out / "sweep_report.md"
    manifest_json = out / "sweep_manifest.json"

    _write_csv(results_csv, csv_rows, SWEEP_COLUMNS)
    _write_json_atomic(
        results_json,
        {
            "schema_version": 1,
            "generated_utc": manifest.get("generated_utc", _now_utc_iso()),
            "request_hash": manifest.get("request_hash", ""),
            "rows": rows,
        },
    )
    _write_text_atomic(report_md, render_sweep_report_md(rows, manifest))
    _write_json_atomic(manifest_json, manifest)

    return {
        "results_csv": str(results_csv),
        "results_json": str(results_json),
        "report_md": str(report_md),
        "manifest_json": str(manifest_json),
    }
