#!/usr/bin/env python3
"""Pure metrics helpers for A1-2 run dataset analytics."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import parse_iso_to_utc, safe_read_json, to_iso_utc

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _safe_std(values: List[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    if var < 0:
        return None
    return math.sqrt(var)


def _tzinfo_or_utc(name: str):
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(str(name or "UTC"))
    except Exception:
        return timezone.utc


def _read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                bad += 1
                continue
            rows.append(dict(row))
    return rows, bad


def load_dataset(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    equity_path = dataset_dir / "equity_curve.csv"
    cycles_path = dataset_dir / "cycles.csv"
    trades_path = dataset_dir / "trades.csv"

    missing_files: List[str] = []
    manifest = safe_read_json(manifest_path) or {}
    if not isinstance(manifest, dict):
        manifest = {}
        missing_files.append(str(manifest_path))
    for p in (equity_path, cycles_path, trades_path):
        if not p.exists():
            missing_files.append(str(p))

    equity_rows_raw, equity_bad_csv = _read_csv_dicts(equity_path)
    cycles_rows_raw, cycles_bad_csv = _read_csv_dicts(cycles_path)
    trades_rows_raw, trades_bad_csv = _read_csv_dicts(trades_path)

    parse_warnings: Dict[str, int] = {
        "equity_bad_rows": 0,
        "cycles_bad_rows": 0,
        "trades_bad_rows": 0,
        "equity_bad_csv": int(equity_bad_csv),
        "cycles_bad_csv": int(cycles_bad_csv),
        "trades_bad_csv": int(trades_bad_csv),
    }

    equity_rows: List[Dict[str, Any]] = []
    for row in equity_rows_raw:
        dt = parse_iso_to_utc(row.get("time_utc"))
        if dt is None:
            parse_warnings["equity_bad_rows"] += 1
            continue
        eq = _num_or_none(row.get("equity"))
        if eq is None:
            eq = _num_or_none(row.get("total_equity"))
        if eq is None:
            parse_warnings["equity_bad_rows"] += 1
            continue
        equity_rows.append(
            {
                "dt_utc": dt,
                "time_utc": to_iso_utc(dt),
                "equity": float(eq),
                "cash": _num_or_none(row.get("cash")),
                "positions_value": _num_or_none(row.get("positions_value")),
            }
        )

    # deterministic order + de-dupe by timestamp (keep last)
    equity_rows.sort(key=lambda r: r["time_utc"])
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in equity_rows:
        dedup[row["time_utc"]] = row
    equity_rows = [dedup[k] for k in sorted(dedup.keys())]

    cycles_rows: List[Dict[str, Any]] = []
    for row in cycles_rows_raw:
        dt = parse_iso_to_utc(row.get("time_utc"))
        if dt is None:
            parse_warnings["cycles_bad_rows"] += 1
            continue
        cycle_id = row.get("cycle_id")
        try:
            cycle_id_int = int(cycle_id) if cycle_id not in (None, "") else None
        except Exception:
            cycle_id_int = None
        cycles_rows.append(
            {
                "time_utc": to_iso_utc(dt),
                "cycle_id": cycle_id_int,
                "skip_reason": str(row.get("skip_reason", "") or "").strip(),
                "decision_path": str(row.get("decision_path", "") or "").strip(),
                "cov_gate_reason": str(row.get("cov_gate_reason", "") or "").strip(),
            }
        )
    cycles_rows.sort(key=lambda r: (r["time_utc"], int(r["cycle_id"] or 0)))

    trades_rows: List[Dict[str, Any]] = []
    for row in trades_rows_raw:
        dt = parse_iso_to_utc(row.get("time_utc"))
        if dt is None:
            parse_warnings["trades_bad_rows"] += 1
            continue
        side = str(row.get("side", "") or "").strip().upper()
        ticker = str(row.get("ticker", "") or "").strip().upper()
        notional = _num_or_none(row.get("notional"))
        if notional is None:
            notional = _num_or_none(row.get("value"))
        qty = _num_or_none(row.get("qty"))
        price = _num_or_none(row.get("price"))
        if notional is None and qty is not None and price is not None:
            notional = qty * price
        trades_rows.append(
            {
                "time_utc": to_iso_utc(dt),
                "ticker": ticker,
                "side": side,
                "notional": notional,
            }
        )
    trades_rows.sort(key=lambda r: (r["time_utc"], r["ticker"]))

    return {
        "manifest": manifest,
        "equity_rows": equity_rows,
        "cycles_rows": cycles_rows,
        "trades_rows": trades_rows,
        "missing_files": missing_files,
        "parse_warnings": parse_warnings,
    }


def _compute_period_returns(equity_rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    rets: List[float] = []
    deltas_sec: List[float] = []
    prev = None
    for row in equity_rows:
        if prev is not None:
            prev_eq = float(prev["equity"])
            cur_eq = float(row["equity"])
            if prev_eq != 0:
                rets.append((cur_eq / prev_eq) - 1.0)
            dt_prev = prev["dt_utc"]
            dt_cur = row["dt_utc"]
            delta = (dt_cur - dt_prev).total_seconds()
            if delta > 0:
                deltas_sec.append(delta)
        prev = row
    return rets, deltas_sec


def _compute_drawdown(equity_rows: List[Dict[str, Any]]) -> float:
    max_dd = 0.0
    peak = None
    for row in equity_rows:
        eq = float(row["equity"])
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = (eq / peak) - 1.0
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _compute_daily_returns(
    equity_rows: List[Dict[str, Any]],
    report_tz: str,
) -> List[Dict[str, Any]]:
    if not equity_rows:
        return []
    tzinfo = _tzinfo_or_utc(report_tz)
    by_date: Dict[str, Dict[str, Any]] = {}
    for row in equity_rows:
        dt_local = row["dt_utc"].astimezone(tzinfo)
        key = dt_local.date().isoformat()
        by_date[key] = row  # keep last of that date due to sorted rows
    ordered_dates = sorted(by_date.keys())
    out: List[Dict[str, Any]] = []
    prev_close = None
    for d in ordered_dates:
        close_eq = float(by_date[d]["equity"])
        daily_ret = None
        if prev_close is not None and prev_close != 0:
            daily_ret = (close_eq / prev_close) - 1.0
        out.append(
            {
                "date_local": d,
                "close_equity": close_eq,
                "daily_return": daily_ret,
            }
        )
        prev_close = close_eq
    return out


def _infer_periods_per_year(
    daily_returns: List[Dict[str, Any]],
    deltas_sec: List[float],
    annualization: int,
) -> Tuple[Optional[float], str]:
    daily_values = [float(r["daily_return"]) for r in daily_returns if r.get("daily_return") is not None]
    if len(daily_values) >= 2:
        return float(annualization), "daily"
    if len(deltas_sec) >= 2:
        avg_delta = sum(deltas_sec) / len(deltas_sec)
        if avg_delta > 0:
            return (365.25 * 24.0 * 3600.0) / avg_delta, "period"
    return None, "none"


def _compute_risk_metrics(
    rets: List[float],
    daily_returns: List[Dict[str, Any]],
    deltas_sec: List[float],
    annualization: int,
    rf_annual: float,
) -> Dict[str, Any]:
    daily_values = [float(r["daily_return"]) for r in daily_returns if r.get("daily_return") is not None]
    periods_per_year, basis = _infer_periods_per_year(daily_returns, deltas_sec, annualization)
    returns_for_risk = daily_values if basis == "daily" else list(rets)

    vol_ann = None
    sharpe = None
    sortino = None
    if periods_per_year and len(returns_for_risk) >= 2:
        std = _safe_std(returns_for_risk)
        if std and std > 0:
            vol_ann = std * math.sqrt(periods_per_year)
            rf_period = (1.0 + float(rf_annual)) ** (1.0 / periods_per_year) - 1.0
            excess = [r - rf_period for r in returns_for_risk]
            mu_excess = mean(excess)
            sharpe = (mu_excess / std) * math.sqrt(periods_per_year)

            downside = [min(0.0, r - rf_period) for r in returns_for_risk]
            down_std = _safe_std(downside)
            if down_std and down_std > 0:
                sortino = (mu_excess / down_std) * math.sqrt(periods_per_year)

    return {
        "vol_annualized": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "risk_basis": basis,
        "periods_per_year": periods_per_year,
    }


def _summarize_gating(cycles_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counts: Dict[str, int] = {}
    mapped = {
        "attempt_cooldown": 0,
        "market_closed": 0,
        "stale_abort": 0,
        "risk_abort": 0,
    }

    for row in cycles_rows:
        reason = (
            str(row.get("skip_reason", "") or "").strip()
            or str(row.get("cov_gate_reason", "") or "").strip()
            or str(row.get("decision_path", "") or "").strip()
        )
        if not reason:
            continue
        k = reason.lower()
        reason_counts[k] = reason_counts.get(k, 0) + 1
        if "attempt_cooldown" in k:
            mapped["attempt_cooldown"] += 1
        if "market_closed" in k:
            mapped["market_closed"] += 1
        if "stale_abort" in k:
            mapped["stale_abort"] += 1
        if ("risk" in k) or ("portfolio_cov_rc_limit" in k) or ("cov_rc" in k):
            mapped["risk_abort"] += 1

    top3 = sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))[:3]
    return {
        "counts": reason_counts,
        "attempt_cooldown": mapped["attempt_cooldown"],
        "market_closed": mapped["market_closed"],
        "stale_abort": mapped["stale_abort"],
        "risk_abort": mapped["risk_abort"],
        "top3": [{"reason": r, "count": c} for r, c in top3],
    }


def compute_metrics(
    dataset: Dict[str, Any],
    *,
    dataset_dir: Path,
    report_tz: str,
    annualization: int,
    rf_annual: float,
    min_points: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest = dataset.get("manifest", {}) or {}
    equity_rows = dataset.get("equity_rows", []) or []
    cycles_rows = dataset.get("cycles_rows", []) or []
    trades_rows = dataset.get("trades_rows", []) or []
    missing_files = dataset.get("missing_files", []) or []
    parse_warnings = dataset.get("parse_warnings", {}) or {}

    run_id = str(manifest.get("run_id", "") or "").strip() or "unknown_run"
    equity_points = len(equity_rows)
    trade_rows_n = len(trades_rows)
    cycle_rows_n = len(cycles_rows)

    start_equity = equity_rows[0]["equity"] if equity_rows else None
    end_equity = equity_rows[-1]["equity"] if equity_rows else None
    total_return = None
    if start_equity not in (None, 0) and end_equity is not None:
        total_return = (float(end_equity) / float(start_equity)) - 1.0

    rets, deltas_sec = _compute_period_returns(equity_rows)
    best_period = max(rets) if rets else None
    worst_period = min(rets) if rets else None
    max_dd = _compute_drawdown(equity_rows) if equity_rows else None

    span_days = None
    cagr = None
    if len(equity_rows) >= 2:
        span_days = (equity_rows[-1]["dt_utc"] - equity_rows[0]["dt_utc"]).total_seconds() / 86400.0
        if span_days is not None and span_days >= 2 and start_equity not in (None, 0) and end_equity is not None:
            cagr = (float(end_equity) / float(start_equity)) ** (365.25 / span_days) - 1.0

    daily_returns = _compute_daily_returns(equity_rows, report_tz=report_tz)
    daily_values = [float(r["daily_return"]) for r in daily_returns if r.get("daily_return") is not None]
    daily_mean = mean(daily_values) if daily_values else None
    daily_std = _safe_std(daily_values) if len(daily_values) >= 2 else None

    risk_block = _compute_risk_metrics(
        rets,
        daily_returns,
        deltas_sec,
        annualization=annualization,
        rf_annual=rf_annual,
    )
    calmar = None
    if cagr is not None and max_dd is not None and max_dd < 0:
        calmar = cagr / abs(max_dd)

    buys = sum(1 for r in trades_rows if str(r.get("side", "")).upper() == "BUY")
    sells = sum(1 for r in trades_rows if str(r.get("side", "")).upper() == "SELL")
    tickers = sorted({str(r.get("ticker", "")).strip().upper() for r in trades_rows if str(r.get("ticker", "")).strip()})
    turnover_notional = None
    notionals = [abs(float(r["notional"])) for r in trades_rows if r.get("notional") is not None]
    if notionals:
        turnover_notional = float(sum(notionals))
    avg_equity = mean([float(r["equity"]) for r in equity_rows]) if equity_rows else None
    turnover_ratio = None
    if turnover_notional is not None and avg_equity not in (None, 0):
        turnover_ratio = turnover_notional / float(avg_equity)

    gating = _summarize_gating(cycles_rows)
    insufficient_points = equity_points < int(min_points)

    metrics: Dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
            "report_tz": report_tz,
            "annualization": int(annualization),
            "equity_points": equity_points,
            "trade_rows": trade_rows_n,
            "cycle_rows": cycle_rows_n,
        },
        "performance": {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "total_return": total_return,
            "cagr": cagr,
            "best_period_return": best_period,
            "worst_period_return": worst_period,
            "daily_return_mean": daily_mean,
            "daily_return_std": daily_std,
        },
        "risk": {
            "vol_annualized": risk_block.get("vol_annualized"),
            "sharpe": risk_block.get("sharpe"),
            "sortino": risk_block.get("sortino"),
            "max_drawdown": max_dd,
            "calmar": calmar,
            "risk_basis": risk_block.get("risk_basis"),
            "periods_per_year": risk_block.get("periods_per_year"),
        },
        "trading": {
            "trades_total": trade_rows_n,
            "buys": buys,
            "sells": sells,
            "unique_tickers": len(tickers),
            "unique_ticker_list": tickers,
            "turnover_notional": turnover_notional,
            "turnover_ratio": turnover_ratio,
        },
        "gating": {
            "summary": gating,
        },
        "data_quality": {
            "insufficient_points": bool(insufficient_points),
            "missing_files": sorted(str(x) for x in missing_files),
            "parse_warnings": parse_warnings,
        },
    }
    return metrics, daily_returns


def render_metrics_markdown(metrics: Dict[str, Any]) -> str:
    meta = metrics.get("meta", {}) or {}
    perf = metrics.get("performance", {}) or {}
    risk = metrics.get("risk", {}) or {}
    trading = metrics.get("trading", {}) or {}
    gating = (metrics.get("gating", {}) or {}).get("summary", {}) or {}
    dq = metrics.get("data_quality", {}) or {}

    def _pct(x: Any) -> str:
        try:
            if x is None:
                return "-"
            return f"{float(x) * 100.0:.2f}%"
        except Exception:
            return "-"

    def _num(x: Any) -> str:
        try:
            if x is None:
                return "-"
            return f"{float(x):,.4f}"
        except Exception:
            return "-"

    lines: List[str] = []
    lines.append("# Quant Metrics Summary")
    lines.append("")
    lines.append(f"- Run ID: `{meta.get('run_id', 'unknown_run')}`")
    lines.append(f"- Generated (UTC): `{meta.get('generated_at_utc', '-')}`")
    lines.append(f"- Points: equity={meta.get('equity_points', 0)} cycles={meta.get('cycle_rows', 0)} trades={meta.get('trade_rows', 0)}")
    lines.append("")
    lines.append("## Performance")
    lines.append(f"- Total Return: {_pct(perf.get('total_return'))}")
    lines.append(f"- Start Equity: {_num(perf.get('start_equity'))}")
    lines.append(f"- End Equity: {_num(perf.get('end_equity'))}")
    lines.append(f"- CAGR: {_pct(perf.get('cagr'))}")
    lines.append("")
    lines.append("## Risk")
    lines.append(f"- Max DD: {_pct(risk.get('max_drawdown'))}")
    lines.append(f"- Vol (ann.): {_pct(risk.get('vol_annualized'))}")
    lines.append(f"- Sharpe: {_num(risk.get('sharpe'))}")
    lines.append(f"- Sortino: {_num(risk.get('sortino'))}")
    lines.append("")
    lines.append("## Trading")
    lines.append(f"- Trades: {trading.get('trades_total', 0)} (BUY={trading.get('buys', 0)} SELL={trading.get('sells', 0)})")
    lines.append(f"- Unique Tickers: {trading.get('unique_tickers', 0)}")
    lines.append(f"- Turnover Notional: {_num(trading.get('turnover_notional'))}")
    lines.append(f"- Turnover Ratio: {_pct(trading.get('turnover_ratio'))}")
    lines.append("")
    lines.append("## Gating Summary")
    top3 = gating.get("top3", []) or []
    if top3:
        for item in top3:
            lines.append(f"- {item.get('reason', '-')}: {item.get('count', 0)}")
    else:
        lines.append("- No gating reasons captured")
    lines.append("")
    lines.append("## Data Quality")
    lines.append(f"- insufficient_points: {bool(dq.get('insufficient_points', False))}")
    missing_files = dq.get("missing_files", []) or []
    if missing_files:
        lines.append("- missing_files:")
        for m in missing_files:
            lines.append(f"  - {m}")
    else:
        lines.append("- missing_files: none")
    pw = dq.get("parse_warnings", {}) or {}
    lines.append(
        "- parse_warnings: "
        f"equity_bad_rows={pw.get('equity_bad_rows', 0)}, "
        f"cycles_bad_rows={pw.get('cycles_bad_rows', 0)}, "
        f"trades_bad_rows={pw.get('trades_bad_rows', 0)}"
    )
    lines.append("")
    return "\n".join(lines)

