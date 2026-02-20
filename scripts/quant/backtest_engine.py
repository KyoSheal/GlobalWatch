#!/usr/bin/env python3
"""A4-2: deterministic offline backtest engine core."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RET_COLUMNS = ["date", "ticker", "ret"]
WEIGHT_COLUMNS = ["date", "ticker", "weight"]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


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
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({c: row.get(c, "") for c in columns})
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_returns(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load returns_daily.csv from cache dir."""
    path = Path(cache_dir).resolve() / "returns_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"returns file missing: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            d = str(row.get("date") or "").strip()
            t = str(row.get("ticker") or "").strip().upper()
            rv = _num_or_none(row.get("ret"))
            if not d or not t or rv is None:
                continue
            rows.append({"date": d, "ticker": t, "ret": float(rv)})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def load_weights(weights_path: Path) -> List[Dict[str, Any]]:
    """Load weights from weights.csv or weights.json."""
    p = Path(weights_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"weights file missing: {p}")
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            for d, wmap in obj.items():
                if not isinstance(wmap, dict):
                    continue
                for t, w in wmap.items():
                    ww = _num_or_none(w)
                    tt = str(t or "").strip().upper()
                    if ww is None or not tt:
                        continue
                    rows.append({"date": str(d), "ticker": tt, "weight": float(ww)})
        elif isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                d = str(item.get("date") or "").strip()
                tt = str(item.get("ticker") or "").strip().upper()
                ww = _num_or_none(item.get("weight"))
                if not d or not tt or ww is None:
                    continue
                rows.append({"date": d, "ticker": tt, "weight": float(ww)})
    else:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                d = str(row.get("date") or "").strip()
                tt = str(row.get("ticker") or "").strip().upper()
                ww = _num_or_none(row.get("weight"))
                if not d or not tt or ww is None:
                    continue
                rows.append({"date": d, "ticker": tt, "weight": float(ww)})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def _normalize_target(raw: Dict[str, float], warnings: List[str], date_key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, w in raw.items():
        tt = str(t or "").strip().upper()
        ww = _num_or_none(w)
        if not tt or ww is None:
            continue
        out[tt] = max(0.0, float(ww))

    if not out:
        return {"CASH": 1.0}

    non_cash = sum(v for k, v in out.items() if k != "CASH")
    cash_present = "CASH" in out
    if cash_present:
        total = sum(out.values())
        if total <= 0:
            return {"CASH": 1.0}
        for k in list(out.keys()):
            out[k] = float(out[k] / total)
        return out

    if non_cash <= 1.0:
        out["CASH"] = float(1.0 - non_cash)
        return out

    if non_cash > 0:
        warnings.append(f"weights_scaled_to_1:{date_key}")
        scale = 1.0 / non_cash
        for k in list(out.keys()):
            out[k] = float(out[k] * scale)
        out["CASH"] = 0.0
    return out


def _build_target_by_date(weight_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in weight_rows:
        d = str(row.get("date") or "").strip()
        t = str(row.get("ticker") or "").strip().upper()
        w = _num_or_none(row.get("weight"))
        if not d or not t or w is None:
            continue
        grouped.setdefault(d, {})[t] = float(w)

    warnings: List[str] = []
    out: Dict[str, Dict[str, float]] = {}
    for d in sorted(grouped.keys()):
        out[d] = _normalize_target(grouped[d], warnings, d)
    return out, warnings


def _latest_target_leq(targets: Dict[str, Dict[str, float]], date_key: str) -> Dict[str, float]:
    keys = [d for d in targets.keys() if d <= date_key]
    if not keys:
        return {"CASH": 1.0}
    latest = sorted(keys)[-1]
    return dict(targets[latest])


def _should_rebalance(date_key: str, last_rebal: Optional[str], rule: str) -> bool:
    if last_rebal is None:
        return True
    rule_n = str(rule or "daily").strip().lower()
    if rule_n == "daily":
        return True
    d = datetime.strptime(date_key, "%Y-%m-%d").date()
    p = datetime.strptime(last_rebal, "%Y-%m-%d").date()
    if rule_n == "weekly":
        return (d.isocalendar()[0], d.isocalendar()[1]) != (p.isocalendar()[0], p.isocalendar()[1])
    if rule_n == "monthly":
        return (d.year, d.month) != (p.year, p.month)
    return True


def _rebalance(
    *,
    date_key: str,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
    cost_rate: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    all_non_cash = sorted(set([k for k in current_weights.keys() if k != "CASH"] + [k for k in target_weights.keys() if k != "CASH"]))
    deltas: Dict[str, float] = {}
    for t in all_non_cash:
        deltas[t] = float(target_weights.get(t, 0.0) - current_weights.get(t, 0.0))

    turnover_w = float(sum(abs(v) for v in deltas.values()))
    turnover_notional = float(turnover_w * equity)
    cash_before_notional = float(current_weights.get("CASH", 0.0) * equity)
    cost = float(turnover_notional * cost_rate)
    scale = 1.0
    notes: List[str] = []

    if cost > cash_before_notional and turnover_notional > 0:
        scale = float(max(0.0, min(1.0, cash_before_notional / cost)))
        notes.append("scaled_due_to_cash_for_cost")

    applied_target = dict(current_weights)
    if scale < 1.0:
        for t in all_non_cash:
            applied_target[t] = float(current_weights.get(t, 0.0) + deltas[t] * scale)
            if applied_target[t] < 0:
                applied_target[t] = 0.0
    else:
        for t in all_non_cash:
            applied_target[t] = float(target_weights.get(t, 0.0))

    non_cash_sum = float(sum(applied_target.get(t, 0.0) for t in all_non_cash))
    if non_cash_sum > 1.0:
        for t in all_non_cash:
            applied_target[t] = float(applied_target[t] / non_cash_sum)
        non_cash_sum = 1.0
        notes.append("renorm_non_cash_after_rebalance")
    applied_target["CASH"] = float(max(0.0, 1.0 - non_cash_sum))

    applied_deltas = {t: float(applied_target.get(t, 0.0) - current_weights.get(t, 0.0)) for t in all_non_cash}
    turnover_w = float(sum(abs(v) for v in applied_deltas.values()))
    turnover_notional = float(turnover_w * equity)
    cost = float(turnover_notional * cost_rate)
    cash_before_notional = float(current_weights.get("CASH", 0.0) * equity)
    if turnover_notional > 0 and cost > cash_before_notional:
        cost = cash_before_notional
        notes.append("cost_clamped_to_cash")

    notionals = {k: float(v * equity) for k, v in applied_target.items()}
    notionals["CASH"] = float(notionals.get("CASH", 0.0) - cost)
    equity_after_cost = float(max(1e-12, equity - cost))

    new_weights: Dict[str, float] = {}
    for k, n in notionals.items():
        new_weights[k] = float(max(0.0, n) / equity_after_cost)
    total_w = sum(new_weights.values())
    if total_w > 0:
        for k in list(new_weights.keys()):
            new_weights[k] = float(new_weights[k] / total_w)
    else:
        new_weights = {"CASH": 1.0}

    trades: List[Dict[str, Any]] = []
    for t in all_non_cash:
        dw = float(applied_deltas.get(t, 0.0))
        if abs(dw) <= 1e-12:
            continue
        trade_notional = float(abs(dw) * equity)
        trade_cost = float(cost * (abs(dw) / turnover_w)) if turnover_w > 0 else 0.0
        trades.append(
            {
                "date": date_key,
                "ticker": t,
                "delta_w": dw,
                "trade_notional": trade_notional,
                "cost": trade_cost,
                "post_weight": float(new_weights.get(t, 0.0)),
                "side": "BUY" if dw > 0 else "SELL",
            }
        )
    trades.sort(key=lambda r: (str(r["ticker"]), str(r["side"])))

    meta = {
        "turnover_w": turnover_w,
        "turnover_notional": turnover_notional,
        "cost": cost,
        "scale": scale,
        "notes": notes,
    }
    return new_weights, trades, meta


def _apply_daily_returns(
    *,
    current_weights: Dict[str, float],
    equity: float,
    ret_map: Dict[str, float],
) -> Tuple[Dict[str, float], float, float, List[str]]:
    notionals: Dict[str, float] = {}
    missing: List[str] = []
    for t, w in current_weights.items():
        ww = float(w)
        if t == "CASH":
            notionals[t] = float(ww * equity)
            continue
        r = _num_or_none(ret_map.get(t))
        if r is None:
            r = 0.0
            if ww > 0:
                missing.append(t)
        notionals[t] = float(ww * equity * (1.0 + float(r)))

    new_equity = float(sum(notionals.values()))
    if new_equity <= 0:
        return {"CASH": 1.0}, 0.0, 0.0, missing

    new_weights: Dict[str, float] = {}
    for t, n in notionals.items():
        new_weights[t] = float(max(0.0, n) / new_equity)
    total_w = sum(new_weights.values())
    if total_w > 0:
        for t in list(new_weights.keys()):
            new_weights[t] = float(new_weights[t] / total_w)

    portfolio_ret = float((new_equity / equity) - 1.0) if equity > 0 else 0.0
    return new_weights, new_equity, portfolio_ret, sorted(set(missing))


def run_backtest(
    returns_df: List[Dict[str, Any]],
    weights_df: List[Dict[str, Any]],
    *,
    initial_equity: float,
    cost_bps: float,
    rebalance_rule: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run deterministic backtest simulation."""
    if float(initial_equity) <= 0:
        raise ValueError("initial_equity must be > 0")
    cost_rate = float(cost_bps) / 10000.0
    returns_rows = []
    for row in returns_df:
        d = str(row.get("date") or "").strip()
        t = str(row.get("ticker") or "").strip().upper()
        r = _num_or_none(row.get("ret"))
        if not d or not t or r is None:
            continue
        returns_rows.append({"date": d, "ticker": t, "ret": float(r)})
    returns_rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))

    target_by_date, weight_warnings = _build_target_by_date(weights_df)
    returns_by_date: Dict[str, Dict[str, float]] = {}
    for row in returns_rows:
        returns_by_date.setdefault(str(row["date"]), {})[str(row["ticker"])] = float(row["ret"])
    dates = sorted(returns_by_date.keys())
    if not dates:
        raise ValueError("no returns rows to simulate")

    current_weights = {"CASH": 1.0}
    active_target = _latest_target_leq(target_by_date, dates[0])
    equity = float(initial_equity)
    peak = float(equity)
    last_rebal_date: Optional[str] = None

    equity_rows: List[Dict[str, Any]] = []
    trades_rows: List[Dict[str, Any]] = []
    warnings: List[str] = list(weight_warnings)
    total_turnover = 0.0
    total_cost = 0.0
    rebalance_count = 0
    missing_ret_count = 0

    for d in dates:
        if d in target_by_date:
            active_target = dict(target_by_date[d])

        if _should_rebalance(d, last_rebal_date, rebalance_rule):
            current_weights, trades, meta = _rebalance(
                date_key=d,
                current_weights=current_weights,
                target_weights=active_target,
                equity=equity,
                cost_rate=cost_rate,
            )
            for tr in trades:
                trades_rows.append(tr)
            total_turnover += float(meta.get("turnover_notional", 0.0) or 0.0)
            total_cost += float(meta.get("cost", 0.0) or 0.0)
            if meta.get("notes"):
                for note in meta["notes"]:
                    warnings.append(f"{d}:{note}")
            equity = float(max(1e-12, equity - float(meta.get("cost", 0.0) or 0.0)))
            rebalance_count += 1
            last_rebal_date = d

        current_weights, equity, day_ret, missing = _apply_daily_returns(
            current_weights=current_weights,
            equity=equity,
            ret_map=returns_by_date.get(d, {}),
        )
        if missing:
            missing_ret_count += len(missing)
            warnings.append(f"{d}:missing_returns:{','.join(missing)}")

        peak = max(peak, equity)
        dd = float((equity / peak) - 1.0) if peak > 0 else 0.0
        equity_rows.append(
            {
                "date": d,
                "equity": equity,
                "ret": day_ret,
                "drawdown": dd,
            }
        )

    equity_rows.sort(key=lambda r: str(r["date"]))
    trades_rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"]), str(r["side"])))

    max_dd = min([float(r["drawdown"]) for r in equity_rows]) if equity_rows else 0.0
    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "params": {
            "initial_equity": float(initial_equity),
            "cost_bps": float(cost_bps),
            "rebalance_rule": str(rebalance_rule),
        },
        "cost_summary": {
            "total_turnover_notional": float(total_turnover),
            "total_cost": float(total_cost),
            "trade_rows": len(trades_rows),
            "rebalance_count": int(rebalance_count),
            "missing_return_points": int(missing_ret_count),
        },
        "summary": {
            "start_equity": float(equity_rows[0]["equity"]) / float(1.0 + equity_rows[0]["ret"]) if equity_rows else float(initial_equity),
            "end_equity": float(equity_rows[-1]["equity"]) if equity_rows else float(initial_equity),
            "total_return": float((equity_rows[-1]["equity"] / initial_equity) - 1.0) if equity_rows else 0.0,
            "max_drawdown": float(max_dd),
            "days": len(equity_rows),
        },
        "warnings": sorted(set(warnings)),
    }
    return equity_rows, trades_rows, manifest


def write_backtest(
    out_dir: Path,
    equity_rows: List[Dict[str, Any]],
    trades_rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, str]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    eq_path = out / "backtest_equity.csv"
    tr_path = out / "backtest_trades.csv"
    mf_path = out / "backtest_manifest.json"
    md_path = out / "backtest_report.md"

    eq_rows_out = []
    for r in equity_rows:
        eq_rows_out.append(
            {
                "date": str(r.get("date", "")),
                "equity": f"{float(r.get('equity', 0.0)):.10f}",
                "ret": f"{float(r.get('ret', 0.0)):.10f}",
                "drawdown": f"{float(r.get('drawdown', 0.0)):.10f}",
            }
        )
    tr_rows_out = []
    for r in trades_rows:
        tr_rows_out.append(
            {
                "date": str(r.get("date", "")),
                "ticker": str(r.get("ticker", "")),
                "delta_w": f"{float(r.get('delta_w', 0.0)):.10f}",
                "trade_notional": f"{float(r.get('trade_notional', 0.0)):.10f}",
                "cost": f"{float(r.get('cost', 0.0)):.10f}",
                "post_weight": f"{float(r.get('post_weight', 0.0)):.10f}",
                "side": str(r.get("side", "")),
            }
        )

    _write_csv(eq_path, eq_rows_out, ["date", "equity", "ret", "drawdown"])
    _write_csv(tr_path, tr_rows_out, ["date", "ticker", "delta_w", "trade_notional", "cost", "post_weight", "side"])
    _write_json_atomic(mf_path, manifest)

    summary = manifest.get("summary", {}) if isinstance(manifest.get("summary"), dict) else {}
    cost_summary = manifest.get("cost_summary", {}) if isinstance(manifest.get("cost_summary"), dict) else {}
    report_lines = [
        "# Backtest Report",
        "",
        f"- generated_utc: `{manifest.get('generated_utc', '')}`",
        f"- start_equity: `{float(summary.get('start_equity', 0.0)):.2f}`",
        f"- end_equity: `{float(summary.get('end_equity', 0.0)):.2f}`",
        f"- total_return: `{float(summary.get('total_return', 0.0)) * 100.0:.2f}%`",
        f"- max_drawdown: `{float(summary.get('max_drawdown', 0.0)) * 100.0:.2f}%`",
        f"- days: `{int(summary.get('days', 0) or 0)}`",
        "",
        "## Trading / Cost",
        f"- trade_rows: `{int(cost_summary.get('trade_rows', 0) or 0)}`",
        f"- rebalance_count: `{int(cost_summary.get('rebalance_count', 0) or 0)}`",
        f"- total_turnover_notional: `{float(cost_summary.get('total_turnover_notional', 0.0)):.2f}`",
        f"- total_cost: `{float(cost_summary.get('total_cost', 0.0)):.2f}`",
        f"- missing_return_points: `{int(cost_summary.get('missing_return_points', 0) or 0)}`",
        "",
        "## Files",
        "- `backtest_equity.csv`",
        "- `backtest_trades.csv`",
        "- `backtest_manifest.json`",
        "",
    ]
    _write_text_atomic(md_path, "\n".join(report_lines))

    return {
        "equity_csv": str(eq_path),
        "trades_csv": str(tr_path),
        "manifest_json": str(mf_path),
        "report_md": str(md_path),
    }
