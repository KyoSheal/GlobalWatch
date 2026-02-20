#!/usr/bin/env python3
"""Run-to-run compare helpers for A1-3."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import to_iso_utc
from quant_metrics import compute_metrics, load_dataset


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
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


def _safe_corr(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    mu_a = sum(a) / len(a)
    mu_b = sum(b) / len(b)
    var_a = sum((x - mu_a) ** 2 for x in a)
    var_b = sum((y - mu_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    cov = sum((a[i] - mu_a) * (b[i] - mu_b) for i in range(len(a)))
    return cov / math.sqrt(var_a * var_b)


def _read_daily_returns_csv(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                d = str(row.get("date_local", "") or "").strip()
                if not d:
                    continue
                close = _num_or_none(row.get("close_equity"))
                r = _num_or_none(row.get("daily_return"))
                out.append(
                    {
                        "date_local": d,
                        "close_equity": close,
                        "daily_return": r,
                    }
                )
        out.sort(key=lambda x: x["date_local"])
        return out
    except Exception:
        return None


def _daily_returns_from_equity(
    dataset_dir: Path,
    *,
    report_tz: str,
    annualization: int,
    rf: float,
) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_dir)
    _, daily = compute_metrics(
        dataset,
        dataset_dir=dataset_dir,
        report_tz=report_tz,
        annualization=annualization,
        rf_annual=rf,
        min_points=1,
    )
    return daily


def load_metrics_and_daily(
    dataset_dir: Path,
    *,
    report_tz: str,
    annualization: int,
    rf: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    metrics_path = dataset_dir / "metrics" / "metrics.json"
    daily_path = dataset_dir / "metrics" / "daily_returns.csv"

    quality = {
        "metrics_source": "file",
        "daily_source": "file",
        "missing_metrics_file": False,
        "missing_daily_file": False,
    }

    metrics = _safe_read_json(metrics_path)
    if not isinstance(metrics, dict):
        quality["metrics_source"] = "computed_in_memory"
        quality["missing_metrics_file"] = True
        dataset = load_dataset(dataset_dir)
        metrics, _ = compute_metrics(
            dataset,
            dataset_dir=dataset_dir,
            report_tz=report_tz,
            annualization=annualization,
            rf_annual=rf,
            min_points=1,
        )

    daily = _read_daily_returns_csv(daily_path)
    if daily is None:
        quality["daily_source"] = "computed_from_equity"
        quality["missing_daily_file"] = True
        daily = _daily_returns_from_equity(
            dataset_dir,
            report_tz=report_tz,
            annualization=annualization,
            rf=rf,
        )

    return metrics, daily, quality


def _build_daily_compare(
    daily_a: List[Dict[str, Any]],
    daily_b: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    map_a = {str(r.get("date_local", "")): _num_or_none(r.get("daily_return")) for r in daily_a}
    map_b = {str(r.get("date_local", "")): _num_or_none(r.get("daily_return")) for r in daily_b}
    overlap = sorted(d for d in set(map_a.keys()).intersection(set(map_b.keys())) if d)

    rows: List[Dict[str, Any]] = []
    ra: List[float] = []
    rb: List[float] = []
    deltas: List[float] = []
    for d in overlap:
        a = map_a.get(d)
        b = map_b.get(d)
        if a is None or b is None:
            continue
        delta = float(b) - float(a)
        rows.append(
            {
                "date_local": d,
                "daily_return_a": float(a),
                "daily_return_b": float(b),
                "delta_b_minus_a": float(delta),
            }
        )
        ra.append(float(a))
        rb.append(float(b))
        deltas.append(float(delta))

    overlap_days = len(rows)
    corr = _safe_corr(ra, rb) if overlap_days >= 2 else None
    te = _safe_std(deltas) if overlap_days >= 2 else None
    hit_rate = (sum(1 for d in deltas if d > 0) / overlap_days) if overlap_days > 0 else None

    overlap_return_a = None
    overlap_return_b = None
    if overlap_days > 0:
        p_a = 1.0
        p_b = 1.0
        for x in ra:
            p_a *= 1.0 + x
        for y in rb:
            p_b *= 1.0 + y
        overlap_return_a = p_a - 1.0
        overlap_return_b = p_b - 1.0

    summary = {
        "overlap_days": overlap_days,
        "corr": corr,
        "tracking_error_std": te,
        "hit_rate_b_gt_a": hit_rate,
        "mean_delta_b_minus_a": mean(deltas) if deltas else None,
        "overlap_return_a": overlap_return_a,
        "overlap_return_b": overlap_return_b,
    }
    return rows, summary


def _gating_counts(metrics: Dict[str, Any]) -> Dict[str, int]:
    g = (metrics.get("gating") or {}).get("summary") or {}
    counts = g.get("counts")
    out: Dict[str, int] = {}
    if isinstance(counts, dict):
        for k, v in counts.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
    return out


def _gating_delta_top5(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> List[Dict[str, Any]]:
    ca = _gating_counts(metrics_a)
    cb = _gating_counts(metrics_b)
    keys = sorted(set(ca.keys()).union(set(cb.keys())))
    diffs: List[Dict[str, Any]] = []
    for k in keys:
        a = int(ca.get(k, 0))
        b = int(cb.get(k, 0))
        dif = b - a
        if dif == 0:
            continue
        diffs.append(
            {
                "reason": k,
                "count_a": a,
                "count_b": b,
                "delta_b_minus_a": dif,
                "abs_delta": abs(dif),
            }
        )
    diffs.sort(key=lambda x: (-int(x["abs_delta"]), str(x["reason"])))
    return diffs[:5]


def _delta(a: Any, b: Any) -> Optional[float]:
    aa = _num_or_none(a)
    bb = _num_or_none(b)
    if aa is None or bb is None:
        return None
    return bb - aa


def _winner(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Tuple[str, str]:
    ret_a = _num_or_none(((metrics_a.get("performance") or {}).get("total_return")))
    ret_b = _num_or_none(((metrics_b.get("performance") or {}).get("total_return")))
    dd_a = _num_or_none(((metrics_a.get("risk") or {}).get("max_drawdown")))
    dd_b = _num_or_none(((metrics_b.get("risk") or {}).get("max_drawdown")))

    if ret_a is not None and ret_b is not None:
        if ret_b > ret_a:
            return "B", "higher_total_return"
        if ret_a > ret_b:
            return "A", "higher_total_return"
    if dd_a is not None and dd_b is not None:
        # higher drawdown value is better because drawdown is negative
        if dd_b > dd_a:
            return "B", "better_max_drawdown"
        if dd_a > dd_b:
            return "A", "better_max_drawdown"
    return "TIE", "equal_or_insufficient_data"


def evaluate_fail_rules(compare: Dict[str, Any], rules: List[str]) -> Dict[str, Any]:
    normalized = [str(x or "").strip().lower() for x in rules if str(x or "").strip()]
    out = {
        "enabled": bool(normalized),
        "rules": normalized,
        "failed": [],
        "unknown_rules": [],
        "passed": [],
        "ok": True,
    }
    if not normalized:
        return out

    winner = str((compare.get("headline") or {}).get("winner", ""))
    total_return_delta = _num_or_none((compare.get("delta_metrics") or {}).get("total_return_delta_b_minus_a"))
    max_dd_delta = _num_or_none((compare.get("delta_metrics") or {}).get("max_drawdown_delta_b_minus_a"))

    for rule in normalized:
        if rule == "winner_b":
            passed = winner == "B"
        elif rule == "positive_delta_return":
            passed = (total_return_delta is not None) and (total_return_delta > 0)
        elif rule == "better_drawdown_b":
            passed = (max_dd_delta is not None) and (max_dd_delta > 0)
        else:
            out["unknown_rules"].append(rule)
            continue

        if passed:
            out["passed"].append(rule)
        else:
            out["failed"].append(rule)

    out["ok"] = len(out["failed"]) == 0
    return out


def compare_two_runs(
    *,
    dataset_a: Path,
    dataset_b: Path,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    daily_a: List[Dict[str, Any]],
    daily_b: List[Dict[str, Any]],
    quality_a: Dict[str, Any],
    quality_b: Dict[str, Any],
    report_tz: str,
    annualization: int,
    rf: float,
    fail_rules: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    winner, winner_reason = _winner(metrics_a, metrics_b)
    daily_delta_rows, daily_summary = _build_daily_compare(daily_a, daily_b)
    gating_delta_top5 = _gating_delta_top5(metrics_a, metrics_b)

    meta_a = metrics_a.get("meta", {}) or {}
    meta_b = metrics_b.get("meta", {}) or {}
    perf_a = metrics_a.get("performance", {}) or {}
    perf_b = metrics_b.get("performance", {}) or {}
    risk_a = metrics_a.get("risk", {}) or {}
    risk_b = metrics_b.get("risk", {}) or {}
    tr_a = metrics_a.get("trading", {}) or {}
    tr_b = metrics_b.get("trading", {}) or {}

    compare: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
            "dataset_a": str(dataset_a),
            "dataset_b": str(dataset_b),
            "run_id_a": str(meta_a.get("run_id", "unknown_run")),
            "run_id_b": str(meta_b.get("run_id", "unknown_run")),
            "report_tz": report_tz,
            "annualization": int(annualization),
            "rf": float(rf),
        },
        "headline": {
            "winner": winner,
            "winner_reason": winner_reason,
        },
        "delta_metrics": {
            "total_return_delta_b_minus_a": _delta(perf_a.get("total_return"), perf_b.get("total_return")),
            "max_drawdown_delta_b_minus_a": _delta(risk_a.get("max_drawdown"), risk_b.get("max_drawdown")),
            "vol_annualized_delta_b_minus_a": _delta(risk_a.get("vol_annualized"), risk_b.get("vol_annualized")),
            "sharpe_delta_b_minus_a": _delta(risk_a.get("sharpe"), risk_b.get("sharpe")),
            "trades_total_delta_b_minus_a": _delta(tr_a.get("trades_total"), tr_b.get("trades_total")),
            "turnover_ratio_delta_b_minus_a": _delta(tr_a.get("turnover_ratio"), tr_b.get("turnover_ratio")),
        },
        "daily_returns_compare": daily_summary,
        "gating_compare": {
            "top3_a": ((metrics_a.get("gating") or {}).get("summary") or {}).get("top3", []),
            "top3_b": ((metrics_b.get("gating") or {}).get("summary") or {}).get("top3", []),
            "delta_top5_b_minus_a": gating_delta_top5,
        },
        "data_quality": {
            "a": quality_a,
            "b": quality_b,
            "missing_files_a": (metrics_a.get("data_quality") or {}).get("missing_files", []),
            "missing_files_b": (metrics_b.get("data_quality") or {}).get("missing_files", []),
            "insufficient_points_a": bool((metrics_a.get("data_quality") or {}).get("insufficient_points", False)),
            "insufficient_points_b": bool((metrics_b.get("data_quality") or {}).get("insufficient_points", False)),
        },
    }

    fail_block = evaluate_fail_rules(compare, fail_rules)
    compare["fail_rules"] = fail_block
    return compare, daily_delta_rows


def render_compare_markdown(compare: Dict[str, Any]) -> str:
    meta = compare.get("meta", {}) or {}
    head = compare.get("headline", {}) or {}
    delta = compare.get("delta_metrics", {}) or {}
    dcmp = compare.get("daily_returns_compare", {}) or {}
    gating = compare.get("gating_compare", {}) or {}
    dq = compare.get("data_quality", {}) or {}

    def _pct(x: Any) -> str:
        v = _num_or_none(x)
        return "-" if v is None else f"{v * 100.0:.2f}%"

    def _num(x: Any) -> str:
        v = _num_or_none(x)
        return "-" if v is None else f"{v:,.4f}"

    lines: List[str] = []
    lines.append("# Run-to-Run Compare")
    lines.append("")
    lines.append(f"- Run A: `{meta.get('run_id_a', '-')}`")
    lines.append(f"- Run B: `{meta.get('run_id_b', '-')}`")
    lines.append(f"- Winner: **{head.get('winner', 'TIE')}** ({head.get('winner_reason', '-')})")
    lines.append("")
    lines.append("## Key Deltas (B - A)")
    lines.append(f"- Return: {_pct(delta.get('total_return_delta_b_minus_a'))}")
    lines.append(f"- Max Drawdown: {_pct(delta.get('max_drawdown_delta_b_minus_a'))}")
    lines.append(f"- Volatility: {_pct(delta.get('vol_annualized_delta_b_minus_a'))}")
    lines.append(f"- Sharpe: {_num(delta.get('sharpe_delta_b_minus_a'))}")
    lines.append(f"- Trades: {_num(delta.get('trades_total_delta_b_minus_a'))}")
    lines.append(f"- Turnover Ratio: {_pct(delta.get('turnover_ratio_delta_b_minus_a'))}")
    lines.append("")
    lines.append("## Daily Returns Compare")
    lines.append(f"- Overlap Days: {dcmp.get('overlap_days', 0)}")
    lines.append(f"- Corr: {_num(dcmp.get('corr'))}")
    lines.append(f"- Tracking Error (std): {_num(dcmp.get('tracking_error_std'))}")
    lines.append(f"- Hit Rate (B>A): {_pct(dcmp.get('hit_rate_b_gt_a'))}")
    lines.append("")
    lines.append("## Gating Deltas")
    delta_top = gating.get("delta_top5_b_minus_a", []) or []
    if delta_top:
        for row in delta_top:
            lines.append(
                f"- {row.get('reason', '-')}: "
                f"A={row.get('count_a', 0)} B={row.get('count_b', 0)} "
                f"delta={row.get('delta_b_minus_a', 0)}"
            )
    else:
        lines.append("- No gating deltas captured")
    lines.append("")
    lines.append("## Data Quality")
    lines.append(f"- A insufficient_points: {bool(dq.get('insufficient_points_a', False))}")
    lines.append(f"- B insufficient_points: {bool(dq.get('insufficient_points_b', False))}")
    lines.append("")
    return "\n".join(lines)


def write_delta_daily_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["date_local", "daily_return_a", "daily_return_b", "delta_b_minus_a"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})

