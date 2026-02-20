#!/usr/bin/env python3
"""Leaderboard helpers for A1-4 multi-run ranking."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from quant_io_utils import to_iso_utc


ROOT = Path(__file__).resolve().parents[2]

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


LEADERBOARD_COLUMNS = [
    "run_id",
    "dataset_dir",
    "start_time_utc",
    "end_time_utc",
    "days",
    "total_return",
    "cagr",
    "vol_annualized",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "trades_total",
    "turnover_ratio",
    "buys",
    "sells",
    "unique_tickers",
    "insufficient_points",
    "missing_files_count",
    "parse_warnings_count",
    "gating_top1",
    "gating_top1_count",
    "gating_top2",
    "gating_top2_count",
    "gating_top3",
    "gating_top3_count",
    "composite_score",
    "rank",
]


def _round_opt(value: Any, ndigits: int = 6):
    try:
        if value in (None, ""):
            return None
        return round(float(value), ndigits)
    except Exception:
        return None


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def discover_datasets(base_dir: Path, pattern: str = "**/run_dataset") -> List[Path]:
    if not base_dir.exists():
        return []
    candidates: List[Path] = []
    for p in base_dir.glob(pattern):
        try:
            if p.is_dir():
                candidates.append(p.resolve())
        except Exception:
            continue
    # deterministic order
    return sorted(set(candidates), key=lambda x: str(x).lower())


def ensure_metrics(
    dataset_dir: Path,
    *,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Dict[str, Any]:
    metrics_path = dataset_dir / "metrics" / "metrics.json"
    daily_path = dataset_dir / "metrics" / "daily_returns.csv"
    result = {
        "metrics_path": str(metrics_path),
        "exists": metrics_path.exists() and daily_path.exists(),
        "generated": False,
        "rc": 0,
    }
    if result["exists"] or not auto_metrics:
        return result

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--out-dir",
        str((dataset_dir / "metrics").resolve()),
        "--report-tz",
        str(report_tz),
        "--annualization",
        str(int(annualization)),
        "--rf",
        str(float(rf)),
        "--min-points",
        str(int(min_points)),
    ]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    result["rc"] = int(proc.returncode)
    result["generated"] = proc.returncode == 0
    result["exists"] = metrics_path.exists() and daily_path.exists()
    return result


def _read_daily_dates(dataset_dir: Path) -> Tuple[Optional[str], Optional[str], int]:
    daily_path = dataset_dir / "metrics" / "daily_returns.csv"
    if not daily_path.exists():
        return None, None, 0
    rows = []
    try:
        with daily_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                d = str(row.get("date_local", "") or "").strip()
                if d:
                    rows.append(d)
    except Exception:
        return None, None, 0
    if not rows:
        return None, None, 0
    rows = sorted(rows)
    return rows[0], rows[-1], len(rows)


def _read_equity_time_bounds(dataset_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    p = dataset_dir / "equity_curve.csv"
    if not p.exists():
        return None, None
    times: List[str] = []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                t = str(row.get("time_utc", "") or "").strip()
                if t:
                    times.append(t)
    except Exception:
        return None, None
    if not times:
        return None, None
    times = sorted(times)
    return times[0], times[-1]


def _gating_from_cycles(dataset_dir: Path) -> Dict[str, int]:
    cycles_path = dataset_dir / "cycles.csv"
    counts: Dict[str, int] = {}
    if not cycles_path.exists():
        return counts
    try:
        with cycles_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                reason = (
                    str(row.get("skip_reason", "") or "").strip()
                    or str(row.get("cov_gate_reason", "") or "").strip()
                    or str(row.get("decision_path", "") or "").strip()
                )
                if not reason:
                    continue
                k = reason.lower()
                counts[k] = counts.get(k, 0) + 1
    except Exception:
        return {}
    return counts


def _gating_top3(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))[:3]


def _parse_warnings_count(metrics: Dict[str, Any]) -> int:
    pw = ((metrics.get("data_quality") or {}).get("parse_warnings") or {})
    if not isinstance(pw, dict):
        return 0
    n = 0
    for v in pw.values():
        try:
            n += int(v)
        except Exception:
            continue
    return int(n)


def _composite_score(row: Dict[str, Any]) -> float:
    sharpe = _num_or_none(row.get("sharpe")) or 0.0
    total_return = _num_or_none(row.get("total_return")) or 0.0
    max_dd = _num_or_none(row.get("max_drawdown"))
    turnover = _num_or_none(row.get("turnover_ratio")) or 0.0
    dd_abs = abs(min(0.0, float(max_dd))) if max_dd is not None else 0.0

    score = (
        1.0 * _clip(sharpe, -2.0, 4.0)
        + 1.0 * _clip(total_return, -0.5, 0.5)
        - 1.5 * _clip(dd_abs, 0.0, 0.5)
        - 0.5 * _clip(turnover, 0.0, 1.0)
    )
    if bool(row.get("insufficient_points", False)):
        score -= 0.2
    if int(row.get("missing_files_count", 0) or 0) > 0:
        score -= 0.2
    return float(score)


def _sort_rows(rows: List[Dict[str, Any]], sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    sort_by = str(sort_by or "composite").strip()
    metric_key = "composite_score" if sort_by == "composite" else sort_by
    if sort_by == "max_drawdown":
        # smaller absolute drawdown is better
        def key_fn(r):
            v = _num_or_none(r.get("max_drawdown"))
            abs_dd = abs(min(0.0, float(v))) if v is not None else float("inf")
            return (abs_dd, str(r.get("run_id", "")))
        return sorted(rows, key=key_fn)

    if sort_by in {"vol_annualized", "turnover_ratio"}:
        def key_fn(r):
            v = _num_or_none(r.get(metric_key))
            vv = float(v) if v is not None else float("inf")
            return (vv, str(r.get("run_id", "")))
        return sorted(rows, key=key_fn)

    # default higher is better
    def key_fn(r):
        v = _num_or_none(r.get(metric_key))
        vv = float(v) if v is not None else float("-inf")
        return (vv, str(r.get("run_id", "")))
    return sorted(rows, key=key_fn, reverse=bool(descending))


def rank_rows(rows: List[Dict[str, Any]], *, sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    ranked = _sort_rows(rows, sort_by=sort_by, descending=descending)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def assemble_rows(
    datasets: List[Path],
    *,
    sort_by: str,
    descending: bool,
    include_raw: bool,
    min_days: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, int]]]:
    rows: List[Dict[str, Any]] = []
    per_run_gating: Dict[str, Dict[str, int]] = {}
    summary = {
        "runs_total": len(datasets),
        "ok": 0,
        "insufficient_points": 0,
        "missing_metrics": 0,
    }

    for ds in datasets:
        manifest = _read_json(ds / "manifest.json") or {}
        metrics = _read_json(ds / "metrics" / "metrics.json")
        if not isinstance(metrics, dict):
            summary["missing_metrics"] += 1
            continue
        meta = metrics.get("meta", {}) or {}
        perf = metrics.get("performance", {}) or {}
        risk = metrics.get("risk", {}) or {}
        tr = metrics.get("trading", {}) or {}
        dq = metrics.get("data_quality", {}) or {}

        run_id = str(meta.get("run_id", "") or manifest.get("run_id", "") or ds.name or "unknown_run").strip()
        start_time_utc, end_time_utc = _read_equity_time_bounds(ds)
        _, _, days = _read_daily_dates(ds)
        missing_files = dq.get("missing_files", []) or []
        if not isinstance(missing_files, list):
            missing_files = []
        insufficient_points = bool(dq.get("insufficient_points", False)) or (int(days) < int(min_days))

        counts = _gating_from_cycles(ds)
        if not counts:
            maybe_counts = (((metrics.get("gating") or {}).get("summary") or {}).get("counts") or {})
            if isinstance(maybe_counts, dict):
                for k, v in maybe_counts.items():
                    try:
                        counts[str(k)] = int(v)
                    except Exception:
                        continue
        per_run_gating[run_id] = dict(counts)
        top3 = _gating_top3(counts)
        g1 = top3[0] if len(top3) > 0 else ("", 0)
        g2 = top3[1] if len(top3) > 1 else ("", 0)
        g3 = top3[2] if len(top3) > 2 else ("", 0)

        row = {
            "run_id": run_id,
            "dataset_dir": str(ds),
            "start_time_utc": start_time_utc or "",
            "end_time_utc": end_time_utc or "",
            "days": int(days),
            "total_return": _round_opt(perf.get("total_return")),
            "cagr": _round_opt(perf.get("cagr")),
            "vol_annualized": _round_opt(risk.get("vol_annualized")),
            "sharpe": _round_opt(risk.get("sharpe")),
            "sortino": _round_opt(risk.get("sortino")),
            "max_drawdown": _round_opt(risk.get("max_drawdown")),
            "calmar": _round_opt(risk.get("calmar")),
            "trades_total": int((_num_or_none(tr.get("trades_total")) or 0)),
            "turnover_ratio": _round_opt(tr.get("turnover_ratio")),
            "buys": int((_num_or_none(tr.get("buys")) or 0)),
            "sells": int((_num_or_none(tr.get("sells")) or 0)),
            "unique_tickers": int((_num_or_none(tr.get("unique_tickers")) or 0)),
            "insufficient_points": bool(insufficient_points),
            "missing_files_count": int(len(missing_files)),
            "parse_warnings_count": int(_parse_warnings_count(metrics)),
            "gating_top1": g1[0],
            "gating_top1_count": int(g1[1]),
            "gating_top2": g2[0],
            "gating_top2_count": int(g2[1]),
            "gating_top3": g3[0],
            "gating_top3_count": int(g3[1]),
        }
        row["composite_score"] = _round_opt(_composite_score(row))
        if include_raw:
            row["raw_metrics"] = metrics

        rows.append(row)
        summary["ok"] += 1
        if insufficient_points:
            summary["insufficient_points"] += 1

    ranked = rank_rows(rows, sort_by=sort_by, descending=descending)
    return ranked, summary, per_run_gating


def build_gating_summary(per_run_gating: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    total_runs = len(per_run_gating)
    total: Dict[str, int] = {}
    coverage: Dict[str, int] = {}
    for _, counts in per_run_gating.items():
        seen = set()
        for reason, c in counts.items():
            total[reason] = total.get(reason, 0) + int(c)
            if reason not in seen and c > 0:
                coverage[reason] = coverage.get(reason, 0) + 1
                seen.add(reason)
    rows = []
    for reason in sorted(total.keys(), key=lambda r: (-int(total[r]), str(r))):
        run_cov = int(coverage.get(reason, 0))
        cov_ratio = (run_cov / total_runs) if total_runs > 0 else 0.0
        rows.append(
            {
                "reason": reason,
                "total_count": int(total.get(reason, 0)),
                "run_coverage_count": run_cov,
                "run_coverage_ratio": _round_opt(cov_ratio),
            }
        )
    return rows


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def render_leaderboard_md(
    rows: List[Dict[str, Any]],
    *,
    scan_summary: Dict[str, Any],
    gating_summary_rows: List[Dict[str, Any]],
    sort_by: str,
    top_n: int,
) -> str:
    top_rows = rows[: max(0, int(top_n))]
    bottom_rows = rows[-5:] if len(rows) >= 5 else list(rows)

    lines: List[str] = []
    lines.append("# Leaderboard Summary")
    lines.append("")
    lines.append(
        f"- runs_total={scan_summary.get('runs_total', 0)} "
        f"ok={scan_summary.get('ok', 0)} "
        f"insufficient_points={scan_summary.get('insufficient_points', 0)} "
        f"missing_metrics={scan_summary.get('missing_metrics', 0)} "
        f"auto_metrics_generated={scan_summary.get('auto_metrics_generated', 0)}"
    )
    lines.append(f"- sort_by={sort_by}")
    lines.append("")
    lines.append("## Top 10")
    lines.append("")
    lines.append("| rank | run_id | total_return | sharpe | max_drawdown | trades_total | composite_score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in top_rows[:10]:
        lines.append(
            f"| {r.get('rank','')} | {r.get('run_id','')} | {r.get('total_return','')} | "
            f"{r.get('sharpe','')} | {r.get('max_drawdown','')} | {r.get('trades_total','')} | {r.get('composite_score','')} |"
        )
    lines.append("")
    lines.append("## Bottom 5")
    lines.append("")
    lines.append("| rank | run_id | total_return | sharpe | max_drawdown | trades_total | composite_score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in bottom_rows:
        lines.append(
            f"| {r.get('rank','')} | {r.get('run_id','')} | {r.get('total_return','')} | "
            f"{r.get('sharpe','')} | {r.get('max_drawdown','')} | {r.get('trades_total','')} | {r.get('composite_score','')} |"
        )
    lines.append("")
    lines.append("## Gating Summary Top 5")
    lines.append("")
    for g in gating_summary_rows[:5]:
        lines.append(
            f"- {g.get('reason','')}: total={g.get('total_count',0)} "
            f"coverage={g.get('run_coverage_count',0)}/{scan_summary.get('ok',0)} "
            f"({g.get('run_coverage_ratio',0)})"
        )
    lines.append("")
    lines.append("## Data Quality")
    lines.append("")
    lines.append("- missing_files_count/top parse_warnings reflected in CSV/JSON outputs.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    *,
    rows: List[Dict[str, Any]],
    leaderboard_json: Dict[str, Any],
    leaderboard_md: str,
    gating_summary_rows: List[Dict[str, Any]],
    manifest_leaderboard: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "leaderboard.csv", LEADERBOARD_COLUMNS, rows)
    _write_json(out_dir / "leaderboard.json", leaderboard_json)
    _write_text(out_dir / "leaderboard.md", leaderboard_md)
    _write_csv(
        out_dir / "gating_summary.csv",
        ["reason", "total_count", "run_coverage_count", "run_coverage_ratio"],
        gating_summary_rows,
    )
    _write_json(out_dir / "manifest_leaderboard.json", manifest_leaderboard)
