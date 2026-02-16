"""Run aggregation utilities (month cache first, fallback to per-run scan)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from atomic_io import safe_read_json
from outpost import infer_run_kind, is_candidate_run_dir, normalize_run_kind

ALLOWED_RANGES = {"1M", "3M", "6M", "1Y", "YTD"}
PROFILE_SET = {"low", "mid", "high", "ultra"}
ALLOWED_RUN_KINDS = {"live", "dryrun", "diagnostics", "test"}
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _month_shift(dt: datetime, months: int) -> datetime:
    y = dt.year
    m = dt.month + months
    while m <= 0:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    d = dt.day
    while d >= 1:
        try:
            return dt.replace(year=y, month=m, day=d)
        except ValueError:
            d -= 1
    return dt.replace(year=y, month=m, day=1)


def _resolve_range(range_key: str, now_utc: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    key = str(range_key or "1M").strip().upper()
    if key not in ALLOWED_RANGES:
        key = "1M"
    now = now_utc or datetime.now(timezone.utc)
    if key == "YTD":
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    elif key == "1M":
        start = _month_shift(now, -1)
    elif key == "3M":
        start = _month_shift(now, -3)
    elif key == "6M":
        start = _month_shift(now, -6)
    else:  # 1Y
        start = _month_shift(now, -12)
    return start, now


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return rows
    return rows


def _normalize_profile(value: Any) -> str:
    p = str(value or "").strip().lower()
    return p if p in PROFILE_SET else "mid"


def _normalize_kinds(kinds: Optional[Iterable[str]]) -> Set[str]:
    if kinds is None:
        return {"live"}
    out: Set[str] = set()
    for kind in kinds:
        k = normalize_run_kind(kind, default="")
        if k in ALLOWED_RUN_KINDS:
            out.add(k)
    return out or {"live"}


def _parse_month_dir_name(path_name: str) -> str | None:
    name = str(path_name or "").strip()
    if MONTH_RE.match(name):
        return name
    return None


def list_month_dirs(base_out_dir: Path) -> List[Path]:
    if not base_out_dir.exists() or not base_out_dir.is_dir():
        return []
    out: List[Path] = []
    for child in base_out_dir.iterdir():
        if not child.is_dir():
            continue
        month_name = _parse_month_dir_name(child.name)
        if month_name is None:
            continue
        out.append(child)
    out.sort(key=lambda p: p.name)
    return out


def month_in_range(month_str: str, start_utc: datetime, end_utc: datetime) -> bool:
    if _parse_month_dir_name(month_str) is None:
        return False
    y, m = map(int, month_str.split("-"))
    month_start = datetime(y, m, 1, tzinfo=timezone.utc)
    if m == 12:
        next_month = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(y, m + 1, 1, tzinfo=timezone.utc)
    return (next_month > start_utc) and (month_start <= end_utc)


def load_month_summary(month_dir: Path) -> Dict[str, Any] | None:
    path = month_dir / "month_summary.json"
    payload = safe_read_json(str(path), retries=2, sleep_ms=15)
    return payload if isinstance(payload, dict) else None


def fallback_scan_month_run_summaries(month_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not month_dir.exists() or not month_dir.is_dir():
        return runs
    for run_dir in sorted(month_dir.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        if not is_candidate_run_dir(run_dir):
            continue
        run_summary_path = run_dir / "run_summary.json"
        payload = safe_read_json(str(run_summary_path), retries=2, sleep_ms=15)
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row.setdefault("run_id", run_dir.name)
        row["run_kind"] = normalize_run_kind(
            row.get("run_kind"),
            default=infer_run_kind(run_dir, run_summary=row, registry_entry=None),
        )
        row["_summary_path"] = str(run_summary_path.resolve())
        runs.append(row)
    return runs


def expand_month_summary_to_runs(month_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs = month_summary.get("runs")
    if not isinstance(runs, list):
        return rows
    for item in runs:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "run_id": str(item.get("run_id") or "").strip(),
                "ended_at_utc": item.get("ended_at_utc"),
                "final_equity": item.get("final_equity"),
                "pnl": item.get("pnl"),
                "total_return": item.get("total_return"),
                "risk_profile": _normalize_profile(item.get("risk_profile")),
                "overrides_hash": item.get("overrides_hash"),
                "run_kind": normalize_run_kind(item.get("run_kind"), default=""),
            }
        )
    return rows


def aggregate_run_summaries(run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in run_summaries if isinstance(r, dict)]
    run_count = len(rows)
    total_pnl = float(sum(_as_float(r.get("pnl"), 0.0) for r in rows))

    max_drawdown = None
    if rows:
        max_drawdown = float(max(_as_float(r.get("max_drawdown"), 0.0) for r in rows))

    with_ended: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in rows:
        dt = _parse_dt(r.get("ended_at_utc"))
        if isinstance(dt, datetime):
            with_ended.append((dt, r))
    with_ended.sort(key=lambda t: t[0])

    start_equity = None
    end_equity = None
    if with_ended:
        first = with_ended[0][1]
        last = with_ended[-1][1]
        first_final = _as_float(first.get("final_equity"), 0.0)
        first_pnl = _as_float(first.get("pnl"), 0.0)
        if "final_equity" in first and "pnl" in first:
            start_equity = float(first_final - first_pnl)
        else:
            start_equity = float(first_final) if "final_equity" in first else None
        end_equity = float(_as_float(last.get("final_equity"), 0.0)) if "final_equity" in last else None

    by_risk_profile: Dict[str, Dict[str, Any]] = {}
    run_count_by_kind: Dict[str, int] = {}
    for r in rows:
        profile = _normalize_profile(r.get("risk_profile"))
        run_kind = normalize_run_kind(r.get("run_kind"), default="live")
        run_count_by_kind[run_kind] = int(run_count_by_kind.get(run_kind, 0)) + 1
        b = by_risk_profile.setdefault(
            profile,
            {"run_count": 0, "total_pnl": 0.0, "_ret_sum": 0.0, "avg_total_return": 0.0},
        )
        b["run_count"] += 1
        b["total_pnl"] += _as_float(r.get("pnl"), 0.0)
        b["_ret_sum"] += _as_float(r.get("total_return"), 0.0)
    for key in list(by_risk_profile.keys()):
        b = by_risk_profile[key]
        n = max(1, int(b.get("run_count", 0)))
        b["total_pnl"] = float(b.get("total_pnl", 0.0))
        b["avg_total_return"] = float(b.get("_ret_sum", 0.0) / n)
        b.pop("_ret_sum", None)

    runs_compact: List[Dict[str, Any]] = []
    sorted_rows = sorted(
        rows,
        key=lambda r: str(r.get("ended_at_utc") or ""),
        reverse=True,
    )[:50]
    for r in sorted_rows:
        runs_compact.append(
            {
                "run_id": str(r.get("run_id") or "").strip(),
                "ended_at_utc": r.get("ended_at_utc"),
                "final_equity": _as_float(r.get("final_equity"), 0.0) if r.get("final_equity") is not None else None,
                "pnl": _as_float(r.get("pnl"), 0.0) if r.get("pnl") is not None else None,
                "total_return": _as_float(r.get("total_return"), 0.0) if r.get("total_return") is not None else None,
                "risk_profile": _normalize_profile(r.get("risk_profile")),
                "run_kind": normalize_run_kind(r.get("run_kind"), default="live"),
                "overrides_hash": str(r.get("overrides_hash") or ""),
                "summary_path": r.get("_summary_path"),
            }
        )

    return {
        "run_count": int(run_count),
        "total_pnl": float(total_pnl),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "max_drawdown": max_drawdown,
        "by_risk_profile": by_risk_profile,
        "run_count_by_kind": run_count_by_kind,
        "runs": runs_compact,
    }


def _build_missing_runs(
    registry_rows: List[Dict[str, Any]],
    range_start: datetime,
    range_end: datetime,
    allowed_kinds: Set[str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    run_index: Dict[str, Dict[str, Any]] = {}
    for row in registry_rows:
        if str(row.get("action") or "").strip().lower() != "start":
            continue
        month_dir = str(row.get("month_dir") or "").strip()
        if _parse_month_dir_name(month_dir) is None:
            continue
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        out_dir = str(row.get("out_dir") or "").strip()
        inferred_kind = normalize_run_kind(row.get("run_kind"), default="")
        if not inferred_kind and out_dir:
            inferred_kind = infer_run_kind(Path(out_dir), run_summary=None, registry_entry=row)
        inferred_kind = normalize_run_kind(inferred_kind, default="live")
        if inferred_kind not in allowed_kinds:
            continue
        ts = _parse_dt(row.get("ts_utc"))
        if isinstance(ts, datetime) and (ts < range_start or ts > range_end):
            continue
        prev = run_index.get(run_id)
        if prev is None:
            run_index[run_id] = row
            continue
        prev_ts = _parse_dt(prev.get("ts_utc"))
        if isinstance(ts, datetime) and (not isinstance(prev_ts, datetime) or ts >= prev_ts):
            run_index[run_id] = row

    missing_run_ids: List[str] = []
    missing_summaries: List[Dict[str, Any]] = []
    for run_id in sorted(run_index.keys()):
        row = run_index[run_id]
        out_dir = str(row.get("out_dir") or "").strip()
        summary_path = Path(out_dir) / "run_summary.json" if out_dir else None
        exists = bool(summary_path and summary_path.exists())
        if exists:
            continue
        missing_run_ids.append(run_id)
        missing_summaries.append(
            {
                "run_id": run_id,
                "ts_utc": row.get("ts_utc"),
                "out_dir": out_dir,
                "month_dir": row.get("month_dir"),
                "reason": "run_summary_not_found",
            }
        )
    return missing_run_ids, missing_summaries


def summarize_range(
    base_out_dir: str | Path,
    range_key: str,
    now_utc: Optional[datetime] = None,
    kinds: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    base = Path(str(base_out_dir or "outputs")).resolve()
    start_dt, end_dt = _resolve_range(range_key, now_utc=now_utc)
    allowed_kinds = _normalize_kinds(kinds)
    registry_rows = _read_jsonl(base / "registry.jsonl")

    candidate_month_dirs = [m for m in list_month_dirs(base) if month_in_range(m.name, start_dt, end_dt)]
    collected: List[Dict[str, Any]] = []
    used_month_cache_count = 0
    fallback_month_scan_count = 0

    for month_dir in candidate_month_dirs:
        month_summary = load_month_summary(month_dir)
        if isinstance(month_summary, dict):
            cached_runs = expand_month_summary_to_runs(month_summary)
            if cached_runs:
                used_month_cache_count += 1
                for r in cached_runs:
                    if isinstance(r, dict):
                        one = dict(r)
                        if not one.get("_summary_path"):
                            rid = str(one.get("run_id") or "").strip()
                            if rid:
                                one["_summary_path"] = str((month_dir / rid / "run_summary.json").resolve())
                        run_dir = Path(str(one.get("_summary_path") or "")).parent if one.get("_summary_path") else None
                        one["run_kind"] = normalize_run_kind(
                            one.get("run_kind"),
                            default=infer_run_kind(run_dir, run_summary=one, registry_entry=None)
                            if isinstance(run_dir, Path) and run_dir.exists()
                            else "live",
                        )
                        collected.append(one)
                continue
        fallback_month_scan_count += 1
        collected.extend(fallback_scan_month_run_summaries(month_dir))

    selected_rows: List[Dict[str, Any]] = []
    for r in collected:
        dt = _parse_dt(r.get("ended_at_utc"))
        if isinstance(dt, datetime) and (dt < start_dt or dt > end_dt):
            continue
        run_kind = normalize_run_kind(r.get("run_kind"), default="")
        if not run_kind:
            run_dir = None
            summary_path = str(r.get("_summary_path") or "").strip()
            if summary_path:
                run_dir = Path(summary_path).parent
            run_kind = infer_run_kind(run_dir, run_summary=r, registry_entry=None) if isinstance(run_dir, Path) else "live"
        run_kind = normalize_run_kind(run_kind, default="live")
        if run_kind not in allowed_kinds:
            continue
        r["run_kind"] = run_kind
        selected_rows.append(r)

    dedup: Dict[str, Dict[str, Any]] = {}
    for r in selected_rows:
        run_id = str(r.get("run_id") or "").strip()
        if not run_id:
            continue
        prev = dedup.get(run_id)
        if prev is None:
            dedup[run_id] = r
            continue
        prev_dt = _parse_dt(prev.get("ended_at_utc"))
        curr_dt = _parse_dt(r.get("ended_at_utc"))
        if isinstance(curr_dt, datetime) and (not isinstance(prev_dt, datetime) or curr_dt >= prev_dt):
            dedup[run_id] = r

    merged = aggregate_run_summaries(list(dedup.values()))
    missing_run_ids, missing_summaries = _build_missing_runs(
        registry_rows=registry_rows,
        range_start=start_dt,
        range_end=end_dt,
        allowed_kinds=allowed_kinds,
    )

    return {
        "schema_version": 1,
        "base_out_dir": str(base),
        "range": str(range_key).upper(),
        "kinds": sorted(allowed_kinds),
        "range_start_utc": start_dt.isoformat(),
        "range_end_utc": end_dt.isoformat(),
        "used_month_cache_count": int(used_month_cache_count),
        "fallback_month_scan_count": int(fallback_month_scan_count),
        "run_count": int(merged.get("run_count", 0)),
        "total_pnl": float(merged.get("total_pnl", 0.0)),
        "start_equity": merged.get("start_equity"),
        "end_equity": merged.get("end_equity"),
        "max_drawdown": merged.get("max_drawdown"),
        "by_risk_profile": merged.get("by_risk_profile", {}),
        "run_count_by_kind": merged.get("run_count_by_kind", {}),
        "missing_run_ids": missing_run_ids,
        "missing_summaries": missing_summaries,
        "selected_runs": merged.get("runs", []),
    }


__all__ = [
    "list_month_dirs",
    "month_in_range",
    "load_month_summary",
    "fallback_scan_month_run_summaries",
    "aggregate_run_summaries",
    "summarize_range",
    "ALLOWED_RUN_KINDS",
]
