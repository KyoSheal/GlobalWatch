#!/usr/bin/env python3
"""Build monthly cached summaries from per-run run_summary.json artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, safe_read_json  # noqa: E402
from outpost import infer_run_kind, is_candidate_run_dir, normalize_run_kind  # noqa: E402

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
ALLOWED_RUN_KINDS = {"live", "dryrun", "diagnostics", "test"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _normalize_kinds(kinds: Optional[Iterable[str]]) -> Set[str]:
    if kinds is None:
        return {"live"}
    out: Set[str] = set()
    for k in kinds:
        key = normalize_run_kind(k, default="")
        if key in ALLOWED_RUN_KINDS:
            out.add(key)
    return out or {"live"}


def _discover_months(base_out_dir: Path) -> List[str]:
    months: List[str] = []
    if not base_out_dir.exists():
        return months
    for child in base_out_dir.iterdir():
        if child.is_dir() and MONTH_RE.match(child.name):
            months.append(child.name)
    months.sort()
    return months


def _normalize_months(base_out_dir: Path, months_arg: str) -> List[str]:
    text = str(months_arg or "").strip()
    if not text:
        return _discover_months(base_out_dir)
    months: List[str] = []
    for part in text.split(","):
        m = part.strip()
        if not m:
            continue
        if MONTH_RE.match(m):
            months.append(m)
    dedup = sorted(set(months))
    return dedup


def _collect_run_summaries_for_month(
    base_out_dir: Path,
    month: str,
    allowed_kinds: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    month_dir = base_out_dir / month
    run_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    seen_missing = set()

    if month_dir.exists() and month_dir.is_dir():
        for run_dir in sorted([p for p in month_dir.iterdir() if p.is_dir()]):
            if not is_candidate_run_dir(run_dir):
                continue
            run_id = run_dir.name
            run_summary_path = run_dir / "run_summary.json"
            if run_summary_path.exists():
                payload = safe_read_json(str(run_summary_path), retries=2, sleep_ms=15)
                if isinstance(payload, dict):
                    run_kind = normalize_run_kind(
                        payload.get("run_kind"),
                        default=infer_run_kind(run_dir, run_summary=payload, registry_entry=None),
                    )
                    if run_kind not in allowed_kinds:
                        continue
                    row = dict(payload)
                    row["_run_id"] = str(payload.get("run_id") or run_id)
                    row["_run_dir"] = str(run_dir.resolve())
                    row["_run_summary_path"] = str(run_summary_path.resolve())
                    row["run_kind"] = run_kind
                    ended = _parse_dt(payload.get("ended_at_utc"))
                    if ended is None:
                        try:
                            ended = datetime.fromtimestamp(run_summary_path.stat().st_mtime, tz=timezone.utc)
                        except Exception:
                            ended = None
                    row["_ended_dt"] = ended
                    run_rows.append(row)
                    continue
            key = (run_id, str(run_dir.resolve()))
            if key not in seen_missing:
                seen_missing.add(key)
                missing.append(
                    {
                        "run_id": run_id,
                        "out_dir": str(run_dir.resolve()),
                        "reason": "run_summary_not_found",
                    }
                )

    #补充 registry 里 start 但目录缺 run_summary 的 run_id
    registry_rows = _read_jsonl(base_out_dir / "registry.jsonl")
    for row in registry_rows:
        if str(row.get("action") or "").strip().lower() != "start":
            continue
        if str(row.get("month_dir") or "").strip() != month:
            continue
        out_dir = str(row.get("out_dir") or "").strip()
        inferred_kind = normalize_run_kind(row.get("run_kind"), default="")
        if not inferred_kind and out_dir:
            inferred_kind = infer_run_kind(Path(out_dir), run_summary=None, registry_entry=row)
        inferred_kind = normalize_run_kind(inferred_kind, default="live")
        if inferred_kind not in allowed_kinds:
            continue
        run_id = str(row.get("run_id") or "").strip()
        if not run_id or not out_dir:
            continue
        summary_path = Path(out_dir) / "run_summary.json"
        if summary_path.exists():
            continue
        key = (run_id, str(Path(out_dir).resolve()))
        if key in seen_missing:
            continue
        seen_missing.add(key)
        missing.append(
            {
                "run_id": run_id,
                "out_dir": str(Path(out_dir).resolve()),
                "reason": "run_summary_not_found",
            }
        )

    return run_rows, missing


def _build_month_summary(base_out_dir: Path, month: str, allowed_kinds: Set[str]) -> Dict[str, Any]:
    run_rows, missing = _collect_run_summaries_for_month(base_out_dir, month, allowed_kinds=allowed_kinds)
    run_rows.sort(key=lambda r: str((r.get("ended_at_utc") or "")).strip() or str((r.get("_ended_dt") or "")))

    run_count = len(run_rows)
    total_pnl = float(sum(_as_float(r.get("pnl"), 0.0) for r in run_rows))

    start_equity = None
    end_equity = None
    max_drawdown = None
    if run_rows:
        first = run_rows[0]
        last = run_rows[-1]
        first_final = _as_float(first.get("final_equity"), 0.0)
        first_pnl = _as_float(first.get("pnl"), 0.0)
        start_equity = float(first.get("start_equity", first_final - first_pnl))
        end_equity = float(_as_float(last.get("final_equity"), 0.0))
        max_drawdown = float(max(_as_float(r.get("max_drawdown"), 0.0) for r in run_rows))

    by_risk_profile: Dict[str, Dict[str, Any]] = {}
    run_count_by_kind: Dict[str, int] = {}
    for r in run_rows:
        profile = str(r.get("risk_profile") or "mid").strip().lower() or "mid"
        kind = normalize_run_kind(r.get("run_kind"), default="live")
        run_count_by_kind[kind] = int(run_count_by_kind.get(kind, 0)) + 1
        b = by_risk_profile.setdefault(
            profile,
            {"run_count": 0, "total_pnl": 0.0, "_ret_sum": 0.0, "avg_total_return": 0.0},
        )
        b["run_count"] += 1
        b["total_pnl"] += _as_float(r.get("pnl"), 0.0)
        b["_ret_sum"] += _as_float(r.get("total_return"), 0.0)
    for k in list(by_risk_profile.keys()):
        b = by_risk_profile[k]
        n = max(1, int(b.get("run_count", 0)))
        b["total_pnl"] = float(b.get("total_pnl", 0.0))
        b["avg_total_return"] = float(b.get("_ret_sum", 0.0) / n)
        b.pop("_ret_sum", None)

    # runs: 最多最近 50 条
    recent_runs = sorted(
        run_rows,
        key=lambda r: str(r.get("ended_at_utc") or r.get("_ended_dt") or ""),
        reverse=True,
    )[:50]
    runs_compact = [
        {
            "run_id": str(r.get("run_id") or r.get("_run_id") or ""),
            "ended_at_utc": str(r.get("ended_at_utc") or (r.get("_ended_dt").isoformat() if isinstance(r.get("_ended_dt"), datetime) else "")),
            "final_equity": _as_float(r.get("final_equity"), 0.0),
            "pnl": _as_float(r.get("pnl"), 0.0),
            "total_return": _as_float(r.get("total_return"), 0.0),
            "risk_profile": str(r.get("risk_profile") or "mid").strip().lower() or "mid",
            "run_kind": normalize_run_kind(r.get("run_kind"), default="live"),
            "overrides_hash": str(r.get("overrides_hash") or ""),
        }
        for r in recent_runs
    ]

    return {
        "schema_version": 1,
        "month": month,
        "kinds": sorted(allowed_kinds),
        "updated_at_utc": _utc_now_iso(),
        "run_count": int(run_count),
        "total_pnl": float(total_pnl),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "max_drawdown": max_drawdown,
        "by_risk_profile": by_risk_profile,
        "run_count_by_kind": run_count_by_kind,
        "runs": runs_compact,
        "missing_runs": missing,
    }


def build_month_summaries(
    base_out_dir: str,
    months: List[str],
    kinds: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    base = Path(str(base_out_dir or "outputs")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    allowed_kinds = _normalize_kinds(kinds)
    summaries: List[Dict[str, Any]] = []
    for month in months:
        if not MONTH_RE.match(month):
            continue
        summary = _build_month_summary(base, month, allowed_kinds=allowed_kinds)
        month_dir = base / month
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / "month_summary.json"
        atomic_write_json(str(out_path), summary, indent=2)
        summaries.append(summary)
    return {
        "schema_version": 1,
        "base_out_dir": str(base),
        "kinds": sorted(allowed_kinds),
        "updated_at_utc": _utc_now_iso(),
        "months": summaries,
        "month_count": len(summaries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build month_summary.json cache files under outputs/YYYY-MM.")
    parser.add_argument("--base-out-dir", type=str, default="outputs", help="Base output dir (default: outputs)")
    parser.add_argument("--months", type=str, default="", help="Comma-separated months like 2026-02,2026-01")
    parser.add_argument("--kinds", type=str, default="", help="Comma-separated run kinds (default: live)")
    parser.add_argument("--include-dryrun", action="store_true", help="Include dryrun runs in month cache.")
    parser.add_argument("--include-diagnostics", action="store_true", help="Include diagnostics runs in month cache.")
    parser.add_argument("--include-test", action="store_true", help="Include test runs in month cache.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    base = Path(str(args.base_out_dir or "outputs")).resolve()
    months = _normalize_months(base, str(args.months or ""))
    if str(args.kinds or "").strip():
        kinds = [part.strip() for part in str(args.kinds).split(",") if part.strip()]
    else:
        kinds = ["live"]
        if bool(args.include_dryrun):
            kinds.append("dryrun")
        if bool(args.include_diagnostics):
            kinds.append("diagnostics")
        if bool(args.include_test):
            kinds.append("test")
    result = build_month_summaries(str(base), months, kinds=kinds)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[OK] month summaries built: {result.get('month_count', 0)}")
        for m in result.get("months", []):
            if not isinstance(m, dict):
                continue
            print(
                f"  - {m.get('month')}: runs={m.get('run_count')} pnl={m.get('total_pnl')} "
                f"missing={len(m.get('missing_runs', []) if isinstance(m.get('missing_runs'), list) else [])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
