#!/usr/bin/env python3
"""A1-6: Daily Quant Pack orchestration helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_io_utils import safe_read_json, to_iso_utc

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


SCHEMA_VERSION = 1


@dataclass
class StepResult:
    status: str
    rc: int
    cmd: List[str]
    duration_ms: int
    stdout_tail: List[str]
    stderr_tail: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rc": int(self.rc),
            "cmd": list(self.cmd),
            "duration_ms": int(self.duration_ms),
            "stdout_tail": list(self.stdout_tail),
            "stderr_tail": list(self.stderr_tail),
            "notes": list(self.notes),
        }


def _tail_lines(text: str, n: int) -> List[str]:
    lines = [x for x in str(text or "").splitlines() if x is not None]
    if n <= 0:
        return []
    return lines[-n:]


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


def _run_step(cmd: Sequence[str], *, cwd: Path, timeout_sec: int = 600, tail_lines: int = 120) -> StepResult:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        rc = int(proc.returncode)
        status = "ok" if rc == 0 else "fail"
        return StepResult(
            status=status,
            rc=rc,
            cmd=[str(x) for x in cmd],
            duration_ms=dt_ms,
            stdout_tail=_tail_lines(proc.stdout, tail_lines),
            stderr_tail=_tail_lines(proc.stderr, tail_lines),
            notes=[],
        )
    except subprocess.TimeoutExpired as exc:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return StepResult(
            status="fail",
            rc=124,
            cmd=[str(x) for x in cmd],
            duration_ms=dt_ms,
            stdout_tail=_tail_lines(exc.stdout or "", tail_lines),
            stderr_tail=_tail_lines(exc.stderr or "", tail_lines),
            notes=[f"timeout_sec={timeout_sec}"],
        )


def _parse_day_name(name: str) -> Optional[date]:
    s = str(name or "").strip()
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _day_from_dataset_dir(dataset_dir: Path, base_dir: Path) -> Optional[date]:
    try:
        rel = dataset_dir.resolve().relative_to(base_dir.resolve())
    except Exception:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    if parts[0] == "quant_packs" and len(parts) >= 3:
        return _parse_day_name(parts[1])
    return _parse_day_name(parts[0])


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _extract_metrics_topline(metrics_obj: Dict[str, Any]) -> Dict[str, Any]:
    perf = (metrics_obj.get("performance") or {}) if isinstance(metrics_obj, dict) else {}
    risk = (metrics_obj.get("risk") or {}) if isinstance(metrics_obj, dict) else {}
    trading = (metrics_obj.get("trading") or {}) if isinstance(metrics_obj, dict) else {}
    gating = ((metrics_obj.get("gating") or {}).get("summary") or {}) if isinstance(metrics_obj, dict) else {}
    return {
        "total_return": perf.get("total_return"),
        "cagr": perf.get("cagr"),
        "sharpe": risk.get("sharpe"),
        "max_drawdown": risk.get("max_drawdown"),
        "vol_annualized": risk.get("vol_annualized"),
        "turnover_ratio": trading.get("turnover_ratio"),
        "trades_total": trading.get("trades_total"),
        "gating_top3": list(gating.get("top3", []) or []),
    }


def _parse_date_str(date_str: str) -> Optional[date]:
    s = str(date_str or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def resolve_daily_context(
    *,
    daily_dir_arg: str,
    daily_base: Path,
    date_str: str,
    out_dir_arg: str,
    dataset_dir_arg: str,
) -> Dict[str, Any]:
    daily_dir_raw = str(daily_dir_arg or "").strip()
    date_raw = str(date_str or "").strip()
    mode = "dir"

    if daily_dir_raw:
        daily_dir = Path(daily_dir_raw).resolve()
        parsed_date = _parse_day_name(daily_dir.name)
        date_value = parsed_date.isoformat() if parsed_date else date_raw
        daily_report_path = (daily_dir / "daily_report.md").resolve()
        default_out_dir = (daily_dir / "quant").resolve()
    else:
        mode = "flat_json"
        parsed = _parse_date_str(date_raw)
        date_value = parsed.isoformat() if parsed else date_raw
        daily_dir = (daily_base / date_value).resolve()
        daily_report_path = (daily_base / f"{date_value}.json").resolve()
        default_out_dir = (daily_base / "quant_packs" / date_value).resolve()

    out_dir = Path(out_dir_arg).resolve() if str(out_dir_arg or "").strip() else default_out_dir
    if str(dataset_dir_arg or "").strip():
        dataset_dir = Path(dataset_dir_arg).resolve()
    else:
        if mode == "flat_json":
            dataset_dir = (out_dir / "run_dataset").resolve()
        else:
            dataset_dir = (daily_dir / "run_dataset").resolve()

    return {
        "mode": mode,
        "date_str": date_value,
        "daily_dir": daily_dir,
        "daily_report_path": daily_report_path,
        "out_dir": out_dir,
        "dataset_dir": dataset_dir,
    }


def resolve_dataset_dir(daily_dir: Path, dataset_dir_arg: str) -> Path:
    if str(dataset_dir_arg or "").strip():
        return Path(dataset_dir_arg).resolve()
    return (daily_dir / "run_dataset").resolve()


def _resolve_prev_day_dataset(current_day: Optional[date], base_dir: Path) -> Tuple[Optional[Path], str]:
    if current_day is None:
        return None, "prev_day_unavailable_invalid_date"
    prev_day = current_day - timedelta(days=1)
    prev = prev_day.isoformat()
    candidates = [
        (base_dir / "quant_packs" / prev / "run_dataset").resolve(),
        (base_dir / prev / "run_dataset").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, "prev_day_exact_match"
    return None, "prev_day_dataset_not_found"


def _iter_recent_dataset_dirs(base_dir: Path, *, before_day: Optional[date], lookback_days: int) -> List[Path]:
    out: List[Tuple[date, Path]] = []
    if not base_dir.exists():
        return []
    roots = [base_dir, (base_dir / "quant_packs")]
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            d = _parse_day_name(child.name)
            if d is None:
                continue
            if before_day is not None and d >= before_day:
                continue
            if before_day is not None and (before_day - d).days > int(max(1, lookback_days)):
                continue
            ds = (child / "run_dataset").resolve()
            if ds.exists() and ds.is_dir():
                out.append((d, ds))
    out.sort(key=lambda x: (x[0], str(x[1]).lower()), reverse=True)
    dedup: List[Path] = []
    seen = set()
    for _, p in out:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup


def _ensure_dataset_metrics(
    dataset_dir: Path,
    *,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> StepResult:
    metrics_json = dataset_dir / "metrics" / "metrics.json"
    daily_csv = dataset_dir / "metrics" / "daily_returns.csv"
    if metrics_json.exists() and daily_csv.exists():
        return StepResult(
            status="ok",
            rc=0,
            cmd=[],
            duration_ms=0,
            stdout_tail=[],
            stderr_tail=[],
            notes=["dataset_metrics_already_present"],
        )
    if not auto_metrics:
        return StepResult(
            status="warn",
            rc=0,
            cmd=[],
            duration_ms=0,
            stdout_tail=[],
            stderr_tail=[],
            notes=["dataset_metrics_missing_auto_metrics_disabled"],
        )

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
    return _run_step(cmd, cwd=ROOT, tail_lines=80)


def _resolve_best_recent_dataset(
    *,
    current_day: Optional[date],
    base_dir: Path,
    lookback_days: int,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[Optional[Path], str, List[Dict[str, Any]]]:
    candidates = _iter_recent_dataset_dirs(base_dir, before_day=current_day, lookback_days=lookback_days)
    scored: List[Tuple[float, float, Path]] = []
    diagnostics: List[Dict[str, Any]] = []
    for ds in candidates:
        ensure_step = _ensure_dataset_metrics(
            ds,
            auto_metrics=auto_metrics,
            report_tz=report_tz,
            annualization=annualization,
            rf=rf,
            min_points=min_points,
            verbose=verbose,
        )
        metrics_obj = safe_read_json(ds / "metrics" / "metrics.json") or {}
        sharpe = _num_or_none(((metrics_obj.get("risk") or {}).get("sharpe")))
        max_dd = _num_or_none(((metrics_obj.get("risk") or {}).get("max_drawdown")))
        dd_score = float(max_dd) if max_dd is not None else float("-inf")
        sharpe_score = float(sharpe) if sharpe is not None else float("-inf")
        diagnostics.append(
            {
                "dataset_dir": str(ds),
                "ensure_metrics": ensure_step.to_dict(),
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            }
        )
        if sharpe is None and max_dd is None:
            continue
        scored.append((sharpe_score, dd_score, ds))
    if not scored:
        return None, "best_recent_no_eligible_dataset", diagnostics
    scored.sort(key=lambda x: (x[0], x[1], str(x[2]).lower()), reverse=True)
    return scored[0][2], "best_recent_ranked_by_sharpe_then_max_drawdown", diagnostics


def resolve_baseline_dataset(
    *,
    current_day: Optional[date],
    base_dir: Path,
    baseline_dataset_arg: str,
    baseline_mode: str,
    lookback_days: int,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[Optional[Path], str, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    if str(baseline_dataset_arg or "").strip():
        p = Path(baseline_dataset_arg).resolve()
        if p.exists():
            return p, "explicit_baseline_arg", trace
        return None, "explicit_baseline_missing", trace

    mode = str(baseline_mode or "prev_day").strip().lower()
    if mode == "prev_day":
        ds, reason = _resolve_prev_day_dataset(current_day, base_dir)
        return ds, reason, trace
    if mode == "best_recent":
        ds, reason, extra = _resolve_best_recent_dataset(
            current_day=current_day,
            base_dir=base_dir,
            lookback_days=lookback_days,
            auto_metrics=auto_metrics,
            report_tz=report_tz,
            annualization=annualization,
            rf=rf,
            min_points=min_points,
            verbose=verbose,
        )
        trace.extend(extra)
        return ds, reason, trace
    return None, f"unsupported_baseline_mode_{mode}", trace


def _copy_metrics_from_dataset(dataset_dir: Path, metrics_out_dir: Path) -> Dict[str, Any]:
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in ("metrics.json", "metrics.md", "daily_returns.csv"):
        src = dataset_dir / "metrics" / name
        dst = metrics_out_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return {"copied": copied, "missing": [x for x in ("metrics.json", "metrics.md", "daily_returns.csv") if x not in copied]}


def ensure_metrics(
    *,
    dataset_dir: Path,
    out_dir: Path,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    metrics_out_dir = (out_dir / "metrics").resolve()
    metrics_json = metrics_out_dir / "metrics.json"
    notes: List[str] = []
    step: StepResult

    if metrics_json.exists():
        step = StepResult("ok", 0, [], 0, [], [], ["metrics_already_present_in_out_dir"])
    else:
        dataset_metrics_json = dataset_dir / "metrics" / "metrics.json"
        dataset_daily_csv = dataset_dir / "metrics" / "daily_returns.csv"
        if dataset_metrics_json.exists() and dataset_daily_csv.exists() and not auto_metrics:
            cp = _copy_metrics_from_dataset(dataset_dir, metrics_out_dir)
            notes.append(f"copied_from_dataset={cp.get('copied', [])}")
            step = StepResult("ok", 0, [], 0, [], [], notes)
        elif auto_metrics:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--out-dir",
                str(metrics_out_dir),
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
            step = _run_step(cmd, cwd=ROOT, tail_lines=120)
        else:
            step = StepResult(
                status="warn",
                rc=0,
                cmd=[],
                duration_ms=0,
                stdout_tail=[],
                stderr_tail=[],
                notes=["metrics_missing_auto_metrics_disabled"],
            )

    metrics_obj = safe_read_json(metrics_json)
    state = {
        "status": step.status if isinstance(metrics_obj, dict) else ("warn" if step.status == "ok" else step.status),
        "metrics_dir": str(metrics_out_dir),
        "metrics_json_exists": bool(metrics_json.exists()),
        "step": step.to_dict(),
    }
    if not isinstance(metrics_obj, dict):
        state.setdefault("warnings", []).append("metrics_json_unavailable")
        return state, None
    return state, metrics_obj


def ensure_leaderboard(
    *,
    base_dir: Path,
    out_dir: Path,
    lookback_days: int,
    auto_leaderboard: bool,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    leaderboard_out = (out_dir / "leaderboard").resolve()
    if not auto_leaderboard:
        state = {
            "status": "skipped",
            "leaderboard_dir": str(leaderboard_out),
            "step": StepResult("skipped", 0, [], 0, [], [], ["auto_leaderboard_disabled"]).to_dict(),
        }
        return state, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a4_build_leaderboard.py"),
        "--base-dir",
        str(base_dir),
        "--out-dir",
        str(leaderboard_out),
        "--sort-by",
        "sharpe",
        "--report-tz",
        str(report_tz),
        "--annualization",
        str(int(annualization)),
        "--rf",
        str(float(rf)),
        "--min-points",
        str(int(min_points)),
        "--min-days",
        str(max(1, int(lookback_days // 2))),
        "--top",
        "200",
    ]
    if auto_metrics:
        cmd.append("--auto-metrics")
    if verbose:
        cmd.append("--verbose")
    step = _run_step(cmd, cwd=ROOT, tail_lines=120)
    leaderboard_obj = safe_read_json(leaderboard_out / "leaderboard.json")
    state = {
        "status": "ok" if step.rc == 0 and isinstance(leaderboard_obj, dict) else ("warn" if step.rc == 0 else "fail"),
        "leaderboard_dir": str(leaderboard_out),
        "step": step.to_dict(),
    }
    if not isinstance(leaderboard_obj, dict):
        state.setdefault("warnings", []).append("leaderboard_json_unavailable")
        return state, None
    return state, leaderboard_obj


def ensure_gate(
    *,
    baseline_dir: Optional[Path],
    dataset_dir: Path,
    out_dir: Path,
    auto_gate: bool,
    auto_metrics: bool,
    strict: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    gate_out = (out_dir / "gate").resolve()
    if not auto_gate:
        return (
            {
                "status": "skipped",
                "gate_dir": str(gate_out),
                "step": StepResult("skipped", 0, [], 0, [], [], ["auto_gate_disabled"]).to_dict(),
            },
            None,
        )
    if baseline_dir is None or not baseline_dir.exists():
        return (
            {
                "status": "warn",
                "gate_dir": str(gate_out),
                "step": StepResult("warn", 0, [], 0, [], [], ["baseline_unavailable_gate_skipped"]).to_dict(),
            },
            None,
        )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a5_quant_gate.py"),
        "--baseline",
        str(baseline_dir),
        "--candidate",
        str(dataset_dir),
        "--out-dir",
        str(gate_out),
        "--report-tz",
        str(report_tz),
        "--annualization",
        str(int(annualization)),
        "--rf",
        str(float(rf)),
        "--min-points",
        str(int(min_points)),
    ]
    if auto_metrics:
        cmd.append("--auto-metrics")
    if strict:
        cmd.append("--strict")
    if verbose:
        cmd.append("--verbose")
    step = _run_step(cmd, cwd=ROOT, tail_lines=140)

    gate_result = safe_read_json(gate_out / "gate_result.json")
    gate_status = ""
    if isinstance(gate_result, dict):
        gate_status = str(gate_result.get("status", "")).upper()
    state = {
        "status": "ok" if step.rc in (0, 3) else "fail",
        "gate_dir": str(gate_out),
        "gate_status": gate_status or ("FAIL" if step.rc == 3 else ""),
        "step": step.to_dict(),
    }
    if not isinstance(gate_result, dict):
        state.setdefault("warnings", []).append("gate_result_unavailable")
        return state, None
    return state, gate_result


def _find_run_dir_by_run_id(base_out_dir: Path, run_id: str) -> Optional[Path]:
    rid = str(run_id or "").strip()
    if not rid:
        return None
    month_prefix = ""
    if len(rid) >= 8 and rid[:8].isdigit():
        month_prefix = f"{rid[:4]}-{rid[4:6]}"
    if month_prefix:
        p = (base_out_dir / month_prefix / rid).resolve()
        if p.exists() and p.is_dir():
            return p
    for p in base_out_dir.rglob(rid):
        if p.is_dir():
            return p.resolve()
    return None


def discover_run_dir_for_date(
    *,
    date_str: str,
    daily_report_path: Path,
    base_out_dir: Path,
) -> Tuple[Optional[Path], str, List[str]]:
    trace: List[str] = []
    report_obj = safe_read_json(daily_report_path) if daily_report_path.exists() else None
    if isinstance(report_obj, dict):
        for key in ("run_dir", "runDir", "output_dir", "out_dir", "run_path"):
            v = str(report_obj.get(key, "") or "").strip()
            if not v:
                continue
            cand = Path(v)
            if not cand.is_absolute():
                cand = (ROOT / cand).resolve()
            trace.append(f"report_field:{key}={cand}")
            if cand.exists() and cand.is_dir():
                return cand, f"report_field_{key}", trace
        for key in ("run_id", "runId", "session_id", "sessionId"):
            rid = str(report_obj.get(key, "") or "").strip()
            if not rid:
                continue
            cand = _find_run_dir_by_run_id(base_out_dir, rid)
            trace.append(f"report_field:{key}={rid}")
            if cand is not None:
                return cand, f"report_field_{key}_resolved_run_id", trace

    date_compact = str(date_str or "").replace("-", "")
    month_dir = (base_out_dir / str(date_str or "")[:7]).resolve()
    month_hits: List[Path] = []
    if month_dir.exists():
        for child in month_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(f"{date_compact}-"):
                month_hits.append(child.resolve())
    if month_hits:
        month_hits.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
        trace.append(f"month_dir_hits={len(month_hits)}")
        return month_hits[0], "month_dir_prefix_match", trace

    global_hits: List[Path] = []
    if base_out_dir.exists():
        for child in base_out_dir.rglob("*"):
            try:
                if child.is_dir() and child.name.startswith(f"{date_compact}-"):
                    global_hits.append(child.resolve())
            except Exception:
                continue
    if global_hits:
        global_hits.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
        trace.append(f"global_hits={len(global_hits)}")
        return global_hits[0], "global_prefix_match", trace

    return None, "run_dir_not_found", trace


def _format_pct(v: Any) -> str:
    try:
        if v is None:
            return "-"
        return f"{float(v) * 100.0:.2f}%"
    except Exception:
        return "-"


def _format_num(v: Any) -> str:
    try:
        if v is None:
            return "-"
        return f"{float(v):,.4f}"
    except Exception:
        return "-"


def _relative_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _extract_recent_top5(
    leaderboard_obj: Dict[str, Any],
    *,
    base_dir: Path,
    current_day: Optional[date],
    lookback_days: int,
) -> List[Dict[str, Any]]:
    rows = list(leaderboard_obj.get("rows", []) or [])
    enriched: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = Path(str(row.get("dataset_dir", "") or ""))
        day = _day_from_dataset_dir(ds, base_dir)
        if current_day is not None and day is not None:
            if day > current_day:
                continue
            if (current_day - day).days > int(max(1, lookback_days)):
                continue
        sharpe = _num_or_none(row.get("sharpe"))
        dd = _num_or_none(row.get("max_drawdown"))
        sharpe_score = float(sharpe) if sharpe is not None else float("-inf")
        dd_score = float(dd) if dd is not None else float("-inf")
        enriched.append((sharpe_score, dd_score, str(row.get("run_id", "")), row))
    enriched.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [x[3] for x in enriched[:5]]


def render_daily_quant_report_md(
    *,
    daily_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    strict: bool,
    baseline_dir: Optional[Path],
    baseline_reason: str,
    lookback_days: int,
    metrics_obj: Optional[Dict[str, Any]],
    gate_result: Optional[Dict[str, Any]],
    leaderboard_obj: Optional[Dict[str, Any]],
    base_dir: Path,
    warnings: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Daily Quant Pack")
    lines.append("")
    lines.append(f"- Daily Dir: `{daily_dir}`")
    lines.append(f"- Dataset Dir: `{dataset_dir}`")
    lines.append(f"- Generated (UTC): `{to_iso_utc(datetime.now(timezone.utc))}`")
    lines.append(f"- strict: `{bool(strict)}`")
    lines.append(f"- baseline: `{baseline_dir}`")
    lines.append(f"- baseline_reason: `{baseline_reason}`")
    lines.append("")

    lines.append("## Today Metrics")
    if isinstance(metrics_obj, dict):
        top = _extract_metrics_topline(metrics_obj)
        lines.append(f"- Total Return: {_format_pct(top.get('total_return'))}")
        lines.append(f"- CAGR: {_format_pct(top.get('cagr'))}")
        lines.append(f"- Sharpe: {_format_num(top.get('sharpe'))}")
        lines.append(f"- Max Drawdown: {_format_pct(top.get('max_drawdown'))}")
        lines.append(f"- Vol (ann): {_format_pct(top.get('vol_annualized'))}")
        lines.append(f"- Turnover Ratio: {_format_pct(top.get('turnover_ratio'))}")
        lines.append(f"- Trades Total: {_format_num(top.get('trades_total'))}")
    else:
        lines.append("- metrics unavailable")
    lines.append("")

    lines.append("## Top No-Trade / Gating Reasons")
    gtop = []
    if isinstance(metrics_obj, dict):
        gtop = (((metrics_obj.get("gating") or {}).get("summary") or {}).get("top3") or [])
    if gtop:
        for item in gtop:
            lines.append(f"- {item.get('reason', '-')}: {item.get('count', 0)}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Gate")
    if isinstance(gate_result, dict):
        status = str(gate_result.get("status", "-"))
        lines.append(f"- Gate: **{status}**")
        eval_block = gate_result.get("gate_eval", {}) or {}
        fail_rules = list(eval_block.get("fail_rules", []) or [])
        warns = list(eval_block.get("warnings", []) or [])
        if fail_rules:
            lines.append("- Fail rules:")
            for fr in fail_rules:
                if isinstance(fr, dict):
                    lines.append(f"  - {fr.get('rule', '-')}")
                else:
                    lines.append(f"  - {fr}")
        else:
            lines.append("- Fail rules: none")
        if warns:
            lines.append("- Warnings:")
            for w in warns:
                lines.append(f"  - {w}")
        else:
            lines.append("- Warnings: none")
    else:
        lines.append("- Gate: skipped/unavailable")
    lines.append("")

    lines.append(f"## Leaderboard (Top 5, lookback={int(lookback_days)}d)")
    top5: List[Dict[str, Any]] = []
    if isinstance(leaderboard_obj, dict):
        current_day = _parse_day_name(daily_dir.name)
        top5 = _extract_recent_top5(
            leaderboard_obj,
            base_dir=base_dir,
            current_day=current_day,
            lookback_days=lookback_days,
        )
    if top5:
        lines.append("| rank | run_id | sharpe | max_dd | total_return | insufficient |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, row in enumerate(top5, start=1):
            lines.append(
                f"| {i} | {row.get('run_id','-')} | {_format_num(row.get('sharpe'))} | "
                f"{_format_pct(row.get('max_drawdown'))} | {_format_pct(row.get('total_return'))} | "
                f"{bool(row.get('insufficient_points', False))} |"
            )
    else:
        lines.append("- leaderboard unavailable")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Outputs")
    lines.append(f"- `{_relative_or_abs(out_dir / 'metrics' / 'metrics.md', out_dir)}`")
    lines.append(f"- `{_relative_or_abs(out_dir / 'gate' / 'gate_report.md', out_dir)}`")
    lines.append(f"- `{_relative_or_abs(out_dir / 'leaderboard' / 'leaderboard.md', out_dir)}`")
    lines.append("")

    return "\n".join(lines)


def build_daily_pack(
    *,
    daily_dir_arg: str,
    daily_base: Path,
    date_str: str,
    dataset_dir_arg: str,
    baseline_dataset_arg: str,
    baseline_mode: str,
    base_dir: Path,
    lookback_days: int,
    auto_extract: bool,
    base_out_dir: Path,
    auto_metrics: bool,
    auto_gate: bool,
    auto_leaderboard: bool,
    out_dir: Path,
    strict: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Tuple[int, Dict[str, Any]]:
    context = resolve_daily_context(
        daily_dir_arg=daily_dir_arg,
        daily_base=daily_base.resolve(),
        date_str=date_str,
        out_dir_arg=str(out_dir),
        dataset_dir_arg=dataset_dir_arg,
    )

    daily_dir = Path(context["daily_dir"]).resolve()
    daily_report_path = Path(context["daily_report_path"]).resolve()
    out_dir = Path(context["out_dir"]).resolve()
    dataset_dir = Path(context["dataset_dir"]).resolve()
    mode = str(context["mode"])
    date_value = str(context["date_str"] or "")

    warnings: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
        "mode": mode,
        "date": date_value,
        "daily_dir": str(daily_dir.resolve()),
        "daily_report_path": str(daily_report_path),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "strict": bool(strict),
        "options": {
            "baseline_mode": str(baseline_mode),
            "lookback_days": int(lookback_days),
            "auto_extract": bool(auto_extract),
            "auto_metrics": bool(auto_metrics),
            "auto_gate": bool(auto_gate),
            "auto_leaderboard": bool(auto_leaderboard),
            "report_tz": str(report_tz),
            "annualization": int(annualization),
            "rf": float(rf),
            "min_points": int(min_points),
        },
        "steps": {},
        "warnings": warnings,
    }

    if mode == "dir" and not daily_dir.exists():
        manifest["status"] = "fail"
        warnings.append(f"daily_dir_not_found:{daily_dir}")
        _write_json(out_dir / "pack_manifest.json", manifest)
        return 2, manifest

    # Step 1: optional A1 extract
    if not dataset_dir.exists():
        if auto_extract:
            run_dir, run_reason, run_trace = discover_run_dir_for_date(
                date_str=date_value,
                daily_report_path=daily_report_path,
                base_out_dir=base_out_dir.resolve(),
            )
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a1_extract_run_dataset.py"),
                "--base-out-dir",
                str(base_out_dir.resolve()),
                "--out-dir",
                str(dataset_dir),
            ]
            if run_dir is not None:
                cmd.extend(["--run-dir", str(run_dir)])
            if verbose:
                cmd.append("--verbose")
            extract_step = _run_step(cmd, cwd=ROOT, tail_lines=120)
            ext = extract_step.to_dict()
            ext["run_dir_discovery"] = {
                "resolved_run_dir": str(run_dir) if run_dir is not None else "",
                "reason": run_reason,
                "trace": run_trace,
            }
            manifest["steps"]["extract"] = ext
        else:
            manifest["steps"]["extract"] = StepResult(
                status="warn",
                rc=0,
                cmd=[],
                duration_ms=0,
                stdout_tail=[],
                stderr_tail=[],
                notes=["dataset_missing_auto_extract_disabled"],
            ).to_dict()
    else:
        manifest["steps"]["extract"] = StepResult(
            status="ok",
            rc=0,
            cmd=[],
            duration_ms=0,
            stdout_tail=[],
            stderr_tail=[],
            notes=["dataset_already_exists"],
        ).to_dict()

    if not dataset_dir.exists():
        warnings.append("dataset_dir_unavailable")
        manifest["status"] = "fail"
        _write_json(out_dir / "pack_manifest.json", manifest)
        return 2, manifest

    # Step 2: baseline resolve
    current_day = _parse_date_str(date_value)
    baseline_dir, baseline_reason, baseline_trace = resolve_baseline_dataset(
        current_day=current_day,
        base_dir=base_dir.resolve(),
        baseline_dataset_arg=baseline_dataset_arg,
        baseline_mode=baseline_mode,
        lookback_days=lookback_days,
        auto_metrics=auto_metrics,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
        min_points=min_points,
        verbose=verbose,
    )
    manifest["baseline"] = {
        "requested": str(baseline_dataset_arg or ""),
        "mode": str(baseline_mode),
        "resolved_dataset_dir": str(baseline_dir) if baseline_dir is not None else "",
        "reason": baseline_reason,
        "trace": baseline_trace,
    }
    if baseline_dir is None:
        warnings.append(f"baseline_unavailable:{baseline_reason}")

    # Step 3: metrics
    metrics_state, metrics_obj = ensure_metrics(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        auto_metrics=auto_metrics,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
        min_points=min_points,
        verbose=verbose,
    )
    manifest["steps"]["metrics"] = metrics_state
    if metrics_obj is None:
        warnings.append("metrics_unavailable")

    # Step 4: leaderboard
    leaderboard_state, leaderboard_obj = ensure_leaderboard(
        base_dir=base_dir.resolve(),
        out_dir=out_dir,
        lookback_days=lookback_days,
        auto_leaderboard=auto_leaderboard,
        auto_metrics=auto_metrics,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
        min_points=min_points,
        verbose=verbose,
    )
    manifest["steps"]["leaderboard"] = leaderboard_state
    if leaderboard_obj is None and auto_leaderboard:
        warnings.append("leaderboard_unavailable")

    # Step 5: gate
    gate_state, gate_result = ensure_gate(
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        auto_gate=auto_gate,
        auto_metrics=auto_metrics,
        strict=strict,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
        min_points=min_points,
        verbose=verbose,
    )
    manifest["steps"]["gate"] = gate_state
    gate_status = ""
    if isinstance(gate_result, dict):
        gate_status = str(gate_result.get("status", "")).upper()
        manifest["gate_result_path"] = str((out_dir / "gate" / "gate_result.json").resolve())
    elif auto_gate and baseline_dir is not None:
        warnings.append("gate_result_unavailable")

    # Render markdown summary
    daily_md = render_daily_quant_report_md(
        daily_dir=daily_dir.resolve(),
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        strict=bool(strict),
        baseline_dir=baseline_dir,
        baseline_reason=baseline_reason,
        lookback_days=lookback_days,
        metrics_obj=metrics_obj,
        gate_result=gate_result,
        leaderboard_obj=leaderboard_obj,
        base_dir=base_dir.resolve(),
        warnings=warnings,
    )
    _write_text(out_dir / "daily_quant_report.md", daily_md)

    exit_code = 0
    if strict:
        strict_fail = False
        if baseline_dir is None:
            strict_fail = True
        if metrics_obj is None:
            strict_fail = True
        if auto_gate and baseline_dir is not None and gate_status == "FAIL":
            strict_fail = True
        if strict_fail:
            exit_code = 3

    manifest["status"] = "ok" if exit_code == 0 else "fail"
    manifest["gate_status"] = gate_status or ("SKIPPED" if not auto_gate else "")
    manifest["daily_md_path"] = str((out_dir / "daily_quant_report.md").resolve())
    _write_json(out_dir / "pack_manifest.json", manifest)
    return exit_code, manifest
