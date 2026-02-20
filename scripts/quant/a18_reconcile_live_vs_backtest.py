#!/usr/bin/env python3
"""A4-6/A4-7: reconcile live vs backtest daily summary and write attribution."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import parse_iso_to_utc, safe_read_json

ROOT = Path(__file__).resolve().parents[2]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_date(s: str) -> Optional[str]:
    text = str(s or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _discover_latest_date(daily_base: Path) -> Optional[str]:
    items: List[str] = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        d = _parse_date(p.stem)
        if d is not None:
            items.append(d)
    if not items:
        return None
    items.sort()
    return items[-1]


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    rows.append({str(k): str(v) for k, v in row.items() if k is not None})
    except Exception:
        return []
    return rows


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


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_name(path.name + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        return bak
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bak2 = path.with_name(path.name + f".{stamp}.bak")
    shutil.copy2(path, bak2)
    return bak2


def _resolve_pack_dir(quant_pack: Dict[str, Any], daily_base: Path, date_str: str) -> Path:
    pack_dir_raw = str(quant_pack.get("pack_dir") or "").strip()
    if pack_dir_raw:
        p = Path(pack_dir_raw)
        if not p.is_absolute():
            p = (daily_base / p).resolve()
        if p.exists() and p.is_dir():
            return p.resolve()
    return (daily_base / "quant_packs" / date_str).resolve()


def _load_metrics_from_quant_pack(quant_pack: Dict[str, Any], daily_base: Path, date_str: str) -> Dict[str, Any]:
    # best effort: infer metrics.json from quant_pack.pack_md_path or default quant_packs path
    md_path_raw = str(quant_pack.get("pack_md_path") or "").strip()
    cand: List[Path] = []
    if md_path_raw:
        p = Path(md_path_raw)
        if not p.is_absolute():
            p = (daily_base / p).resolve()
        if p.exists():
            cand.append((p.parent / "metrics" / "metrics.json").resolve())
    cand.append((daily_base / "quant_packs" / date_str / "metrics" / "metrics.json").resolve())
    for c in cand:
        obj = safe_read_json(c)
        if isinstance(obj, dict):
            return obj
    return {}


def _resolve_dataset_dir(quant_pack: Dict[str, Any], pack_dir: Path) -> Optional[Path]:
    artifacts = quant_pack.get("artifacts") if isinstance(quant_pack.get("artifacts"), dict) else {}
    ds_raw = str(artifacts.get("dataset_dir") or "").strip() if isinstance(artifacts, dict) else ""
    if ds_raw:
        p = Path(ds_raw)
        if not p.is_absolute():
            p = (pack_dir / ds_raw).resolve()
        if p.exists() and p.is_dir():
            return p.resolve()
    p2 = (pack_dir / "run_dataset").resolve()
    if p2.exists() and p2.is_dir():
        return p2
    return None


def _resolve_candidate_path(raw: str, *, pack_dir: Path, daily_base: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((pack_dir / p).resolve())
        candidates.append((daily_base / p).resolve())
        candidates.append((ROOT / p).resolve())
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


def _safe_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _extract_path_from_md(md_path: Path, key: str) -> str:
    if not md_path.exists():
        return ""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    pat = re.compile(rf"^\s*-\s*{re.escape(key)}\s*:\s*`([^`]+)`\s*$", re.IGNORECASE | re.MULTILINE)
    m = pat.search(text)
    if not m:
        return ""
    return str(m.group(1) or "").strip()


def _compute_equity_stats(series: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in series if _num_or_none(x) is not None]
    if len(vals) < 2:
        return None, None
    start = vals[0]
    end = vals[-1]
    if start == 0:
        total_return = None
    else:
        total_return = float((end / start) - 1.0)
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v / peak) - 1.0
            if dd < max_dd:
                max_dd = dd
    return total_return, float(max_dd)


def _live_stats_from_dataset_equity(dataset_dir: Path) -> Tuple[Optional[float], Optional[float], List[str]]:
    warnings: List[str] = []
    eq_path = (dataset_dir / "equity_curve.csv").resolve()
    rows = _read_csv_rows(eq_path)
    if not rows:
        return None, None, [f"missing_equity_curve_csv:{eq_path}"]
    prepared: List[Tuple[str, float]] = []
    for row in rows:
        t = str(row.get("time_utc") or row.get("time") or row.get("date") or "").strip()
        v = _num_or_none(row.get("total_equity") or row.get("equity") or row.get("close_equity"))
        if not t or v is None:
            continue
        dt = parse_iso_to_utc(t)
        if dt is None:
            # allow date-only fallback
            prepared.append((t, float(v)))
        else:
            prepared.append((dt.isoformat(), float(v)))
    prepared.sort(key=lambda x: x[0])
    if len(prepared) < 2:
        return None, None, warnings + [f"insufficient_equity_points:{len(prepared)}"]
    vals = [x[1] for x in prepared]
    ret, dd = _compute_equity_stats(vals)
    return ret, dd, warnings


def _live_stats_from_equity_history(report_obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], List[str]]:
    warnings: List[str] = []
    rows = report_obj.get("equity_history")
    if not isinstance(rows, list):
        return None, None, ["missing_equity_history"]
    prepared: List[Tuple[str, float]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        t = str(item.get("ts") or item.get("time") or item.get("time_utc") or item.get("timestamp") or "").strip()
        v = _num_or_none(item.get("equity") or item.get("total_equity") or item.get("value"))
        if not t or v is None:
            continue
        dt = parse_iso_to_utc(t)
        prepared.append(((dt.isoformat() if dt else t), float(v)))
    prepared.sort(key=lambda x: x[0])
    if len(prepared) < 2:
        return None, None, warnings + [f"insufficient_equity_history_points:{len(prepared)}"]
    vals = [x[1] for x in prepared]
    ret, dd = _compute_equity_stats(vals)
    return ret, dd, warnings


def _extract_trading_costs(
    *,
    trades_total: int,
    metrics_obj: Dict[str, Any],
    dataset_dir: Optional[Path],
) -> Tuple[float, float, List[str]]:
    warnings: List[str] = []
    trading = metrics_obj.get("trading") if isinstance(metrics_obj.get("trading"), dict) else {}
    turnover = _num_or_none(trading.get("turnover_notional"))
    total_cost = _num_or_none(trading.get("total_cost"))

    if trades_total <= 0:
        return 0.0, 0.0, warnings

    if dataset_dir is not None:
        trades_csv = (dataset_dir / "trades.csv").resolve()
        rows = _read_csv_rows(trades_csv)
        if rows:
            turn_sum = 0.0
            cost_sum = 0.0
            cost_found = False
            for r in rows:
                n = _num_or_none(
                    r.get("trade_notional")
                    or r.get("notional")
                    or r.get("value")
                    or r.get("desired_trade_value")
                    or r.get("turnover_notional")
                )
                if n is not None:
                    turn_sum += abs(float(n))
                c = _num_or_none(r.get("cost") or r.get("total_cost") or r.get("cost_est_total"))
                if c is not None:
                    cost_found = True
                    cost_sum += float(c)
            turnover = float(turn_sum)
            total_cost = float(cost_sum) if cost_found else 0.0
            if not cost_found:
                warnings.append("trades_csv_cost_missing_default_0")
        else:
            warnings.append(f"missing_trades_csv:{trades_csv}")

    if turnover is None:
        turnover = 0.0
        warnings.append("turnover_missing_default_0")
    if total_cost is None:
        total_cost = 0.0
        warnings.append("total_cost_missing_default_0")
    return float(turnover), float(total_cost), warnings


def _extract_live_metrics(
    *,
    report_obj: Dict[str, Any],
    quant_pack: Dict[str, Any],
    metrics_obj: Dict[str, Any],
    dataset_dir: Optional[Path],
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    summary = quant_pack.get("summary") if isinstance(quant_pack.get("summary"), dict) else {}
    trading = metrics_obj.get("trading") if isinstance(metrics_obj.get("trading"), dict) else {}
    risk = metrics_obj.get("risk") if isinstance(metrics_obj.get("risk"), dict) else {}
    live_ret = _num_or_none(summary.get("total_return"))
    live_dd = _num_or_none(summary.get("max_drawdown") if "max_drawdown" in summary else risk.get("max_drawdown"))
    trades_total = int(_num_or_none(summary.get("trades_total") if "trades_total" in summary else trading.get("trades_total")) or 0)

    if live_ret is None or live_dd is None:
        # fallback 1: metrics object itself
        if live_ret is None:
            live_ret = _num_or_none((metrics_obj.get("performance") or {}).get("total_return")) if isinstance(metrics_obj, dict) else None
        if live_dd is None:
            live_dd = _num_or_none((metrics_obj.get("risk") or {}).get("max_drawdown")) if isinstance(metrics_obj, dict) else None

    if (live_ret is None or live_dd is None) and dataset_dir is not None:
        ds_ret, ds_dd, ds_warn = _live_stats_from_dataset_equity(dataset_dir)
        warnings.extend(ds_warn)
        if live_ret is None:
            live_ret = ds_ret
        if live_dd is None:
            live_dd = ds_dd

    if live_ret is None or live_dd is None:
        eh_ret, eh_dd, eh_warn = _live_stats_from_equity_history(report_obj)
        warnings.extend(eh_warn)
        if live_ret is None:
            live_ret = eh_ret
        if live_dd is None:
            live_dd = eh_dd

    if live_ret is None:
        warnings.append("missing_live_total_return")
    if live_dd is None:
        warnings.append("missing_live_max_drawdown")

    turnover_notional, total_cost, tw = _extract_trading_costs(
        trades_total=trades_total,
        metrics_obj=metrics_obj,
        dataset_dir=dataset_dir,
    )
    warnings.extend(tw)

    return (
        {
            "total_return": live_ret,
            "max_drawdown": live_dd,
            "trades_total": trades_total,
            "turnover_notional": float(turnover_notional),
            "total_cost": float(total_cost),
        },
        warnings,
    )


def _extract_backtest_metrics(quant_pack: Dict[str, Any], pack_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    bt = quant_pack.get("backtest_from_run") if isinstance(quant_pack.get("backtest_from_run"), dict) else {}
    if not bt:
        m = safe_read_json((pack_dir / "backtest_from_run" / "backtest" / "backtest_manifest.json").resolve()) or {}
        summ = m.get("summary") if isinstance(m.get("summary"), dict) else {}
        csum = m.get("cost_summary") if isinstance(m.get("cost_summary"), dict) else {}
        bt = {
            "total_return": summ.get("total_return"),
            "max_drawdown": summ.get("max_drawdown"),
            "trade_rows": csum.get("trade_rows"),
            "turnover_notional": csum.get("total_turnover_notional"),
            "total_cost": csum.get("total_cost"),
        }
        if not m:
            warnings.append("missing_backtest_metrics")
    return (
        {
            "total_return": _num_or_none(bt.get("total_return")),
            "max_drawdown": _num_or_none(bt.get("max_drawdown")),
            "trade_rows": int(_num_or_none(bt.get("trade_rows")) or 0),
            "turnover_notional": _num_or_none(bt.get("turnover_notional")),
            "total_cost": _num_or_none(bt.get("total_cost")),
        },
        warnings,
    )


def _load_gating_top3(metrics_obj: Dict[str, Any], pack_dir: Path, dataset_dir: Optional[Path]) -> List[Dict[str, Any]]:
    # preferred: quant_packs/<date>/gating_summary.csv
    import csv

    candidates = [
        (pack_dir / "gating_summary.csv").resolve(),
        (pack_dir / "leaderboard" / "gating_summary.csv").resolve(),
    ]
    rows_out: List[Dict[str, Any]] = []
    for p in candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        reason = str(row.get("reason") or row.get("skip_reason") or "").strip()
                        cnt = int(_num_or_none(row.get("count")) or 0)
                        if reason:
                            rows_out.append({"reason": reason, "count": cnt})
                if rows_out:
                    rows_out.sort(key=lambda x: (int(x.get("count", 0)), str(x.get("reason", ""))), reverse=True)
                    return rows_out[:3]
            except Exception:
                pass

    # fallback: metrics.gating.summary.top3
    top = (((metrics_obj.get("gating") or {}).get("summary") or {}).get("top3") or []) if isinstance(metrics_obj, dict) else []
    if isinstance(top, list):
        out: List[Dict[str, Any]] = []
        for item in top:
            if isinstance(item, dict):
                out.append({"reason": str(item.get("reason", "")), "count": int(_num_or_none(item.get("count")) or 0)})
        if out:
            return out[:3]

    # fallback: run_dataset/cycles.csv
    if dataset_dir is not None:
        cycles_path = (dataset_dir / "cycles.csv").resolve()
        rows = _read_csv_rows(cycles_path)
        if rows:
            counts: Dict[str, int] = {}
            for row in rows:
                reason = (
                    str(row.get("skip_reason", "") or "").strip()
                    or str(row.get("abort_reason", "") or "").strip()
                    or str(row.get("cov_gate_reason", "") or "").strip()
                    or str(row.get("decision_path", "") or "").strip()
                )
                if not reason:
                    continue
                counts[reason] = int(counts.get(reason, 0)) + 1
            if counts:
                ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                return [{"reason": r, "count": int(c)} for r, c in ranked[:3]]
    return []


def _load_gate_status_from_files(pack_dir: Path) -> str:
    gate_obj = safe_read_json((pack_dir / "gate" / "gate_result.json").resolve()) or {}
    if isinstance(gate_obj, dict):
        s = str(gate_obj.get("status", "") or "").strip()
        if s:
            return s
    gate_obj2 = safe_read_json((pack_dir / "gate_result.json").resolve()) or {}
    if isinstance(gate_obj2, dict):
        s = str(gate_obj2.get("status", "") or "").strip()
        if s:
            return s
    return ""


def _discover_baseline_dataset(daily_base: Path, date_str: str) -> Optional[Path]:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None
    # try previous 7 days for a stable baseline
    for days in range(1, 8):
        dd = d.fromordinal(d.toordinal() - days).isoformat()
        p = (daily_base / "quant_packs" / dd / "run_dataset").resolve()
        if p.exists() and p.is_dir():
            return p
    return None


def _infer_gate_datasets(
    *,
    quant_pack: Dict[str, Any],
    pack_dir: Path,
    daily_base: Path,
    date_str: str,
    dataset_dir_hint: Optional[Path],
    warnings: List[str],
) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    trace: List[str] = []
    candidate: Optional[Path] = dataset_dir_hint if dataset_dir_hint is not None else None
    baseline: Optional[Path] = None

    # 0) explicit from quant_pack artifacts-like hints
    if candidate is None:
        for key in ("dataset_dir", "candidate_dataset_dir", "run_dataset_dir"):
            v = str(quant_pack.get(key, "") or "").strip()
            if not v:
                continue
            p = _resolve_candidate_path(v, pack_dir=pack_dir, daily_base=daily_base)
            trace.append(f"quant_pack.{key}")
            if p is not None:
                candidate = p
                break
    for key in ("baseline_dataset_dir", "baseline_dir"):
        v = str(quant_pack.get(key, "") or "").strip()
        if not v:
            continue
        p = _resolve_candidate_path(v, pack_dir=pack_dir, daily_base=daily_base)
        trace.append(f"quant_pack.{key}")
        if p is not None:
            baseline = p
            break

    # 1) pipeline manifest / pack manifest
    for mf_name in ("pipeline_manifest.json", "pack_manifest.json"):
        mf = safe_read_json((pack_dir / mf_name).resolve()) or {}
        if not isinstance(mf, dict):
            continue
        if candidate is None:
            for path in (
                "dataset_dir",
                "candidate_dataset_dir",
                "run_dataset_dir",
                "steps.build_pack.dataset_dir",
                "step_results.build_pack.dataset_dir",
            ):
                raw = _safe_get(mf, path)
                p = _resolve_candidate_path(str(raw or ""), pack_dir=pack_dir, daily_base=daily_base)
                if p is not None:
                    candidate = p
                    trace.append(f"{mf_name}:{path}")
                    break
        if baseline is None:
            for path in (
                "baseline_dataset_dir",
                "baseline_dir",
                "baseline.resolved_dataset_dir",
                "step_results.build_pack.baseline_dataset_dir",
                "step_results.build_pack.baseline_dir",
            ):
                raw = _safe_get(mf, path)
                p = _resolve_candidate_path(str(raw or ""), pack_dir=pack_dir, daily_base=daily_base)
                if p is not None:
                    baseline = p
                    trace.append(f"{mf_name}:{path}")
                    break

    # 2) parse daily quant report markdown
    md_path = (pack_dir / "daily_quant_report.md").resolve()
    if candidate is None:
        raw = _extract_path_from_md(md_path, "Dataset Dir")
        p = _resolve_candidate_path(raw, pack_dir=pack_dir, daily_base=daily_base)
        if p is not None:
            candidate = p
            trace.append("daily_quant_report.md:Dataset Dir")
    if baseline is None:
        raw = _extract_path_from_md(md_path, "baseline")
        p = _resolve_candidate_path(raw, pack_dir=pack_dir, daily_base=daily_base)
        if p is not None:
            baseline = p
            trace.append("daily_quant_report.md:baseline")

    # 3) deterministic fallback baseline
    if baseline is None:
        baseline = _discover_baseline_dataset(daily_base, date_str)
        if baseline is not None:
            trace.append("fallback:prev_day_dataset")

    if candidate is None:
        warnings.append("evidence_auto:gate_candidate_missing")
    if baseline is None:
        warnings.append("evidence_auto:gate_baseline_missing")
    return candidate, baseline, trace


def _run_cmd(cmd: List[str]) -> int:
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        return int(proc.returncode)
    except Exception:
        return 2


def _resolve_gate_status(
    quant_pack: Dict[str, Any],
    pack_dir: Path,
    *,
    auto_evidence: bool,
    candidate_dataset_dir: Optional[Path],
    baseline_dataset_dir: Optional[Path],
    warnings: List[str],
) -> str:
    gate_status = ""
    if isinstance(quant_pack.get("gate_result"), dict):
        gate_status = str((quant_pack.get("gate_result") or {}).get("status", "") or "")
    if not gate_status and isinstance(quant_pack.get("summary"), dict):
        gate_status = str((quant_pack.get("summary") or {}).get("gate_status", "") or "")
    if not gate_status:
        gate_status = _load_gate_status_from_files(pack_dir)
    if gate_status:
        return gate_status

    if auto_evidence:
        warnings.append("evidence_auto:gate_status_attempted")
        candidate_ds = candidate_dataset_dir
        baseline_ds = baseline_dataset_dir
        if candidate_ds is not None and candidate_ds.exists() and baseline_ds is not None and baseline_ds.exists():
            cmd = [
                sys.executable,
                str((ROOT / "scripts" / "quant" / "a5_quant_gate.py").resolve()),
                "--baseline",
                str(baseline_ds),
                "--candidate",
                str(candidate_ds),
                "--out-dir",
                str((pack_dir / "gate").resolve()),
                "--auto-metrics",
            ]
            rc = _run_cmd(cmd)
            warnings.append(f"evidence_auto:gate_status_a5_rc={rc}")
            gate_status = _load_gate_status_from_files(pack_dir)
            if gate_status:
                return gate_status
            return "ERROR"
        else:
            miss = []
            if candidate_ds is None or not candidate_ds.exists():
                miss.append("candidate")
            if baseline_ds is None or not baseline_ds.exists():
                miss.append("baseline")
            warnings.append("evidence_auto:gate_status_missing_inputs:" + ",".join(miss))
        return "MISSING_INPUT"

    return "NA"


def _load_replay_status_from_files(pack_dir: Path) -> str:
    rdm = safe_read_json((pack_dir / "replay_drift" / "replay_drift_manifest.json").resolve()) or {}
    if isinstance(rdm, dict):
        s = str(rdm.get("status", "") or "")
        if not s and isinstance(rdm.get("summary"), dict):
            s = str((rdm.get("summary") or {}).get("status", "") or "")
        if s:
            return s
    dgr = safe_read_json((pack_dir / "replay_drift" / "drift_gate_result.json").resolve()) or {}
    if isinstance(dgr, dict):
        s = str(dgr.get("status", "") or "")
        if s:
            return s
    return ""


def _resolve_replay_drift_status(
    quant_pack: Dict[str, Any],
    pack_dir: Path,
    *,
    auto_evidence: bool,
    daily_base: Path,
    date_str: str,
    warnings: List[str],
) -> str:
    if isinstance(quant_pack.get("replay_drift"), dict):
        s = str((quant_pack.get("replay_drift") or {}).get("status", "") or "")
        if s:
            return s
    s = _load_replay_status_from_files(pack_dir)
    if s:
        return s

    if auto_evidence:
        warnings.append("evidence_auto:replay_drift_attempted")
        cmd = [
            sys.executable,
            str((ROOT / "scripts" / "quant" / "a12_attach_replay_drift_to_daily.py").resolve()),
            "--daily-base",
            str(daily_base),
            "--date",
            str(date_str),
        ]
        rc = _run_cmd(cmd)
        warnings.append(f"evidence_auto:replay_drift_a12_rc={rc}")
        # check daily json first because a12 writes back there
        report_obj = safe_read_json((daily_base / f"{date_str}.json").resolve()) or {}
        qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
        if isinstance(qp.get("replay_drift"), dict):
            s2 = str((qp.get("replay_drift") or {}).get("status", "") or "")
            if s2:
                return s2
        s3 = _load_replay_status_from_files(pack_dir)
        if s3:
            return s3
        return "NOT_RUN"

    return "NA"


def _build_reconcile(
    *,
    quant_pack: Dict[str, Any],
    report_obj: Dict[str, Any],
    daily_base: Path,
    date_str: str,
    auto_evidence: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    metrics_obj = _load_metrics_from_quant_pack(quant_pack, daily_base, date_str)
    pack_dir = _resolve_pack_dir(quant_pack, daily_base, date_str)
    dataset_dir = _resolve_dataset_dir(quant_pack, pack_dir)
    candidate_dataset_dir, baseline_dataset_dir, infer_trace = _infer_gate_datasets(
        quant_pack=quant_pack,
        pack_dir=pack_dir,
        daily_base=daily_base,
        date_str=date_str,
        dataset_dir_hint=dataset_dir,
        warnings=warnings,
    )
    if dataset_dir is None and candidate_dataset_dir is not None:
        dataset_dir = candidate_dataset_dir
    if auto_evidence and infer_trace:
        warnings.append("evidence_auto:infer_trace=" + " > ".join(infer_trace[:6]))

    live, live_warn = _extract_live_metrics(
        report_obj=report_obj,
        quant_pack=quant_pack,
        metrics_obj=metrics_obj,
        dataset_dir=dataset_dir,
    )
    warnings.extend(live_warn)
    backtest, bt_warn = _extract_backtest_metrics(quant_pack, pack_dir)
    warnings.extend(bt_warn)

    live_ret = _num_or_none(live.get("total_return"))
    bt_ret = _num_or_none(backtest.get("total_return"))
    live_dd = _num_or_none(live.get("max_drawdown"))
    bt_dd = _num_or_none(backtest.get("max_drawdown"))
    live_turn = _num_or_none(live.get("turnover_notional"))
    bt_turn = _num_or_none(backtest.get("turnover_notional"))
    live_cost = _num_or_none(live.get("total_cost"))
    bt_cost = _num_or_none(backtest.get("total_cost"))

    missing_for_gap: List[str] = []
    def _gap(a: Optional[float], b: Optional[float], label: str) -> Optional[float]:
        if a is None or b is None:
            missing_for_gap.append(label)
            return None
        return float(a - b)

    gaps = {
        "return_gap_live_minus_backtest": _gap(live_ret, bt_ret, "return_gap"),
        "drawdown_gap": _gap(live_dd, bt_dd, "drawdown_gap"),
        "turnover_gap": _gap(live_turn, bt_turn, "turnover_gap"),
        "cost_gap": _gap(live_cost, bt_cost, "cost_gap"),
    }
    if missing_for_gap:
        warnings.append("gaps_missing:" + ",".join(sorted(set(missing_for_gap))))

    evidence_gating_top3 = _load_gating_top3(metrics_obj, pack_dir, dataset_dir)
    if auto_evidence and not evidence_gating_top3:
        warnings.append("evidence_auto:gating_top3_missing")
    replay_status = _resolve_replay_drift_status(
        quant_pack,
        pack_dir,
        auto_evidence=auto_evidence,
        daily_base=daily_base,
        date_str=date_str,
        warnings=warnings,
    )
    gate_status = _resolve_gate_status(
        quant_pack,
        pack_dir,
        auto_evidence=auto_evidence,
        candidate_dataset_dir=candidate_dataset_dir,
        baseline_dataset_dir=baseline_dataset_dir,
        warnings=warnings,
    )

    drivers: List[str] = []
    notes: List[str] = []

    bt_trades = int(backtest.get("trade_rows") or 0)
    live_trades = int(live.get("trades_total") or 0)
    if live_trades <= 0 and bt_trades > 0:
        drivers.append("execution/gating drag (no-trade day)")
        notes.append("Live traded ~0 while backtest had target-driven trades.")

    cost_gap = _num_or_none(gaps.get("cost_gap"))
    if cost_gap is not None and cost_gap > 0:
        drivers.append("cost/turnover drag")
        notes.append("Live cost exceeds backtest cost.")

    if str(replay_status).upper() == "FAIL":
        drivers.append("nondeterminism / data drift")
        notes.append("Replay drift status indicates mismatch risk.")

    if str(gate_status).upper() == "FAIL":
        drivers.append("quant gate fail / risk control limiting")
        notes.append("Gate status is FAIL, likely limiting execution.")

    rg = _num_or_none(gaps.get("return_gap_live_minus_backtest"))
    if rg is not None and rg < -0.005:
        drivers.append("live execution underperformed backtest")
        notes.append("Live return is materially lower than backtest.")

    if not drivers:
        drivers = ["no strong drag signal"]
        notes = ["Live/backtest deltas are small or insufficient data."]

    status = "OK"
    if live_ret is None and bt_ret is None:
        status = "MISSING"

    evidence_payload = {
        "gating_top3": evidence_gating_top3[:3] if isinstance(evidence_gating_top3, list) else [],
        "replay_drift_status": replay_status or ("NOT_RUN" if auto_evidence else "NA"),
        "gate_status": gate_status or ("NOT_RUN" if auto_evidence else "NA"),
    }

    return {
        "status": status,
        "generated_utc": _now_utc_iso(),
        "live": live,
        "backtest": backtest,
        "gaps": gaps,
        "attribution": {
            "likely_driver_top3": drivers[:3],
            "notes": " ".join(notes[:3]),
        },
        "evidence": evidence_payload,
        "evidence_summary": evidence_payload,
        "warnings": sorted(set(warnings)),
    }


def reconcile_live_vs_backtest(
    *,
    daily_base: Path,
    date_str: str,
    strict: bool = False,
    auto_evidence: bool = False,
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    daily_base = daily_base.resolve()
    date_norm = _parse_date(date_str)
    if not date_norm:
        return 2, {"error": f"invalid date: {date_str}"}
    daily_path = (daily_base / f"{date_norm}.json").resolve()
    report_obj = safe_read_json(daily_path)
    if not isinstance(report_obj, dict):
        if strict:
            return 2, {"error": f"missing/invalid daily report: {daily_path}"}
        report_obj = {"date": date_norm, "summary": {}}

    quant_pack = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    reconcile = _build_reconcile(
        quant_pack=quant_pack,
        report_obj=report_obj,
        daily_base=daily_base,
        date_str=date_norm,
        auto_evidence=bool(auto_evidence),
    )
    if reconcile["status"] == "MISSING" and strict:
        return 2, {"error": "insufficient data for reconcile in strict mode"}

    quant_pack["reconcile"] = reconcile
    report_obj["quant_pack"] = quant_pack
    report_obj["updated_at_utc"] = _now_utc_iso()

    bak = _backup_file(daily_path)
    _write_json_atomic(daily_path, report_obj)

    out_dir = (daily_base / "quant_packs" / date_norm / "reconcile").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "daily_base": str(daily_base),
        "date": date_norm,
        "daily_report_path": str(daily_path),
        "daily_report_backup": str(bak) if bak else "",
        "status": reconcile.get("status"),
        "auto_evidence": bool(auto_evidence),
        "reconcile": reconcile,
    }
    _write_json_atomic(out_dir / "reconcile_manifest.json", manifest)

    if verbose:
        print(f"[A18] date={date_norm} status={reconcile.get('status')} report={daily_path}")
        print(f"[A18] out_dir={out_dir}")
    rc = 0 if reconcile.get("status") == "OK" else 1
    return rc, manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconcile daily live vs backtest and write attribution summary.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--auto-evidence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2
    date_norm = _parse_date(args.date) or _discover_latest_date(daily_base)
    if not date_norm:
        print(f"[ERROR] no valid date found under {daily_base}")
        return 2
    rc, info = reconcile_live_vs_backtest(
        daily_base=daily_base,
        date_str=date_norm,
        strict=bool(args.strict),
        auto_evidence=bool(args.auto_evidence),
        verbose=bool(args.verbose),
    )
    if rc == 2:
        print(f"[ERROR] {info.get('error', 'reconcile failed')}")
    elif args.verbose:
        print("[PASS] a18_reconcile_live_vs_backtest")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
