#!/usr/bin/env python3
"""Execution blocker and no-trade attribution helpers for daily quant reports."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_io_utils import safe_read_json


BLOCKER_TAXONOMY: List[str] = [
    "market_closed",
    "attempt_cooldown",
    "stale_abort",
    "risk_gate_abort",
    "cov_gate_abort",
    "already_balanced_or_filtered",
    "trades_executed",
    "unknown",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_date_str(s: str) -> Optional[str]:
    text = str(s or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def discover_latest_date(daily_base: Path) -> Optional[str]:
    days: List[str] = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        d = parse_date_str(p.stem)
        if d:
            days.append(d)
    if not days:
        return None
    days.sort()
    return days[-1]


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


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def backup_file(path: Path) -> Optional[Path]:
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    rows.append({str(k): str(v) for k, v in row.items()})
    except Exception:
        return []
    return rows


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def normalize_blocker_reason(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"

    if "trades_executed" in s or ("trade" in s and ("executed" in s or "filled" in s)):
        return "trades_executed"

    if "market_closed" in s or "market closed" in s or "session_closed" in s:
        return "market_closed"

    if "attempt_cooldown" in s or ("cooldown" in s and "attempt" in s) or "cooldown_guard" in s:
        return "attempt_cooldown"

    if "stale_abort" in s or ("stale" in s and "abort" in s):
        return "stale_abort"

    if (
        "cov_gate" in s
        or "portfolio_cov_rc_limit" in s
        or "max_rc_fraction" in s
        or "max_rc_ticker" in s
        or "rc_limit" in s
    ):
        return "cov_gate_abort"

    if "risk_gate" in s or ("risk" in s and "abort" in s):
        return "risk_gate_abort"

    if (
        "already_balanced" in s
        or "already balanced" in s
        or "filtered" in s
        or "below_threshold" in s
        or "no_trade_candidate" in s
    ):
        return "already_balanced_or_filtered"

    return "unknown"


def _cycle_has_trades(row: Dict[str, str]) -> bool:
    numeric_keys = (
        "trades_count",
        "num_trades",
        "trade_count",
        "executed_trades",
        "n_trades",
        "filled_trades",
    )
    for key in numeric_keys:
        n = _num_or_none(row.get(key))
        if n is not None and n > 0:
            return True

    bool_keys = ("has_trades", "did_trade", "trades_executed", "executed")
    for key in bool_keys:
        if _boolish(row.get(key)):
            return True

    status_blob = " ".join(
        str(row.get(k, ""))
        for k in ("status", "decision", "decision_path", "reason", "skip_reason", "rebalance_skip_reason")
    ).lower()
    if "trades_executed" in status_blob:
        return True
    return False


def _cycle_reason_from_row(row: Dict[str, str]) -> str:
    if _cycle_has_trades(row):
        return "trades_executed"

    for key in (
        "blocker_reason",
        "skip_reason",
        "rebalance_skip_reason",
        "abort_reason",
        "cov_gate_reason",
        "reason",
        "decision",
        "decision_path",
        "status",
    ):
        v = row.get(key)
        r = normalize_blocker_reason(v)
        if r != "unknown":
            return r

    return "unknown"


def _longest_streak(reasons: List[str]) -> Dict[str, Any]:
    best_reason = "unknown"
    best_len = 0
    cur_reason = ""
    cur_len = 0
    for r in reasons:
        if r == cur_reason:
            cur_len += 1
        else:
            cur_reason = r
            cur_len = 1
        if cur_len > best_len:
            best_len = cur_len
            best_reason = r
    return {"reason": str(best_reason), "len": int(best_len)}


def _sorted_top3(counts: Dict[str, int], total: int) -> List[Dict[str, Any]]:
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    out: List[Dict[str, Any]] = []
    denom = max(1, int(total))
    for reason, count in items:
        if int(count) <= 0:
            continue
        out.append({"reason": str(reason), "count": int(count), "ratio": float(int(count) / denom)})
        if len(out) >= 3:
            break
    return out


def _resolve_candidate_path(raw: Any, *, pack_dir: Path, daily_base: Path) -> Optional[Path]:
    v = str(raw or "").strip()
    if not v:
        return None
    p = Path(v)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p.resolve())
    else:
        candidates.append((pack_dir / p).resolve())
        candidates.append((daily_base / p).resolve())
        candidates.append((daily_base.parent / p).resolve())
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def resolve_dataset_dir(report_obj: Dict[str, Any], *, daily_base: Path, date_str: str) -> Path:
    pack_dir = (daily_base / "quant_packs" / date_str).resolve()
    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    art = qp.get("artifacts") if isinstance(qp.get("artifacts"), dict) else {}
    for raw in (
        art.get("dataset_dir"),
        qp.get("dataset_dir"),
        qp.get("run_dataset_dir"),
        (qp.get("backtest_from_run") or {}).get("dataset_dir")
        if isinstance(qp.get("backtest_from_run"), dict)
        else None,
    ):
        p = _resolve_candidate_path(raw, pack_dir=pack_dir, daily_base=daily_base)
        if p is not None and p.exists():
            return p
    return (pack_dir / "run_dataset").resolve()


def _extract_fallback_top3(report_obj: Dict[str, Any], metrics_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    rec = qp.get("reconcile") if isinstance(qp.get("reconcile"), dict) else {}
    for ev_key in ("evidence_summary", "evidence"):
        ev = rec.get(ev_key) if isinstance(rec.get(ev_key), dict) else {}
        top3 = ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else []
        if top3:
            return [x for x in top3 if isinstance(x, dict)]

    gating = metrics_obj.get("gating") if isinstance(metrics_obj.get("gating"), dict) else {}
    summary = gating.get("summary") if isinstance(gating.get("summary"), dict) else {}
    for key in ("top3", "top_reasons", "reasons"):
        top3 = summary.get(key) if isinstance(summary.get(key), list) else []
        if top3:
            return [x for x in top3 if isinstance(x, dict)]
    return []


def _determine_trades_total(report_obj: Dict[str, Any], metrics_obj: Dict[str, Any], trades_rows: List[Dict[str, str]]) -> int:
    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    summary = qp.get("summary") if isinstance(qp.get("summary"), dict) else {}
    for raw in (
        summary.get("trades_total"),
        qp.get("trades_total"),
        (metrics_obj.get("trading") or {}).get("trades_total")
        if isinstance(metrics_obj.get("trading"), dict)
        else None,
    ):
        n = _num_or_none(raw)
        if n is not None:
            return int(n)
    return int(len(trades_rows))


def compute_exec_blockers_payload(
    *,
    date_str: str,
    cycles_rows: List[Dict[str, str]],
    fallback_top3: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    counts = {k: 0 for k in BLOCKER_TAXONOMY}
    source = "missing"
    reasons_seq: List[str] = []
    cycles_total = 0

    if cycles_rows:
        source = "cycles.csv"
        for row in cycles_rows:
            reason = _cycle_reason_from_row(row)
            if reason not in counts:
                reason = "unknown"
            counts[reason] += 1
            reasons_seq.append(reason)
        cycles_total = int(len(cycles_rows))
    elif fallback_top3:
        source = "fallback"
        for item in fallback_top3:
            reason = normalize_blocker_reason(item.get("reason"))
            cnt = int(_num_or_none(item.get("count")) or 0)
            if cnt <= 0:
                continue
            if reason not in counts:
                reason = "unknown"
            counts[reason] += cnt
            cycles_total += cnt
        if cycles_total <= 0:
            cycles_total = int(sum(counts.values()))
        warnings.append("cycles_csv_missing:fallback_top3_used")
    else:
        warnings.append("cycles_csv_missing:no_fallback")

    cycles_blocked = max(0, int(cycles_total - counts.get("trades_executed", 0)))
    blocked_ratio = float(cycles_blocked / max(1, cycles_total))
    blocker_counts = {k: int(v) for k, v in counts.items() if k != "trades_executed"}
    top3 = _sorted_top3(blocker_counts if sum(blocker_counts.values()) > 0 else counts, cycles_total)

    if reasons_seq:
        longest = _longest_streak(reasons_seq)
    else:
        top_reason = top3[0]["reason"] if top3 else "unknown"
        top_count = top3[0]["count"] if top3 else 0
        longest = {"reason": str(top_reason), "len": int(top_count)}

    status = "MISSING"
    if source == "cycles.csv":
        status = "WARN" if warnings else "OK"
    elif source == "fallback":
        status = "WARN"

    return {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "date": str(date_str),
        "status": str(status),
        "source": str(source),
        "cycles_total": int(cycles_total),
        "cycles_blocked": int(cycles_blocked),
        "blocked_ratio": float(blocked_ratio),
        "top3": top3,
        "counts": {k: int(counts.get(k, 0)) for k in BLOCKER_TAXONOMY},
        "longest_streak": longest,
        "warnings": list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
    }


def infer_no_trade_payload(
    *,
    date_str: str,
    trades_total: int,
    exec_blockers: Dict[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    is_no_trade_day = int(trades_total) == 0
    top3 = exec_blockers.get("top3") if isinstance(exec_blockers.get("top3"), list) else []
    top_blocker = top3[0] if top3 and isinstance(top3[0], dict) else {}
    source = str(exec_blockers.get("source", "missing") or "missing")

    primary_reason = "unknown"
    if not is_no_trade_day:
        primary_reason = "trades_executed"
    else:
        top_reason = str(top_blocker.get("reason", "") or "")
        top_ratio = _num_or_none(top_blocker.get("ratio"))
        if top_reason and top_reason != "trades_executed" and top_ratio is not None and top_ratio >= 0.6:
            primary_reason = top_reason
        else:
            counts = exec_blockers.get("counts") if isinstance(exec_blockers.get("counts"), dict) else {}
            mc = int(_num_or_none((counts or {}).get("market_closed")) or 0)
            ac = int(_num_or_none((counts or {}).get("attempt_cooldown")) or 0)
            if mc >= ac and mc > 0:
                primary_reason = "market_closed"
            elif ac > 0:
                primary_reason = "attempt_cooldown"
            else:
                primary_reason = "unknown"

    if str(exec_blockers.get("status", "")).upper() == "MISSING":
        warnings.append("exec_blockers_missing")
    if is_no_trade_day and source == "missing":
        warnings.append("no_trade_reason_inference_weak")

    notes = "No trade day inferred from cycle blockers." if is_no_trade_day else "Trades executed on this day."
    if primary_reason == "unknown" and is_no_trade_day:
        notes = "No-trade day but blocker evidence is insufficient."

    return {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "date": str(date_str),
        "is_no_trade_day": bool(is_no_trade_day),
        "trades_total": int(trades_total),
        "primary_reason": str(primary_reason),
        "evidence": {
            "top_blocker": {
                "reason": str(top_blocker.get("reason", "") if isinstance(top_blocker, dict) else ""),
                "ratio": _num_or_none(top_blocker.get("ratio")) if isinstance(top_blocker, dict) else None,
            },
            "exec_blockers_source": source,
            "notes": notes,
        },
        "warnings": list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
    }


def _render_exec_blockers_md(exec_blockers: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Execution Blockers")
    lines.append("")
    lines.append(f"- generated_utc: `{exec_blockers.get('generated_utc', '')}`")
    lines.append(f"- status: `{exec_blockers.get('status', '')}`")
    lines.append(f"- source: `{exec_blockers.get('source', '')}`")
    lines.append(f"- cycles_total: `{exec_blockers.get('cycles_total', 0)}`")
    lines.append(f"- cycles_blocked: `{exec_blockers.get('cycles_blocked', 0)}`")
    lines.append(f"- blocked_ratio: `{exec_blockers.get('blocked_ratio', 0.0)}`")
    lines.append("")
    lines.append("## Top3")
    top3 = exec_blockers.get("top3") if isinstance(exec_blockers.get("top3"), list) else []
    if top3:
        for item in top3:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('reason', 'unknown')}: count={int(_num_or_none(item.get('count')) or 0)} "
                f"ratio={float(_num_or_none(item.get('ratio')) or 0.0):.4f}"
            )
    else:
        lines.append("- (empty)")
    lines.append("")
    ls = exec_blockers.get("longest_streak") if isinstance(exec_blockers.get("longest_streak"), dict) else {}
    lines.append(
        f"- longest_streak: reason=`{ls.get('reason', 'unknown')}` len=`{int(_num_or_none(ls.get('len')) or 0)}`"
    )
    warns = exec_blockers.get("warnings") if isinstance(exec_blockers.get("warnings"), list) else []
    if warns:
        lines.append("")
        lines.append("## Warnings")
        for w in warns:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def _render_no_trade_md(no_trade: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# No Trade Summary")
    lines.append("")
    lines.append(f"- generated_utc: `{no_trade.get('generated_utc', '')}`")
    lines.append(f"- is_no_trade_day: `{bool(no_trade.get('is_no_trade_day', False))}`")
    lines.append(f"- trades_total: `{int(_num_or_none(no_trade.get('trades_total')) or 0)}`")
    lines.append(f"- primary_reason: `{no_trade.get('primary_reason', 'unknown')}`")
    ev = no_trade.get("evidence") if isinstance(no_trade.get("evidence"), dict) else {}
    tb = ev.get("top_blocker") if isinstance(ev.get("top_blocker"), dict) else {}
    lines.append(
        f"- top_blocker: reason=`{tb.get('reason', '')}` ratio=`{_num_or_none(tb.get('ratio'))}`"
    )
    lines.append(f"- exec_blockers_source: `{ev.get('exec_blockers_source', '')}`")
    lines.append(f"- notes: {ev.get('notes', '')}")
    warns = no_trade.get("warnings") if isinstance(no_trade.get("warnings"), list) else []
    if warns:
        lines.append("")
        lines.append("## Warnings")
        for w in warns:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def write_exec_blockers_outputs(
    *,
    pack_dir: Path,
    exec_blockers: Dict[str, Any],
    no_trade: Dict[str, Any],
) -> Dict[str, str]:
    eb_dir = (pack_dir / "execution_blockers").resolve()
    nt_dir = (pack_dir / "no_trade").resolve()
    eb_dir.mkdir(parents=True, exist_ok=True)
    nt_dir.mkdir(parents=True, exist_ok=True)

    exec_json_path = (eb_dir / "exec_blockers.json").resolve()
    exec_md_path = (eb_dir / "exec_blockers.md").resolve()
    exec_csv_path = (eb_dir / "exec_blockers_cycles.csv").resolve()
    no_trade_json_path = (nt_dir / "no_trade.json").resolve()
    no_trade_md_path = (nt_dir / "no_trade.md").resolve()

    write_json_atomic(exec_json_path, exec_blockers)
    _write_text_atomic(exec_md_path, _render_exec_blockers_md(exec_blockers))

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(eb_dir)) as tf:
        writer = csv.DictWriter(tf, fieldnames=["reason", "count", "ratio"])
        writer.writeheader()
        total = max(1, int(_num_or_none(exec_blockers.get("cycles_total")) or 0))
        counts = exec_blockers.get("counts") if isinstance(exec_blockers.get("counts"), dict) else {}
        for reason in BLOCKER_TAXONOMY:
            count = int(_num_or_none((counts or {}).get(reason)) or 0)
            writer.writerow({"reason": reason, "count": count, "ratio": float(count / total)})
        tmp_csv = Path(tf.name)
    os.replace(tmp_csv, exec_csv_path)

    write_json_atomic(no_trade_json_path, no_trade)
    _write_text_atomic(no_trade_md_path, _render_no_trade_md(no_trade))

    return {
        "exec_blockers_json": str(exec_json_path),
        "exec_blockers_md": str(exec_md_path),
        "exec_blockers_cycles_csv": str(exec_csv_path),
        "no_trade_json": str(no_trade_json_path),
        "no_trade_md": str(no_trade_md_path),
    }


def compute_exec_blockers_and_no_trade(
    *,
    daily_base: Path,
    date_str: str,
    strict: bool = False,
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    daily_base = daily_base.resolve()
    date_norm = parse_date_str(date_str)
    if not date_norm:
        return 2, {"error": f"invalid date: {date_str}"}

    report_path = (daily_base / f"{date_norm}.json").resolve()
    report_obj = safe_read_json(report_path) or {}
    if not report_obj and strict:
        return 2, {"error": f"missing_or_invalid_daily_report:{report_path}"}
    if not report_obj:
        report_obj = {"date": date_norm, "schema_version": 1}

    pack_dir = (daily_base / "quant_packs" / date_norm).resolve()
    dataset_dir = resolve_dataset_dir(report_obj, daily_base=daily_base, date_str=date_norm)
    cycles_path = (dataset_dir / "cycles.csv").resolve()
    trades_path = (dataset_dir / "trades.csv").resolve()
    metrics_path = (pack_dir / "metrics" / "metrics.json").resolve()

    warnings: List[str] = []
    cycles_rows = read_csv_rows(cycles_path) if cycles_path.exists() else []
    trades_rows = read_csv_rows(trades_path) if trades_path.exists() else []
    metrics_obj = safe_read_json(metrics_path) or {}
    fallback_top3 = _extract_fallback_top3(report_obj, metrics_obj)

    if not cycles_rows:
        warnings.append(f"missing_cycles_csv:{cycles_path}")
    if not trades_path.exists():
        warnings.append(f"missing_trades_csv:{trades_path}")
    if not metrics_obj:
        warnings.append(f"missing_metrics_json:{metrics_path}")

    exec_blockers = compute_exec_blockers_payload(
        date_str=date_norm,
        cycles_rows=cycles_rows,
        fallback_top3=fallback_top3,
        warnings=list(warnings),
    )
    trades_total = _determine_trades_total(report_obj, metrics_obj, trades_rows)
    no_trade = infer_no_trade_payload(
        date_str=date_norm,
        trades_total=trades_total,
        exec_blockers=exec_blockers,
    )
    out_paths = write_exec_blockers_outputs(
        pack_dir=pack_dir,
        exec_blockers=exec_blockers,
        no_trade=no_trade,
    )

    manifest = {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "date": date_norm,
        "daily_base": str(daily_base),
        "daily_report_path": str(report_path),
        "pack_dir": str(pack_dir),
        "dataset_dir": str(dataset_dir),
        "inputs": {
            "cycles_csv": str(cycles_path),
            "trades_csv": str(trades_path),
            "metrics_json": str(metrics_path),
        },
        "exec_blockers_status": str(exec_blockers.get("status", "MISSING")),
        "no_trade_status": "OK",
        "outputs": out_paths,
        "warnings": list(dict.fromkeys(warnings + list(no_trade.get("warnings", [])))),
    }
    write_json_atomic((pack_dir / "exec_blockers_manifest.json").resolve(), manifest)

    rc = 0
    if str(exec_blockers.get("status", "")).upper() == "MISSING":
        rc = 2 if strict else 1
    elif str(exec_blockers.get("status", "")).upper() == "WARN":
        rc = 1

    if verbose:
        print(
            f"[A19X] date={date_norm} status={exec_blockers.get('status')} "
            f"source={exec_blockers.get('source')} cycles={exec_blockers.get('cycles_total')}"
        )
        print(f"[A19X] out={out_paths.get('exec_blockers_json')}")

    return rc, manifest


def _load_or_placeholder(path: Path, *, kind: str) -> Tuple[Dict[str, Any], List[str]]:
    obj = safe_read_json(path) or {}
    warnings: List[str] = []
    if isinstance(obj, dict) and obj:
        return obj, warnings
    warnings.append(f"missing_{kind}:{path}")
    if kind == "exec_blockers":
        return {
            "schema_version": 1,
            "generated_utc": now_utc_iso(),
            "status": "MISSING",
            "source": "missing",
            "cycles_total": 0,
            "cycles_blocked": 0,
            "blocked_ratio": 0.0,
            "top3": [],
            "counts": {k: 0 for k in BLOCKER_TAXONOMY},
            "longest_streak": {"reason": "unknown", "len": 0},
            "warnings": [warnings[0]],
        }, warnings
    return {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "status": "MISSING",
        "is_no_trade_day": False,
        "trades_total": 0,
        "primary_reason": "unknown",
        "evidence": {"top_blocker": {"reason": "", "ratio": None}, "exec_blockers_source": "missing", "notes": ""},
        "warnings": [warnings[0]],
    }, warnings


def _summarize_exec_blockers_for_daily(obj: Dict[str, Any]) -> Dict[str, Any]:
    top3 = obj.get("top3") if isinstance(obj.get("top3"), list) else []
    return {
        "schema_version": 1,
        "generated_utc": str(obj.get("generated_utc") or now_utc_iso()),
        "status": str(obj.get("status", "MISSING") or "MISSING"),
        "source": str(obj.get("source", "missing") or "missing"),
        "cycles_total": int(_num_or_none(obj.get("cycles_total")) or 0),
        "cycles_blocked": int(_num_or_none(obj.get("cycles_blocked")) or 0),
        "blocked_ratio": _num_or_none(obj.get("blocked_ratio")) or 0.0,
        "top3": [x for x in top3[:3] if isinstance(x, dict)],
        "longest_streak": obj.get("longest_streak") if isinstance(obj.get("longest_streak"), dict) else {},
        "warnings": list(obj.get("warnings")) if isinstance(obj.get("warnings"), list) else [],
    }


def attach_exec_blockers_to_daily(
    *,
    daily_base: Path,
    date_str: str,
    strict: bool = False,
    auto_compute: bool = True,
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    daily_base = daily_base.resolve()
    date_norm = parse_date_str(date_str)
    if not date_norm:
        return 2, {"error": f"invalid date: {date_str}"}

    report_path = (daily_base / f"{date_norm}.json").resolve()
    report_obj = safe_read_json(report_path) or {}
    if not report_obj and strict:
        return 2, {"error": f"missing_or_invalid_daily_report:{report_path}"}
    if not report_obj:
        report_obj = {"date": date_norm, "schema_version": 1}

    pack_dir = (daily_base / "quant_packs" / date_norm).resolve()
    exec_json_path = (pack_dir / "execution_blockers" / "exec_blockers.json").resolve()
    no_trade_json_path = (pack_dir / "no_trade" / "no_trade.json").resolve()
    warnings: List[str] = []
    compute_rc = 0

    if auto_compute and (not exec_json_path.exists() or not no_trade_json_path.exists()):
        compute_rc, _ = compute_exec_blockers_and_no_trade(
            daily_base=daily_base,
            date_str=date_norm,
            strict=False,
            verbose=verbose,
        )
        if compute_rc != 0:
            warnings.append(f"compute_rc={compute_rc}")

    exec_obj, w1 = _load_or_placeholder(exec_json_path, kind="exec_blockers")
    no_trade_obj, w2 = _load_or_placeholder(no_trade_json_path, kind="no_trade")
    warnings.extend(w1)
    warnings.extend(w2)

    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    qp["execution_blockers"] = _summarize_exec_blockers_for_daily(exec_obj)
    qp["no_trade"] = no_trade_obj
    report_obj["quant_pack"] = qp
    report_obj["updated_at_utc"] = now_utc_iso()

    bak = backup_file(report_path) if report_path.exists() else None
    write_json_atomic(report_path, report_obj)

    embed_manifest_path = (pack_dir / "embed_manifest.json").resolve()
    embed_manifest = safe_read_json(embed_manifest_path) or {}
    embed_manifest["exec_blockers_attach"] = {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "daily_report_in": str(report_path),
        "daily_report_out": str(report_path),
        "daily_report_backup": str(bak) if bak else "",
        "is_json": True,
        "mode": "replace",
        "fields_written": ["quant_pack.execution_blockers", "quant_pack.no_trade"],
        "exec_blockers_path": str(exec_json_path),
        "no_trade_path": str(no_trade_json_path),
        "warnings": list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
    }
    write_json_atomic(embed_manifest_path, embed_manifest)

    status = str(exec_obj.get("status", "MISSING") or "MISSING").upper()
    rc = 0
    if status == "MISSING":
        rc = 2 if strict else 1
    elif status == "WARN":
        rc = 1
    elif warnings:
        rc = 1

    info = {
        "schema_version": 1,
        "generated_utc": now_utc_iso(),
        "date": date_norm,
        "daily_report_path": str(report_path),
        "daily_report_backup": str(bak) if bak else "",
        "pack_dir": str(pack_dir),
        "exec_blockers_status": status,
        "warnings": list(dict.fromkeys(warnings)),
        "embed_manifest_path": str(embed_manifest_path),
        "compute_rc": int(compute_rc),
    }
    write_json_atomic((pack_dir / "exec_blockers_attach_manifest.json").resolve(), info)

    if verbose:
        print(f"[A20X] date={date_norm} status={status} report={report_path}")

    return rc, info

