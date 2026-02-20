#!/usr/bin/env python3
"""A2-1: enrich daily_reports_index.json with quant summary per date."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _backup_index(path: Path) -> Optional[Path]:
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


def _ensure_backup_after_write(path: Path, existing_backup: Optional[Path]) -> Optional[Path]:
    if existing_backup is not None:
        return existing_backup
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


def _parse_date(s: str) -> Optional[datetime]:
    try:
        d = datetime.strptime(str(s), "%Y-%m-%d")
        return d
    except Exception:
        return None


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


def _extract_replay_drift_summary(obj: Any) -> Optional[Dict[str, Any]]:
    rd = obj if isinstance(obj, dict) else {}
    if not rd:
        return None
    tag_top = rd.get("tag_top") if isinstance(rd.get("tag_top"), list) else []
    return {
        "status": str(rd.get("status", "") or "MISSING").upper(),
        "strict": bool(rd.get("strict", False)),
        "fail_on_drift": bool(rd.get("fail_on_drift", False)),
        "cycles": int(_num_or_none(rd.get("cycles")) or 0),
        "fails": int(_num_or_none(rd.get("fails")) or 0),
        "warns": int(_num_or_none(rd.get("warns")) or 0),
        "fail_cycle_ratio": _num_or_none(rd.get("fail_cycle_ratio")),
        "worst_cycle": rd.get("worst_cycle") if isinstance(rd.get("worst_cycle"), dict) else {},
        "tag_top": tag_top[:3],
        "generated_utc": str(rd.get("generated_utc") or _now_utc_iso()),
    }


def _extract_quant_from_report(report_obj: Dict[str, Any], report_path: Path) -> Optional[Dict[str, Any]]:
    qp = report_obj.get("quant_pack")
    if not isinstance(qp, dict):
        return None
    summary = qp.get("summary") if isinstance(qp.get("summary"), dict) else {}
    quant = {
        "total_return": _num_or_none(summary.get("total_return")),
        "sharpe": _num_or_none(summary.get("sharpe")),
        "max_drawdown": _num_or_none(summary.get("max_drawdown")),
        "trades_total": int(_num_or_none(summary.get("trades_total")) or 0),
        "gate_status": str(summary.get("gate_status", "NA") or "NA"),
        "pack_path": str(qp.get("pack_md_path") or ""),
        "updated_at_utc": str(qp.get("generated_at_utc") or _now_utc_iso()),
        "source": "report.quant_pack",
    }
    replay_drift = _extract_replay_drift_summary(qp.get("replay_drift"))
    if replay_drift is not None:
        quant["replay_drift"] = replay_drift
    bfr = qp.get("backtest_from_run") if isinstance(qp.get("backtest_from_run"), dict) else None
    if isinstance(bfr, dict):
        quant["backtest_from_run"] = {
            "status": str(bfr.get("status", "MISSING") or "MISSING"),
            "total_return": _num_or_none(bfr.get("total_return")),
            "max_drawdown": _num_or_none(bfr.get("max_drawdown")),
            "days": int(_num_or_none(bfr.get("days")) or 0),
            "generated_utc": str(bfr.get("generated_utc") or _now_utc_iso()),
        }
    bsw = qp.get("backtest_sweep") if isinstance(qp.get("backtest_sweep"), dict) else None
    if isinstance(bsw, dict):
        warn_list = bsw.get("warnings") if isinstance(bsw.get("warnings"), list) else []
        quant["backtest_sweep"] = {
            "status": str(bsw.get("status", "MISSING") or "MISSING"),
            "break_even_cost_bps": _num_or_none(bsw.get("break_even_cost_bps")),
            "sensitivity_per_1bp": _num_or_none(bsw.get("sensitivity_per_1bp")),
            "return_at_10bps": _num_or_none(bsw.get("return_at_10bps")),
            "warnings_count": int(len(warn_list)),
            "generated_utc": str(bsw.get("generated_utc") or _now_utc_iso()),
        }
    rec = qp.get("reconcile") if isinstance(qp.get("reconcile"), dict) else None
    if isinstance(rec, dict):
        gaps = rec.get("gaps") if isinstance(rec.get("gaps"), dict) else {}
        attr = rec.get("attribution") if isinstance(rec.get("attribution"), dict) else {}
        top = attr.get("likely_driver_top3") if isinstance(attr.get("likely_driver_top3"), list) else []
        warn_list = rec.get("warnings") if isinstance(rec.get("warnings"), list) else []
        ev = rec.get("evidence_summary") if isinstance(rec.get("evidence_summary"), dict) else {}
        if not ev:
            ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
        ev_top = ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else []
        ev_top1 = ev_top[0] if ev_top else {}
        quant["reconcile"] = {
            "status": str(rec.get("status", "MISSING") or "MISSING"),
            "return_gap_live_minus_backtest": _num_or_none(gaps.get("return_gap_live_minus_backtest")),
            "likely_driver_top1": str(top[0]) if top else "",
            "warnings_count": int(len(warn_list)),
            "gate_status": str(ev.get("gate_status", "") or "NA"),
            "replay_drift_status": str(ev.get("replay_drift_status", "") or "NA"),
            "gating_top1": {
                "reason": str(ev_top1.get("reason", "")) if isinstance(ev_top1, dict) else "",
                "count": int(_num_or_none(ev_top1.get("count")) or 0) if isinstance(ev_top1, dict) else 0,
            },
            "evidence_summary": {
                "gate_status": str(ev.get("gate_status", "") or "NA"),
                "replay_drift_status": str(ev.get("replay_drift_status", "") or "NA"),
                "gating_top3": ev.get("gating_top3") if isinstance(ev.get("gating_top3"), list) else [],
            },
            "generated_utc": str(rec.get("generated_utc") or _now_utc_iso()),
        }
    eb = qp.get("execution_blockers") if isinstance(qp.get("execution_blockers"), dict) else None
    nt = qp.get("no_trade") if isinstance(qp.get("no_trade"), dict) else None
    if isinstance(eb, dict):
        top3 = eb.get("top3") if isinstance(eb.get("top3"), list) else []
        top1 = top3[0] if top3 and isinstance(top3[0], dict) else {}
        quant["exec_blocker_top1_reason"] = str(top1.get("reason", "") or "")
        quant["exec_blocker_top1_ratio"] = _num_or_none(top1.get("ratio")) if isinstance(top1, dict) else None
        quant["exec_blocked_ratio"] = _num_or_none(eb.get("blocked_ratio"))
    if isinstance(nt, dict):
        quant["no_trade_primary_reason"] = str(nt.get("primary_reason", "") or "")
        quant["no_trade_flag"] = bool(_boolish(nt.get("is_no_trade_day")))
    if isinstance(eb, dict) or isinstance(nt, dict):
        eb_warn = eb.get("warnings") if isinstance(eb.get("warnings"), list) else []
        nt_warn = nt.get("warnings") if isinstance(nt.get("warnings"), list) else []
        quant["warnings_count"] = int(len(eb_warn) + len(nt_warn))
    return quant


def _extract_quant_from_pack_dir(daily_base: Path, date_str: str) -> Optional[Dict[str, Any]]:
    pack_dir = (daily_base / "quant_packs" / date_str).resolve()
    metrics = _read_json(pack_dir / "metrics" / "metrics.json") or {}
    gate = _read_json(pack_dir / "gate" / "gate_result.json") or {}
    pack_manifest = _read_json(pack_dir / "pack_manifest.json") or {}
    daily_quant_md = pack_dir / "daily_quant_report.md"

    if not metrics and not gate and not pack_manifest and not daily_quant_md.exists():
        # fallback legacy path
        pack_dir2 = (daily_base / "quant" / date_str).resolve()
        metrics = _read_json(pack_dir2 / "metrics" / "metrics.json") or metrics
        gate = _read_json(pack_dir2 / "gate" / "gate_result.json") or gate
        pack_manifest = _read_json(pack_dir2 / "pack_manifest.json") or pack_manifest
        if not daily_quant_md.exists():
            daily_quant_md = pack_dir2 / "daily_quant_report.md"
        if metrics or gate or pack_manifest or daily_quant_md.exists():
            pack_dir = pack_dir2

    if not metrics and not gate and not pack_manifest and not daily_quant_md.exists():
        return None

    perf = metrics.get("performance") if isinstance(metrics.get("performance"), dict) else {}
    risk = metrics.get("risk") if isinstance(metrics.get("risk"), dict) else {}
    trading = metrics.get("trading") if isinstance(metrics.get("trading"), dict) else {}
    gate_status = str(gate.get("status", "") or "NA").upper()
    if gate_status == "":
        gate_status = "NA"
    quant = {
        "total_return": _num_or_none(perf.get("total_return")),
        "sharpe": _num_or_none(risk.get("sharpe")),
        "max_drawdown": _num_or_none(risk.get("max_drawdown")),
        "trades_total": int(_num_or_none(trading.get("trades_total")) or 0),
        "gate_status": gate_status,
        "pack_path": str(daily_quant_md.resolve()) if daily_quant_md.exists() else "",
        "updated_at_utc": str(pack_manifest.get("generated_at_utc") or _now_utc_iso()),
        "source": "quant_packs.metrics",
    }
    drift_manifest = _read_json(pack_dir / "replay_drift" / "replay_drift_manifest.json") or {}
    drift_summary = _extract_replay_drift_summary(drift_manifest.get("summary"))
    if drift_summary is not None:
        quant["replay_drift"] = drift_summary
    sweep_attach = _read_json(pack_dir / "backtest_sweep" / "attach_manifest.json") or {}
    sweep_summary = sweep_attach.get("summary") if isinstance(sweep_attach.get("summary"), dict) else {}
    if isinstance(sweep_summary, dict) and sweep_summary:
        warn_list = sweep_summary.get("warnings") if isinstance(sweep_summary.get("warnings"), list) else []
        quant["backtest_sweep"] = {
            "status": str(sweep_summary.get("status", "MISSING") or "MISSING"),
            "break_even_cost_bps": _num_or_none(sweep_summary.get("break_even_cost_bps")),
            "sensitivity_per_1bp": _num_or_none(sweep_summary.get("sensitivity_per_1bp")),
            "return_at_10bps": _num_or_none(sweep_summary.get("return_at_10bps")),
            "warnings_count": int(len(warn_list)),
            "generated_utc": str(sweep_summary.get("generated_utc") or _now_utc_iso()),
        }
    eb = _read_json(pack_dir / "execution_blockers" / "exec_blockers.json") or {}
    nt = _read_json(pack_dir / "no_trade" / "no_trade.json") or {}
    if isinstance(eb, dict) and eb:
        top3 = eb.get("top3") if isinstance(eb.get("top3"), list) else []
        top1 = top3[0] if top3 and isinstance(top3[0], dict) else {}
        quant["exec_blocker_top1_reason"] = str(top1.get("reason", "") or "")
        quant["exec_blocker_top1_ratio"] = _num_or_none(top1.get("ratio")) if isinstance(top1, dict) else None
        quant["exec_blocked_ratio"] = _num_or_none(eb.get("blocked_ratio"))
    if isinstance(nt, dict) and nt:
        quant["no_trade_primary_reason"] = str(nt.get("primary_reason", "") or "")
        quant["no_trade_flag"] = bool(_boolish(nt.get("is_no_trade_day")))
    if (isinstance(eb, dict) and eb) or (isinstance(nt, dict) and nt):
        eb_warn = eb.get("warnings") if isinstance(eb.get("warnings"), list) else []
        nt_warn = nt.get("warnings") if isinstance(nt.get("warnings"), list) else []
        quant["warnings_count"] = int(len(eb_warn) + len(nt_warn))
    return quant


def _scan_daily_json_files(daily_base: Path, lookback_days: int) -> List[Path]:
    today = datetime.now(timezone.utc).date()
    min_day = today - timedelta(days=max(1, int(lookback_days)))
    out: List[Path] = []
    for p in daily_base.glob("*.json"):
        if not p.is_file():
            continue
        if p.name == "daily_reports_index.json":
            continue
        if not DATE_RE.match(p.name):
            continue
        date_str = p.stem
        d = _parse_date(date_str)
        if d is None:
            continue
        if d.date() < min_day:
            continue
        out.append(p.resolve())
    out.sort(key=lambda x: x.name, reverse=True)
    return out


def _normalize_entry(date_str: str, report_path: Path, report_obj: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entry = dict(existing or {})
    entry["date"] = date_str
    entry["path"] = str(report_path.resolve())
    if not entry.get("generated_at_local"):
        entry["generated_at_local"] = str(report_obj.get("generated_at_local") or report_obj.get("generated_at") or "")
    if isinstance(report_obj.get("summary"), dict):
        entry["summary"] = report_obj.get("summary")
    if isinstance(report_obj.get("risk_top"), list):
        entry["risk_top"] = report_obj.get("risk_top")
    if isinstance(report_obj.get("conviction_counts"), dict):
        entry["conviction_counts"] = report_obj.get("conviction_counts")
    return entry


def update_daily_reports_index(daily_base: Path, lookback_days: int, verbose: bool = False) -> Dict[str, Any]:
    daily_base = daily_base.resolve()
    index_path = daily_base / "daily_reports_index.json"
    existing_index = _read_json(index_path) or {}
    existing_reports = existing_index.get("reports") if isinstance(existing_index.get("reports"), list) else []
    by_date: Dict[str, Dict[str, Any]] = {}
    for item in existing_reports:
        if isinstance(item, dict) and str(item.get("date", "")).strip():
            by_date[str(item["date"])] = dict(item)

    daily_files = _scan_daily_json_files(daily_base, lookback_days=int(lookback_days))
    updated = 0
    missing_quant = 0
    for p in daily_files:
        date_str = p.stem
        report_obj = _read_json(p) or {}
        entry = _normalize_entry(date_str, p, report_obj, by_date.get(date_str))

        quant = _extract_quant_from_report(report_obj, p)
        if quant is None:
            quant = _extract_quant_from_pack_dir(daily_base, date_str)
        if quant is None:
            quant = {
                "total_return": None,
                "sharpe": None,
                "max_drawdown": None,
                "trades_total": None,
                "gate_status": "NA",
                "pack_path": "",
                "updated_at_utc": _now_utc_iso(),
                "source": "missing",
                "exec_blocker_top1_reason": "",
                "exec_blocker_top1_ratio": None,
                "exec_blocked_ratio": None,
                "no_trade_primary_reason": "",
                "no_trade_flag": False,
                "warnings_count": 0,
            }
            missing_quant += 1
        entry["quant"] = quant
        by_date[date_str] = entry
        updated += 1

    # Keep only dates present in scanned set for lookback window
    keep_dates = set([p.stem for p in daily_files])
    rows = [v for k, v in by_date.items() if k in keep_dates]
    rows.sort(key=lambda x: str(x.get("date", "")), reverse=True)

    out_obj = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "report_dir": str(daily_base),
        "reports": rows,
    }

    bak = _backup_index(index_path)
    _write_json_atomic(index_path, out_obj)
    bak = _ensure_backup_after_write(index_path, bak)
    result = {
        "index_path": str(index_path),
        "backup_path": str(bak) if bak else "",
        "updated_reports": int(updated),
        "missing_quant": int(missing_quant),
        "total_reports": len(rows),
    }
    if verbose:
        print(f"[INFO] daily_base={daily_base}")
        print(f"[INFO] index_path={index_path}")
        if bak:
            print(f"[INFO] backup={bak}")
        print(
            "[INFO] "
            f"updated_reports={updated} total_reports={len(rows)} missing_quant={missing_quant}"
        )
        print("[PASS] a7_update_daily_reports_index")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Update daily_reports_index.json with quant summary fields.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2
    update_daily_reports_index(daily_base, lookback_days=int(args.lookback_days), verbose=bool(args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
