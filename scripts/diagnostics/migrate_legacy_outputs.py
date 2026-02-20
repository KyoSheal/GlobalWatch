#!/usr/bin/env python3
"""Import legacy output folders into outpost layout: outputs/YYYY-MM/<run_id>/."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, safe_read_json
from outpost import (
    append_registry,
    infer_run_kind,
    is_candidate_run_dir,
    make_run_id,
    normalize_run_kind,
    write_latest_pointer,
)

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


ALLOWED_PROFILES = {"low", "mid", "high", "ultra"}


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


def _normalize_profile(value: Any) -> str:
    p = str(value or "").strip().lower()
    return p if p in ALLOWED_PROFILES else "mid"


def _safe_jsonl_rows(path: Path, limit: int = 200000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return rows
    return rows


def _most_common_run_id(rows: List[Dict[str, Any]], key: str = "run_id") -> str:
    cnt = Counter()
    for r in rows:
        rid = str(r.get(key) or "").strip()
        if rid:
            cnt[rid] += 1
    if not cnt:
        return ""
    return cnt.most_common(1)[0][0]


def _latest_mtime_utc(dir_path: Path) -> datetime:
    latest_ts = 0.0
    try:
        for p in dir_path.rglob("*"):
            try:
                if p.is_file():
                    ts = float(p.stat().st_mtime)
                    if ts > latest_ts:
                        latest_ts = ts
            except Exception:
                continue
    except Exception:
        pass
    if latest_ts <= 0:
        latest_ts = datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(latest_ts, tz=timezone.utc)


def _month_from_ended(ended_utc: datetime) -> str:
    if ZoneInfo is not None:
        try:
            local = ended_utc.astimezone(ZoneInfo("America/Vancouver"))
            return local.strftime("%Y-%m")
        except Exception:
            pass
    return ended_utc.astimezone().strftime("%Y-%m")


def _is_month_dir_name(name: str) -> bool:
    text = str(name or "").strip()
    if len(text) != 7:
        return False
    if text[4] != "-":
        return False
    y = text[:4]
    m = text[5:]
    return y.isdigit() and m.isdigit() and 1 <= int(m) <= 12


def _is_already_new_structure(path: Path, legacy_root: Path) -> bool:
    """Check if path matches <legacy_root>/<YYYY-MM>/<run_id>/ with run_summary.json."""
    rel = None
    try:
        rel = path.resolve().relative_to(legacy_root.resolve())
    except Exception:
        return False
    parts = rel.parts
    if len(parts) != 2:
        return False
    if not _is_month_dir_name(parts[0]):
        return False
    return (path / "run_summary.json").exists()


def _is_legacy_candidate(path: Path, legacy_root: Path) -> bool:
    if not path.is_dir():
        return False
    if _is_month_dir_name(path.name):
        return False
    if _is_already_new_structure(path, legacy_root):
        return False
    return is_candidate_run_dir(path)


def _infer_run_id(candidate: Path, ended_utc: datetime) -> Tuple[str, Dict[str, Any]]:
    diag: Dict[str, Any] = {}
    snapshot = safe_read_json(str(candidate / "snapshot_live.json"), retries=2, sleep_ms=15)
    if isinstance(snapshot, dict):
        rid = str(snapshot.get("run_id") or "").strip()
        if rid:
            diag["run_id_source"] = "snapshot"
            return rid, diag
    events = _safe_jsonl_rows(candidate / "telemetry" / "events.jsonl", limit=200000)
    rid = _most_common_run_id(events, key="run_id")
    if rid:
        diag["run_id_source"] = "telemetry_events"
        return rid, diag
    trades = _safe_jsonl_rows(candidate / "trade_history.jsonl", limit=200000)
    rid = _most_common_run_id(trades, key="run_id")
    if rid:
        diag["run_id_source"] = "trade_history"
        return rid, diag
    rid = make_run_id(ended_utc)
    diag["run_id_source"] = "generated"
    return rid, diag


def _unique_target_dir(base_out_dir: Path, month_dir: str, run_id: str) -> Tuple[Path, str]:
    month_path = base_out_dir / month_dir
    target = month_path / run_id
    if not target.exists():
        return target, run_id
    suffix = f"-imp{uuid.uuid4().hex[:4]}"
    run_id2 = f"{run_id}{suffix}"
    target2 = month_path / run_id2
    i = 0
    while target2.exists():
        i += 1
        run_id2 = f"{run_id}{suffix}{i}"
        target2 = month_path / run_id2
    return target2, run_id2


def _copy_or_move(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        if src.is_dir():
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src), str(dst))


def _infer_started_ended(snapshot: Dict[str, Any], events: List[Dict[str, Any]], trades: List[Dict[str, Any]], fallback_ended: datetime) -> Tuple[Optional[str], str]:
    ended = _parse_dt(snapshot.get("timestamp")) if isinstance(snapshot, dict) else None
    if not isinstance(ended, datetime):
        event_ts = [_parse_dt(e.get("ts_utc")) for e in events]
        event_ts = [t for t in event_ts if isinstance(t, datetime)]
        if event_ts:
            ended = max(event_ts)
    if not isinstance(ended, datetime):
        trade_ts = [_parse_dt(t.get("timestamp")) for t in trades]
        trade_ts = [t for t in trade_ts if isinstance(t, datetime)]
        if trade_ts:
            ended = max(trade_ts)
    if not isinstance(ended, datetime):
        ended = fallback_ended

    started = _parse_dt(snapshot.get("run_started_at")) if isinstance(snapshot, dict) else None
    if not isinstance(started, datetime):
        event_ts = [_parse_dt(e.get("ts_utc")) for e in events]
        event_ts = [t for t in event_ts if isinstance(t, datetime)]
        if event_ts:
            started = min(event_ts)
    if not isinstance(started, datetime):
        trade_ts = [_parse_dt(t.get("timestamp")) for t in trades]
        trade_ts = [t for t in trade_ts if isinstance(t, datetime)]
        if trade_ts:
            started = min(trade_ts)

    return (started.isoformat() if isinstance(started, datetime) else None), ended.isoformat()


def _best_effort_run_summary(candidate: Path, target_dir: Path, run_id: str, ended_utc: datetime) -> Dict[str, Any]:
    snapshot = safe_read_json(str(candidate / "snapshot_live.json"), retries=2, sleep_ms=15)
    if not isinstance(snapshot, dict):
        snapshot = {}
    events = _safe_jsonl_rows(candidate / "telemetry" / "events.jsonl", limit=300000)
    trades = _safe_jsonl_rows(candidate / "trade_history.jsonl", limit=300000)

    started_at_utc, ended_at_utc = _infer_started_ended(snapshot, events, trades, ended_utc)
    session_id = str(snapshot.get("session_id") or "").strip() or None
    if not session_id:
        cnt = Counter([str(t.get("session_id") or "").strip() for t in trades if str(t.get("session_id") or "").strip()])
        session_id = cnt.most_common(1)[0][0] if cnt else None

    cycle = snapshot.get("cycle")
    cycle_id = snapshot.get("cycle_id")
    cycles = 0
    if cycle_id not in (None, ""):
        try:
            cycles = int(cycle_id)
        except Exception:
            cycles = 0
    elif cycle not in (None, ""):
        try:
            cycles = int(cycle)
        except Exception:
            cycles = 0
    if cycles <= 0 and trades:
        vals = []
        for t in trades:
            try:
                vals.append(int(t.get("cycle_id", t.get("cycle", 0)) or 0))
            except Exception:
                continue
        if vals:
            cycles = max(vals)

    final_equity = snapshot.get("total_equity")
    initial_cash = snapshot.get("initial_cash")
    if initial_cash in (None, ""):
        initial_cash = snapshot.get("initial_cash_usd")
    pnl = None
    total_return = snapshot.get("total_return")
    if final_equity not in (None, "") and initial_cash not in (None, ""):
        try:
            pnl = float(final_equity) - float(initial_cash)
        except Exception:
            pnl = None
    if total_return in (None, "") and pnl is not None and initial_cash not in (None, ""):
        try:
            total_return = float(pnl) / float(initial_cash) if float(initial_cash) != 0 else None
        except Exception:
            total_return = None

    risk_profile = _normalize_profile(snapshot.get("active_risk_profile") or snapshot.get("requested_risk_profile"))
    if risk_profile == "mid" and trades:
        cnt = Counter([_normalize_profile(t.get("risk_profile")) for t in trades])
        if cnt:
            risk_profile = cnt.most_common(1)[0][0]

    overrides_hash = snapshot.get("risk_profile_overrides_hash")
    template_version = snapshot.get("risk_profile_template_version")
    max_drawdown = snapshot.get("drawdown")

    telemetry_paths = None
    tele_dir = target_dir / "telemetry"
    if tele_dir.exists():
        telemetry_paths = {
            "out_dir": str(tele_dir.resolve()),
            "events": str((tele_dir / "events.jsonl").resolve()),
            "metrics": str((tele_dir / "metrics.jsonl").resolve()),
        }

    trade_history_path = str((target_dir / "trade_history.jsonl").resolve()) if (target_dir / "trade_history.jsonl").exists() else None
    daily_reports_dir = str((target_dir / "reports").resolve()) if (target_dir / "reports").exists() else None

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "session_id": session_id,
        "status": "IMPORTED",
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "cycles": int(cycles or 0),
        "final_equity": float(final_equity) if final_equity not in (None, "") else None,
        "pnl": float(pnl) if pnl is not None else None,
        "total_return": float(total_return) if total_return not in (None, "") else None,
        "max_drawdown": float(max_drawdown) if max_drawdown not in (None, "") else None,
        "risk_profile": risk_profile,
        "overrides_hash": str(overrides_hash or "") if overrides_hash not in (None, "") else None,
        "template_version": int(template_version) if template_version not in (None, "") else None,
        "telemetry_paths": telemetry_paths,
        "trade_history_path": trade_history_path,
        "daily_reports_dir": daily_reports_dir,
        "imported_from": str(candidate.resolve()),
        "notes": "legacy import; best-effort summary with possible missing fields",
    }
    summary["run_kind"] = infer_run_kind(candidate, run_summary=summary, registry_entry=None)
    return summary


def _copy_legacy_candidate(candidate: Path, target_dir: Path, mode: str, dry_run: bool) -> None:
    file_whitelist = [
        "snapshot_live.json",
        "runtime_control.json",
        "trade_history.jsonl",
        "paper_trades.jsonl",
        "paper_trades.csv",
        "portfolio_snapshots.jsonl",
        "run_summary.json",
        "paper_summary.txt",
        "paper_summary_live.txt",
        "scoreboard.jsonl",
        "equity_curve.png",
    ]
    dir_whitelist = ["telemetry", "reports", "trade_history", "paper_trades"]

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for name in file_whitelist:
        src = candidate / name
        if src.exists() and src.is_file():
            dst = target_dir / name
            _copy_or_move(src, dst, mode=mode, dry_run=dry_run)

    for name in dir_whitelist:
        src = candidate / name
        if src.exists() and src.is_dir():
            dst = target_dir / name
            _copy_or_move(src, dst, mode=mode, dry_run=dry_run)


def _load_build_month_summaries():
    try:
        from scripts.diagnostics.build_month_summaries import build_month_summaries
        return build_month_summaries
    except Exception:
        pass
    try:
        import importlib.util
        mod_path = ROOT / "scripts" / "diagnostics" / "build_month_summaries.py"
        spec = importlib.util.spec_from_file_location("build_month_summaries_mod", str(mod_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            fn = getattr(mod, "build_month_summaries", None)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


def migrate_legacy_outputs(
    legacy_root: str,
    base_out_dir: str,
    mode: str = "copy",
    dry_run: bool = False,
    update_latest: bool = False,
    build_month_summaries: bool = True,
) -> Dict[str, Any]:
    legacy = Path(str(legacy_root or "outputs")).resolve()
    base = Path(str(base_out_dir or "outputs")).resolve()
    mode_norm = str(mode or "copy").strip().lower()
    if mode_norm not in {"copy", "move"}:
        mode_norm = "copy"

    candidates = []
    for child in sorted(legacy.iterdir(), key=lambda p: p.name):
        if _is_legacy_candidate(child, legacy):
            candidates.append(child)

    imported: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    months_touched = set()

    for cand in candidates:
        ended_utc = _latest_mtime_utc(cand)
        run_id, run_diag = _infer_run_id(cand, ended_utc=ended_utc)
        month_dir = _month_from_ended(ended_utc)
        target_dir, final_run_id = _unique_target_dir(base, month_dir, run_id)

        if not dry_run:
            target_dir.parent.mkdir(parents=True, exist_ok=True)

        has_any = is_candidate_run_dir(cand)
        if not has_any:
            skipped.append(
                {
                    "legacy_path": str(cand),
                    "reason": "no_valid_markers",
                }
            )
            continue

        _copy_legacy_candidate(cand, target_dir, mode=mode_norm, dry_run=dry_run)

        summary_path = target_dir / "run_summary.json"
        summary_obj = None
        if summary_path.exists():
            summary_obj = safe_read_json(str(summary_path), retries=2, sleep_ms=15)
        if not isinstance(summary_obj, dict):
            summary_obj = _best_effort_run_summary(cand, target_dir, final_run_id, ended_utc)
            if not dry_run:
                atomic_write_json(str(summary_path), summary_obj, indent=2)
        run_kind = infer_run_kind(cand, run_summary=summary_obj, registry_entry=None)
        run_kind = normalize_run_kind(run_kind, default="live")
        if isinstance(summary_obj, dict) and normalize_run_kind(summary_obj.get("run_kind"), default="") == "":
            summary_obj["run_kind"] = run_kind
            if not dry_run:
                atomic_write_json(str(summary_path), summary_obj, indent=2)

        risk_profile = _normalize_profile((summary_obj or {}).get("risk_profile"))
        overrides_hash = (summary_obj or {}).get("overrides_hash")
        template_version = (summary_obj or {}).get("template_version")
        started_at_utc = (summary_obj or {}).get("started_at_utc")
        ended_at_utc = (summary_obj or {}).get("ended_at_utc") or ended_utc.isoformat()
        status = (summary_obj or {}).get("status") or "IMPORTED"

        record_base = {
            "schema_version": 1,
            "run_id": final_run_id,
            "out_dir": str(target_dir.resolve()),
            "month_dir": month_dir,
            "run_kind": run_kind,
            "risk_profile": risk_profile,
            "overrides_hash": str(overrides_hash or ""),
            "template_version": template_version,
            "status": status,
        }
        start_record = dict(record_base)
        start_record.update({"action": "start", "ts_utc": str(started_at_utc or ended_at_utc)})
        end_record = dict(record_base)
        end_record.update(
            {
                "action": "end",
                "ts_utc": str(ended_at_utc),
                "run_summary_path": str(summary_path.resolve()),
                "ended_at_utc": str(ended_at_utc),
            }
        )
        if not dry_run:
            append_registry(str(base), start_record)
            append_registry(str(base), end_record)
            if update_latest:
                write_latest_pointer(str(base), end_record)

        months_touched.add(month_dir)
        imported.append(
            {
                "legacy_path": str(cand.resolve()),
                "target_dir": str(target_dir.resolve()),
                "run_id": final_run_id,
                "month": month_dir,
                "run_id_source": run_diag.get("run_id_source"),
            }
        )

    if build_month_summaries and months_touched:
        fn = _load_build_month_summaries()
        if callable(fn):
            if not dry_run:
                try:
                    fn(str(base), sorted(months_touched))
                except Exception:
                    pass

    return {
        "legacy_root": str(legacy),
        "base_out_dir": str(base),
        "mode": mode_norm,
        "dry_run": bool(dry_run),
        "update_latest": bool(update_latest),
        "build_month_summaries": bool(build_month_summaries),
        "candidates_found": len(candidates),
        "imported_count": len(imported) if not dry_run else 0,
        "skipped_count": len(skipped),
        "months_touched": sorted(months_touched),
        "imports": imported,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy outputs into outpost run layout.")
    parser.add_argument("--legacy-root", type=str, default="outputs")
    parser.add_argument("--base-out-dir", type=str, default="outputs")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-latest", action="store_true")
    parser.add_argument("--build-month-summaries", dest="build_month_summaries", action="store_true", default=True)
    parser.add_argument("--no-build-month-summaries", dest="build_month_summaries", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = migrate_legacy_outputs(
        legacy_root=args.legacy_root,
        base_out_dir=args.base_out_dir,
        mode=args.mode,
        dry_run=bool(args.dry_run),
        update_latest=bool(args.update_latest),
        build_month_summaries=bool(args.build_month_summaries),
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("[MIGRATE LEGACY] summary")
    print(f"  candidates_found: {result['candidates_found']}")
    print(f"  imported_count: {result['imported_count']}")
    print(f"  skipped_count: {result['skipped_count']}")
    print(f"  months_touched: {', '.join(result['months_touched']) if result['months_touched'] else '-'}")
    print(f"  mode: {result['mode']} dry_run={result['dry_run']}")
    for item in result.get("imports", []):
        print(
            f"  - {item.get('legacy_path')} -> {item.get('target_dir')} "
            f"run_id={item.get('run_id')} month={item.get('month')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
