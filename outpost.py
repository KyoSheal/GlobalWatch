"""Output directory governance helpers (run folders, latest pointer, registry)."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from atomic_io import atomic_write_json, atomic_write_jsonl, safe_read_json

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


SCHEMA_VERSION = 1
_REGISTRY_LOCK = threading.Lock()
RUN_KIND_CHOICES = ("live", "dryrun", "diagnostics", "test")
_RUN_KIND_SET = set(RUN_KIND_CHOICES)
_DRYRUN_HINT_RE = re.compile(r"dryrun|debug[-_]?system[-_]?s1[-_]?s5|gw_dryrun", re.IGNORECASE)


def _utc_now(now_utc: Optional[datetime] = None) -> datetime:
    if isinstance(now_utc, datetime):
        if now_utc.tzinfo is None:
            return now_utc.replace(tzinfo=timezone.utc)
        return now_utc.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _to_local_dt(now_local: Optional[datetime] = None) -> datetime:
    if isinstance(now_local, datetime):
        if now_local.tzinfo is None:
            return now_local.astimezone()
        return now_local
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("America/Vancouver"))
        except Exception:
            pass
    return datetime.now().astimezone()


def _coerce_record(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in record.items():
        key = str(k)
        if isinstance(v, Path):
            out[key] = str(v)
        else:
            out[key] = v
    return out


def normalize_run_kind(value: Any, default: str = "live") -> str:
    kind = str(value or "").strip().lower()
    if kind in _RUN_KIND_SET:
        return kind
    return default


def resolve_base_out_dir(run_kind: Any, base_out_dir: str = "outputs") -> str:
    """
    Route non-live runs into outputs/test to avoid polluting live artifacts.

    live         -> <base_out_dir>
    dryrun/test/diagnostics -> <base_out_dir>/test
    """
    kind = normalize_run_kind(run_kind, default="live")
    root = Path(str(base_out_dir or "outputs")).resolve()
    if kind == "live":
        return str(root)
    return str((root / "test").resolve())


def is_candidate_run_dir(p: Path) -> bool:
    """Conservative candidate detector for formal run-like directories."""
    if not isinstance(p, Path) or not p.is_dir():
        return False
    checks = [
        p / "run_summary.json",
        p / "trade_history",
        p / "paper_trades",
        p / "trade_history.jsonl",
        p / "paper_trades.jsonl",
    ]
    return any(x.exists() for x in checks)


def _looks_dryrun_meta(*objs: Any) -> bool:
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        fields = [
            obj.get("run_kind"),
            obj.get("mode"),
            obj.get("argv"),
            obj.get("status"),
            obj.get("source"),
            obj.get("out_dir"),
            obj.get("run_id"),
            obj.get("session_id"),
        ]
        haystack = " ".join(str(v or "") for v in fields)
        if _DRYRUN_HINT_RE.search(haystack):
            return True
        if bool(obj.get("is_dryrun", False)):
            return True
    return False


def infer_run_kind(
    run_dir: Path,
    run_summary: Optional[Dict[str, Any]] = None,
    registry_entry: Optional[Dict[str, Any]] = None,
) -> str:
    """Best-effort run_kind inference for legacy rows/files."""
    summary = run_summary if isinstance(run_summary, dict) else {}
    entry = registry_entry if isinstance(registry_entry, dict) else {}

    # 1) explicit metadata wins
    explicit = normalize_run_kind(summary.get("run_kind"), default="")
    if explicit:
        return explicit
    explicit = normalize_run_kind(entry.get("run_kind"), default="")
    if explicit:
        return explicit

    # 2) dryrun hints
    if _looks_dryrun_meta(summary, entry):
        return "dryrun"
    if _DRYRUN_HINT_RE.search(str(run_dir or "")):
        return "dryrun"

    # 3) trade artifacts => live
    trade_markers = [
        run_dir / "trade_history",
        run_dir / "paper_trades",
        run_dir / "trade_history.jsonl",
        run_dir / "paper_trades.jsonl",
        run_dir / "paper_trades.csv",
    ]
    if any(m.exists() for m in trade_markers):
        return "live"

    # For old summaries: if summary resembles paper_trading run summary, treat as live.
    if isinstance(summary, dict):
        if any(k in summary for k in ("risk_profile", "template_version", "overrides_hash", "final_equity", "cycles")):
            return "live"

    # 4) telemetry-only folders are diagnostics
    tele_markers = [
        run_dir / "telemetry" / "events.jsonl",
        run_dir / "telemetry" / "metrics.jsonl",
    ]
    if any(m.exists() for m in tele_markers):
        return "diagnostics"

    # 5) fallback
    return "test"


def _load_registry_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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
        return []
    return rows


def make_run_id(now_utc: Optional[datetime] = None) -> str:
    # Minute-level sortable stamp + random suffix for same-minute uniqueness.
    dt = _to_local_dt(now_utc)
    stamp = dt.strftime("%Y%m%d-%H%M")
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}-{suffix}"


def resolve_out_dir(base_dir: str, now_local: Optional[datetime] = None, run_id: Optional[str] = None) -> Path:
    root = Path(str(base_dir or "outputs")).resolve()
    local_dt = _to_local_dt(now_local)
    month_dir = local_dt.strftime("%Y-%m")
    rid = str(run_id or "").strip() or make_run_id()
    out_dir = root / month_dir / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_latest_pointer(base_dir: str, record: Dict[str, Any]) -> None:
    root = Path(str(base_dir or "outputs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest_path = root / "LATEST.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _utc_now().isoformat(),
    }
    payload.update(_coerce_record(record))
    atomic_write_json(str(latest_path), payload, indent=2)


def append_registry(base_dir: str, record: Dict[str, Any]) -> None:
    root = Path(str(base_dir or "outputs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry.jsonl"
    row = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now().isoformat(),
    }
    row.update(_coerce_record(record))
    run_kind = normalize_run_kind(row.get("run_kind"), default="")
    if run_kind:
        row["run_kind"] = run_kind
    else:
        out_dir = str(row.get("out_dir") or "").strip()
        if out_dir:
            try:
                run_dir = Path(out_dir)
                summary = safe_read_json(str(run_dir / "run_summary.json"), retries=1, sleep_ms=5)
                row["run_kind"] = infer_run_kind(
                    run_dir,
                    run_summary=summary if isinstance(summary, dict) else None,
                    registry_entry=row,
                )
            except Exception:
                row["run_kind"] = "live"
        else:
            row["run_kind"] = "live"
    with _REGISTRY_LOCK:
        rows = _load_registry_rows(registry_path)
        action = str(row.get("action", "") or "").strip().lower()
        run_id = str(row.get("run_id", "") or "").strip()
        if action == "start" and run_id:
            for prev in rows[-20:]:
                if not isinstance(prev, dict):
                    continue
                prev_action = str(prev.get("action", "") or "").strip().lower()
                prev_run_id = str(prev.get("run_id", "") or "").strip()
                if prev_action == "start" and prev_run_id == run_id:
                    return
        rows.append(row)
        atomic_write_jsonl(str(registry_path), rows)


def load_latest_pointer(base_dir: str) -> Dict[str, Any] | None:
    root = Path(str(base_dir or "outputs")).resolve()
    latest_path = root / "LATEST.json"
    payload = safe_read_json(str(latest_path), retries=3, sleep_ms=30)
    return payload if isinstance(payload, dict) else None
