"""Output directory governance helpers (run folders, latest pointer, registry)."""

from __future__ import annotations

import json
import os
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
    dt = _utc_now(now_utc)
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
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
