"""Structured telemetry logging helpers (JSONL event + metric streams)."""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_LOCK_GUARD = threading.Lock()
_PATH_LOCKS: Dict[str, threading.Lock] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _path_lock(path: str) -> threading.Lock:
    norm = os.path.abspath(path)
    with _LOCK_GUARD:
        lock = _PATH_LOCKS.get(norm)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[norm] = lock
        return lock


def _json_safe(value: Any) -> Any:
    """Convert objects to JSON-serializable forms."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _json_safe(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return repr(value)


def resolve_telemetry_out_dir(config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve telemetry output dir from config, defaulting to outputs/telemetry."""
    root = "outputs"
    if isinstance(config, dict):
        reporting_cfg = config.get("reporting", {})
        if isinstance(reporting_cfg, dict):
            configured = _coerce_str(reporting_cfg.get("out_dir"))
            if configured:
                root = configured
    return os.path.join(root, "telemetry")


class TelemetryLogger:
    """Thread-safe JSONL telemetry writer."""

    def __init__(self, out_dir: str, run_id: str, *, fsync: bool = False) -> None:
        resolved = _coerce_str(out_dir) or os.path.join("outputs", "telemetry")
        self.out_dir = os.path.abspath(resolved)
        self.run_id = _coerce_str(run_id) or "-"
        self.fsync = bool(fsync)
        os.makedirs(self.out_dir, exist_ok=True)
        self.events_path = os.path.join(self.out_dir, "events.jsonl")
        self.metrics_path = os.path.join(self.out_dir, "metrics.jsonl")

    def append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        lock = _path_lock(path)
        row = _json_safe(obj)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with lock:
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(line)
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

    def log_event(
        self,
        event: str,
        *,
        cycle_id: int,
        stream: str = "events",
        level: str = "INFO",
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error: Optional[Any] = None,
        module: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "level": _coerce_str(level).upper() or "INFO",
            "event": _coerce_str(event),
            "run_id": self.run_id,
            "cycle_id": int(cycle_id),
        }
        if module is not None:
            entry["module"] = _coerce_str(module)
        if message is not None:
            entry["message"] = _coerce_str(message)
        if payload is not None:
            entry["payload"] = _json_safe(payload)
        if duration_ms is not None:
            entry["duration_ms"] = float(duration_ms)
        if status is not None:
            entry["status"] = _coerce_str(status)
        if error is not None:
            if isinstance(error, BaseException):
                entry["error_type"] = error.__class__.__name__
                entry["error"] = _coerce_str(error)
                tb = traceback.format_exc()
                if tb and tb.strip() and tb.strip() != "NoneType: None":
                    entry["error_trace"] = tb
            else:
                entry["error_type"] = type(error).__name__
                entry["error"] = _coerce_str(error)
        stream_norm = _coerce_str(stream).lower()
        path = self.metrics_path if stream_norm == "metrics" else self.events_path
        self.append_jsonl(path, entry)

    def log_metric(
        self,
        name: str,
        *,
        cycle_id: int,
        value: float | int | str,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "cycle_id": int(cycle_id),
            "name": _coerce_str(name),
            "value": _json_safe(value),
        }
        if tags is not None:
            entry["tags"] = _json_safe(tags)
        self.append_jsonl(self.metrics_path, entry)
