"""Shared helpers for risk-profile single-source-of-truth state."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from atomic_io import atomic_write_json as io_atomic_write_json, safe_read_json as io_safe_read_json

RISK_PROFILE_CHOICES = ("low", "mid", "high", "ultra")
RISK_PROFILE_DEFAULT = "mid"
RISK_PROFILE_STATE_SCHEMA_VERSION = 1


def normalize_risk_profile(value: Any, *, default: str = RISK_PROFILE_DEFAULT) -> str:
    """Normalize risk profile into allowed choices."""
    raw = str(value or "").strip().lower()
    if raw in RISK_PROFILE_CHOICES:
        return raw
    return str(default or RISK_PROFILE_DEFAULT).strip().lower()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_risk_profile_state_path(reporting_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Resolve absolute risk profile state file path."""
    cfg = reporting_cfg if isinstance(reporting_cfg, dict) else {}
    configured = str(cfg.get("risk_profile_state_path", "") or "").strip()
    if configured:
        return os.path.abspath(configured)
    base_out_dir = str(cfg.get("base_out_dir", "outputs") or "outputs").strip() or "outputs"
    return os.path.abspath(os.path.join(base_out_dir, "state", "risk_profile_state.json"))


def resolve_risk_profile_events_path(
    reporting_cfg: Optional[Dict[str, Any]] = None,
    *,
    state_path: str = "",
) -> str:
    """Resolve absolute risk profile events jsonl path."""
    if str(state_path or "").strip():
        base_dir = os.path.dirname(os.path.abspath(str(state_path)))
    else:
        base_dir = os.path.dirname(resolve_risk_profile_state_path(reporting_cfg))
    return os.path.abspath(os.path.join(base_dir, "risk_profile_events.jsonl"))


def _normalize_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    requested = normalize_risk_profile(payload.get("requested"), default=RISK_PROFILE_DEFAULT)
    set_at = str(payload.get("set_at", "") or "").strip() or _now_utc_iso()
    set_by = str(payload.get("set_by", "") or "").strip() or "unknown"
    version = str(payload.get("version", "") or "").strip() or uuid.uuid4().hex[:16]
    out = dict(payload)
    out["schema_version"] = int(payload.get("schema_version", RISK_PROFILE_STATE_SCHEMA_VERSION) or RISK_PROFILE_STATE_SCHEMA_VERSION)
    out["requested"] = requested
    out["set_at"] = set_at
    out["set_by"] = set_by
    out["version"] = version
    return out


def read_risk_profile_state(path: str) -> Optional[Dict[str, Any]]:
    """Read and normalize risk profile state file."""
    obj = io_safe_read_json(str(path), retries=2, sleep_ms=15)
    if not isinstance(obj, dict):
        return None
    return _normalize_state_payload(obj)


@contextmanager
def _exclusive_file_lock(lock_path: str):
    """Best-effort cross-platform lock for append operations."""
    lock_abs = os.path.abspath(str(lock_path))
    os.makedirs(os.path.dirname(lock_abs) or ".", exist_ok=True)
    with open(lock_abs, "a+b") as lock_f:
        if os.name == "nt":
            import msvcrt  # type: ignore

            lock_f.seek(0, os.SEEK_END)
            if lock_f.tell() <= 0:
                lock_f.write(b"0")
                lock_f.flush()
                os.fsync(lock_f.fileno())
            lock_f.seek(0)
            msvcrt.locking(lock_f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_f.seek(0)
                msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def append_risk_profile_event(events_path: str, event: Dict[str, Any]) -> None:
    """Append one audit event line atomically with lock."""
    target = os.path.abspath(str(events_path))
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    payload = dict(event or {})
    line = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
    lock_path = f"{target}.lock"
    with _exclusive_file_lock(lock_path):
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def read_last_risk_profile_event(events_path: str) -> Optional[Dict[str, Any]]:
    """Read last valid JSON line from risk profile events file."""
    target = os.path.abspath(str(events_path))
    if not os.path.exists(target):
        return None
    last: Optional[Dict[str, Any]] = None
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    last = obj
    except Exception:
        return None
    return last


def write_risk_profile_state(
    path: str,
    *,
    requested: Any,
    set_by: str = "ui",
    version: str = "",
    set_at: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write risk profile state atomically and return normalized payload."""
    payload: Dict[str, Any] = dict(extra or {})
    payload.update(
        {
            "schema_version": int(RISK_PROFILE_STATE_SCHEMA_VERSION),
            "requested": normalize_risk_profile(requested, default=RISK_PROFILE_DEFAULT),
            "set_at": str(set_at or "").strip() or _now_utc_iso(),
            "set_by": str(set_by or "unknown").strip() or "unknown",
            "version": str(version or "").strip() or uuid.uuid4().hex[:16],
        }
    )
    normalized = _normalize_state_payload(payload)
    io_atomic_write_json(str(path), normalized, indent=2)
    return normalized


def ensure_risk_profile_state(
    path: str,
    *,
    default_requested: Any = RISK_PROFILE_DEFAULT,
    set_by: str = "system_default",
) -> Dict[str, Any]:
    """Ensure state file exists and returns valid normalized payload."""
    state_obj = read_risk_profile_state(path)
    if isinstance(state_obj, dict):
        # Patch missing required fields if needed.
        patched = _normalize_state_payload(state_obj)
        if patched != state_obj:
            io_atomic_write_json(str(path), patched, indent=2)
            return patched
        return state_obj
    return write_risk_profile_state(
        str(path),
        requested=default_requested,
        set_by=str(set_by or "system_default"),
    )


def request_risk_profile_change(
    state_path: str,
    *,
    requested: Any,
    source: str = "ui",
    actor: str = "",
    run_id: str = "",
    cycle_id: Any = None,
    ts: str = "",
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Update requested risk profile with append-only audit event.
    Strategy: only write event when old != new; same-value request is a no-op.
    """
    state_abs = os.path.abspath(str(state_path))
    old_state = ensure_risk_profile_state(state_abs, default_requested=RISK_PROFILE_DEFAULT, set_by="system_default")
    old_requested = normalize_risk_profile(old_state.get("requested"), default=RISK_PROFILE_DEFAULT)
    new_requested = normalize_risk_profile(requested, default=old_requested or RISK_PROFILE_DEFAULT)
    events_path = resolve_risk_profile_events_path(state_path=state_abs)
    source_norm = str(source or "unknown").strip() or "unknown"
    ts_iso = str(ts or "").strip() or _now_utc_iso()

    if old_requested == new_requested:
        return {
            "changed": False,
            "state": dict(old_state),
            "event": None,
            "events_path": events_path,
        }

    new_version = uuid.uuid4().hex[:16]
    event = {
        "ts": ts_iso,
        "old": old_requested,
        "new": new_requested,
        "source": source_norm,
        "actor": str(actor or "").strip(),
        "run_id": str(run_id or "").strip(),
        "cycle_id": cycle_id,
        "state_version": new_version,
    }
    state_extra = dict(extra_state or {})
    state_extra.update(
        {
            "last_change_ts": ts_iso,
            "last_change_old": old_requested,
            "last_change_new": new_requested,
            "last_change_source": source_norm,
            "request_id": state_extra.get("request_id", ""),
            "actor": str(actor or "").strip(),
            "run_id": str(run_id or "").strip(),
            "cycle_id": cycle_id,
        }
    )
    new_state = write_risk_profile_state(
        state_abs,
        requested=new_requested,
        set_by=source_norm,
        version=new_version,
        set_at=ts_iso,
        extra=state_extra,
    )
    append_risk_profile_event(events_path, event)
    return {
        "changed": True,
        "state": dict(new_state),
        "event": event,
        "events_path": events_path,
    }


class RiskProfileStateManager:
    """Small file-backed manager with mtime/version change detection."""

    def __init__(self, state_path: str, *, default_requested: Any = RISK_PROFILE_DEFAULT):
        self.state_path = os.path.abspath(str(state_path))
        self.events_path = resolve_risk_profile_events_path(state_path=self.state_path)
        self.default_requested = normalize_risk_profile(default_requested, default=RISK_PROFILE_DEFAULT)
        self.state: Dict[str, Any] = {}
        self.last_mtime: Optional[float] = None
        self.last_version: str = ""
        self.last_reload_diag: Dict[str, Any] = {}

    def _get_mtime(self) -> Optional[float]:
        try:
            return float(os.path.getmtime(self.state_path))
        except Exception:
            return None

    def load(self, *, ensure: bool = True) -> Dict[str, Any]:
        if ensure:
            state_obj = ensure_risk_profile_state(
                self.state_path,
                default_requested=self.default_requested,
                set_by="system_default",
            )
        else:
            state_obj = read_risk_profile_state(self.state_path) or {}
            if not state_obj:
                state_obj = {
                    "schema_version": int(RISK_PROFILE_STATE_SCHEMA_VERSION),
                    "requested": self.default_requested,
                    "set_at": _now_utc_iso(),
                    "set_by": "missing",
                    "version": "",
                }
        self.state = dict(state_obj)
        self.last_mtime = self._get_mtime()
        self.last_version = str(self.state.get("version", "") or "").strip()
        self.last_reload_diag = {
            "path": self.state_path,
            "exists": bool(os.path.exists(self.state_path)),
            "prev_mtime": None,
            "curr_mtime": self.last_mtime,
            "prev_version": "",
            "curr_version": self.last_version,
            "mtime_changed": True,
            "version_changed": bool(self.last_version),
            "state_changed": bool(self.state),
            "changed": True,
            "reason": "load",
            "force": False,
            "parse_failed": False,
        }
        return dict(self.state)

    def reload_if_changed(self, *, force: bool = False) -> bool:
        prev_state = dict(self.state or {})
        prev_mtime = self.last_mtime
        prev_version = str(self.last_version or "").strip()
        current_mtime = self._get_mtime()
        parse_failed = False
        new_state = read_risk_profile_state(self.state_path)
        if not isinstance(new_state, dict):
            parse_failed = bool(os.path.exists(self.state_path))
            new_state = ensure_risk_profile_state(
                self.state_path,
                default_requested=self.default_requested,
                set_by="system_default",
            )
        new_version = str(new_state.get("version", "") or "").strip()
        mtime_changed = False
        if current_mtime is None and prev_mtime is None:
            mtime_changed = False
        elif current_mtime is None or prev_mtime is None:
            mtime_changed = True
        else:
            mtime_changed = bool(abs(float(current_mtime) - float(prev_mtime)) >= 1e-9)
        version_changed = bool(new_version != prev_version)
        state_changed = bool(new_state != prev_state)
        changed = bool(force or mtime_changed or version_changed or state_changed)
        reason = "no_change"
        if force:
            reason = "force"
        elif version_changed:
            reason = "version_changed"
        elif state_changed:
            reason = "state_changed"
        elif mtime_changed:
            reason = "mtime_changed"
        self.state = dict(new_state)
        self.last_mtime = current_mtime if current_mtime is not None else self._get_mtime()
        self.last_version = new_version
        self.last_reload_diag = {
            "path": self.state_path,
            "exists": bool(os.path.exists(self.state_path)),
            "prev_mtime": prev_mtime,
            "curr_mtime": self.last_mtime,
            "prev_version": prev_version,
            "curr_version": self.last_version,
            "mtime_changed": mtime_changed,
            "version_changed": version_changed,
            "state_changed": state_changed,
            "changed": bool(changed),
            "reason": reason,
            "force": bool(force),
            "parse_failed": bool(parse_failed),
        }
        return bool(changed)

    def update_requested(
        self,
        requested: Any,
        *,
        set_by: str = "ui",
        actor: str = "",
        run_id: str = "",
        cycle_id: Any = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = request_risk_profile_change(
            self.state_path,
            requested=normalize_risk_profile(requested, default=self.default_requested),
            source=str(set_by or "ui"),
            actor=actor,
            run_id=run_id,
            cycle_id=cycle_id,
            extra_state=extra_state,
        )
        new_state = result.get("state", {}) if isinstance(result, dict) else {}
        self.state = dict(new_state if isinstance(new_state, dict) else {})
        self.last_mtime = self._get_mtime()
        self.last_version = str(self.state.get("version", "") or "").strip()
        return dict(self.state)

    def get_requested(self) -> str:
        if not isinstance(self.state, dict) or not self.state:
            self.load(ensure=True)
        return normalize_risk_profile((self.state or {}).get("requested"), default=self.default_requested)
