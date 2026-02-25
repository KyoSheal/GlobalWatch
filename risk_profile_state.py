"""Shared helpers for risk-profile single-source-of-truth state."""

from __future__ import annotations

import os
import uuid
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


class RiskProfileStateManager:
    """Small file-backed manager with mtime/version change detection."""

    def __init__(self, state_path: str, *, default_requested: Any = RISK_PROFILE_DEFAULT):
        self.state_path = os.path.abspath(str(state_path))
        self.default_requested = normalize_risk_profile(default_requested, default=RISK_PROFILE_DEFAULT)
        self.state: Dict[str, Any] = {}
        self.last_mtime: Optional[float] = None
        self.last_version: str = ""

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
        return dict(self.state)

    def reload_if_changed(self, *, force: bool = False) -> bool:
        current_mtime = self._get_mtime()
        if not force and self.state and current_mtime is not None and self.last_mtime is not None:
            if abs(current_mtime - self.last_mtime) < 1e-9:
                return False
        new_state = read_risk_profile_state(self.state_path)
        if not isinstance(new_state, dict):
            new_state = ensure_risk_profile_state(
                self.state_path,
                default_requested=self.default_requested,
                set_by="system_default",
            )
        new_version = str(new_state.get("version", "") or "").strip()
        changed = force or (new_version != self.last_version) or (new_state != self.state)
        self.state = dict(new_state)
        self.last_mtime = current_mtime if current_mtime is not None else self._get_mtime()
        self.last_version = new_version
        return bool(changed)

    def update_requested(self, requested: Any, *, set_by: str = "ui") -> Dict[str, Any]:
        new_state = write_risk_profile_state(
            self.state_path,
            requested=normalize_risk_profile(requested, default=self.default_requested),
            set_by=str(set_by or "ui"),
        )
        self.state = dict(new_state)
        self.last_mtime = self._get_mtime()
        self.last_version = str(self.state.get("version", "") or "").strip()
        return dict(self.state)

    def get_requested(self) -> str:
        if not isinstance(self.state, dict) or not self.state:
            self.load(ensure=True)
        return normalize_risk_profile((self.state or {}).get("requested"), default=self.default_requested)

