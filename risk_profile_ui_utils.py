"""UI helper utilities for risk profile display state."""

from __future__ import annotations

import os
from typing import Any, Dict
from atomic_io import safe_read_json as io_safe_read_json
from outpost import normalize_run_kind, resolve_base_out_dir
from risk_profile_state import (
    normalize_risk_profile,
    read_risk_profile_state,
    resolve_risk_profile_state_path,
)

_CHOICES = {"low", "mid", "high", "ultra"}


def _norm_profile(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in _CHOICES else "mid"


def _raw_profile(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in _CHOICES else ""


def _resolve_runtime_control_path(reporting_cfg: Dict[str, Any]) -> str:
    runtime_control_path_cfg = str((reporting_cfg or {}).get("runtime_control_path", "") or "").strip()
    if runtime_control_path_cfg:
        return os.path.abspath(runtime_control_path_cfg)
    reporting_out_dir = str((reporting_cfg or {}).get("out_dir", "") or "").strip()
    if reporting_out_dir:
        return os.path.abspath(os.path.join(reporting_out_dir, "runtime_control.json"))
    return os.path.abspath(os.path.join("outputs", "runtime_control.json"))


def _infer_run_kind(cfg: Dict[str, Any], reporting_cfg: Dict[str, Any]) -> str:
    explicit = normalize_run_kind((reporting_cfg or {}).get("run_kind"), default="")
    if explicit:
        return explicit

    mode_hint = str((cfg or {}).get("run_mode", "") or "").strip().lower()
    if mode_hint in {"dryrun", "debug", "test", "diagnostics"}:
        if mode_hint == "test":
            return "test"
        if mode_hint == "diagnostics":
            return "diagnostics"
        return "dryrun"

    env_hint = str((reporting_cfg or {}).get("env", "") or "").strip().lower()
    if env_hint in {"dryrun", "debug", "test"}:
        return "dryrun"

    out_hint = str((reporting_cfg or {}).get("out_dir", "") or "").strip().lower()
    if "dryrun" in out_hint:
        return "dryrun"
    return "live"


def resolve_effective_reporting_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror engine run-kind routing to keep UI/engine path resolution aligned."""
    cfg_obj = cfg if isinstance(cfg, dict) else {}
    reporting_cfg = cfg_obj.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        reporting_cfg = {}
    out: Dict[str, Any] = dict(reporting_cfg)
    run_kind = _infer_run_kind(cfg_obj, reporting_cfg)
    base_raw = str(out.get("base_out_dir", "outputs") or "outputs").strip() or "outputs"
    out["base_out_dir"] = str(resolve_base_out_dir(run_kind, base_raw))
    out["_resolved_run_kind"] = str(run_kind)
    return out


def get_active_risk_profile(
    *,
    config_path: str = "paper_config.json",
    runtime_control_path: str = "",
    snapshot_path: str = "",
) -> str:
    """
    Source-of-truth profile for UI default selection.
    Priority:
      1) runtime control requested_risk_profile
      2) snapshot active_risk_profile
      3) config execution.risk_profile
      4) fallback mid
    """
    cfg = io_safe_read_json(config_path, retries=2, sleep_ms=15) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    reporting_cfg = resolve_effective_reporting_cfg(cfg)

    state_path = resolve_risk_profile_state_path(reporting_cfg)
    state_obj = read_risk_profile_state(state_path)
    if isinstance(state_obj, dict):
        requested_from_state = _raw_profile(state_obj.get("requested"))
        if requested_from_state:
            return requested_from_state

    runtime_path = str(runtime_control_path or "").strip() or _resolve_runtime_control_path(reporting_cfg)
    runtime_obj = io_safe_read_json(runtime_path, retries=2, sleep_ms=15) or {}
    if isinstance(runtime_obj, dict):
        requested = _raw_profile(runtime_obj.get("requested_risk_profile"))
        if requested:
            return requested

    snapshot_live_path = (
        str(snapshot_path or "").strip()
        or str(reporting_cfg.get("snapshot_live_path", "outputs/snapshot_live.json")).strip()
        or "outputs/snapshot_live.json"
    )
    snapshot_obj = io_safe_read_json(snapshot_live_path, retries=2, sleep_ms=15) or {}
    if isinstance(snapshot_obj, dict):
        active = _raw_profile(snapshot_obj.get("active_risk_profile"))
        if active:
            return active

    execution_cfg = cfg.get("execution", {})
    if not isinstance(execution_cfg, dict):
        execution_cfg = {}
    config_profile = _raw_profile(execution_cfg.get("risk_profile"))
    if config_profile:
        return config_profile
    return "mid"


def resolve_widget_profile_default(
    *,
    source_profile: Any,
    current_widget_value: Any,
    previous_source_profile: Any,
) -> str:
    """Pure helper for deciding widget default without writing widget key after creation."""
    source_norm = normalize_risk_profile(source_profile, default="mid")
    current_norm = str(current_widget_value or "").strip().lower()
    prev_norm = str(previous_source_profile or "").strip().lower()
    if current_norm not in _CHOICES:
        return source_norm
    if prev_norm in _CHOICES and prev_norm != source_norm and current_norm == prev_norm:
        return source_norm
    return current_norm


def format_risk_profile_status(active: Any, requested: Any) -> Dict[str, Any]:
    """Normalize active/requested profiles and return display-ready status."""
    active_n = _norm_profile(active)
    requested_raw = str(requested or "").strip().lower()
    requested_n = _norm_profile(requested_raw) if requested_raw else active_n
    pending = bool(requested_n != active_n)

    return {
        "active": active_n,
        "requested": requested_n,
        "pending": pending,
        "active_text": f"Active Risk Profile: {active_n.upper()}",
        "requested_text": (
            f"Requested Risk Profile: {requested_n.upper()} (pending)"
            if pending
            else f"Requested Risk Profile: {requested_n.upper()} (active)"
        ),
    }


def set_filter_to_active(session_state: Dict[str, Any], snapshot: Dict[str, Any], key: str = "diag_risk_profile_filter") -> str:
    """Set diagnostics risk-profile filter to snapshot active profile."""
    active = _norm_profile((snapshot or {}).get("active_risk_profile"))
    if isinstance(session_state, dict):
        session_state[key] = active
    return active


__all__ = [
    "format_risk_profile_status",
    "set_filter_to_active",
    "get_active_risk_profile",
    "resolve_widget_profile_default",
    "resolve_effective_reporting_cfg",
]
