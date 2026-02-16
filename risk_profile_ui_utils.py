"""UI helper utilities for risk profile display state."""

from __future__ import annotations

from typing import Any, Dict

_CHOICES = {"low", "mid", "high", "ultra"}


def _norm_profile(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in _CHOICES else "mid"


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


__all__ = ["format_risk_profile_status", "set_filter_to_active"]
