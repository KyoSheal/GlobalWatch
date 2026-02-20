"""UI-side helpers for risk profile display logic."""

from __future__ import annotations

from typing import Dict


_RISK_PROFILE_CHOICES = {"low", "mid", "high", "ultra"}


def _norm_profile(value, default: str = "mid") -> str:
    text = str(value or "").strip().lower()
    return text if text in _RISK_PROFILE_CHOICES else default


def format_risk_profile_status(active, requested) -> Dict[str, object]:
    """Return normalized active/requested profile state for UI display."""
    active_norm = _norm_profile(active, default="mid")
    requested_norm = _norm_profile(requested, default=active_norm)
    pending = requested_norm != active_norm
    return {
        "active": active_norm,
        "requested": requested_norm,
        "pending": pending,
        "active_text": f"Active Risk Profile: {active_norm.upper()}",
        "requested_text": (
            f"Requested Risk Profile: {requested_norm.upper()} (pending)"
            if pending
            else f"Requested Risk Profile: {requested_norm.upper()} (active)"
        ),
    }


def set_filter_to_active(session_state: dict, snapshot: dict) -> str:
    """Set diagnostics risk-profile filter to active profile from snapshot."""
    if not isinstance(session_state, dict):
        raise TypeError("session_state must be a dict")
    active_norm = _norm_profile((snapshot or {}).get("active_risk_profile"), default="mid")
    session_state["diag_risk_profile_filter"] = active_norm
    return active_norm
