"""Pure window preset helpers for UI/reporting modules."""

from __future__ import annotations


WINDOW_PRESETS = {
    "1D": {"trading_days": 1, "calendar_days": 1, "label": "1 Day"},
    "2D": {"trading_days": 2, "calendar_days": 2, "label": "2 Days"},
    "3D": {"trading_days": 3, "calendar_days": 3, "label": "3 Days"},
    "1W": {"trading_days": 5, "calendar_days": 7, "label": "1 Week"},
    "2W": {"trading_days": 10, "calendar_days": 14, "label": "2 Weeks"},
    "1M": {"trading_days": 21, "calendar_days": 30, "label": "1 Month"},
    "3M": {"trading_days": 63, "calendar_days": 90, "label": "3 Months"},
    "6M": {"trading_days": 126, "calendar_days": 180, "label": "6 Months"},
    "1Y": {"trading_days": 252, "calendar_days": 365, "label": "1 Year"},
}


def get_window_preset(key: str) -> dict:
    """Return a window preset; fallback to 1M for unknown keys."""
    normalized = str(key or "").strip().upper()
    preset = WINDOW_PRESETS.get(normalized)
    if isinstance(preset, dict):
        return dict(preset)
    return dict(WINDOW_PRESETS["1M"])


def format_till_now(available: int, required: int) -> str:
    """Format till-now display for trading-day based windows."""
    try:
        avail = max(0, int(available))
    except Exception:
        avail = 0
    try:
        req = max(0, int(required))
    except Exception:
        req = 0
    return f"Till now: showing {avail}/{req} trading days"

