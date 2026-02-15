"""Pure logic helpers for reports/statistics trading-day windows."""

from __future__ import annotations

from ui_window_presets import format_till_now, get_window_preset


def select_window_effective_count(available_days: int, window_key: str) -> dict:
    """Compute effective window size and user-facing status message."""
    preset = get_window_preset(window_key)
    required_days = int(preset.get("trading_days", 21) or 21)

    try:
        available = int(available_days)
    except Exception:
        available = 0

    if available <= 0:
        return {
            "required_days": required_days,
            "available_days": 0,
            "effective_days": 0,
            "status": "no_data",
            "message": "No data yet",
        }

    effective_days = min(available, required_days)
    if available < required_days:
        return {
            "required_days": required_days,
            "available_days": available,
            "effective_days": effective_days,
            "status": "till_now",
            "message": format_till_now(available, required_days),
        }

    return {
        "required_days": required_days,
        "available_days": available,
        "effective_days": effective_days,
        "status": "ok",
        "message": f"Coverage: {required_days}/{required_days} trading days",
    }

