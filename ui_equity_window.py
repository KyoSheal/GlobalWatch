"""Equity curve window helpers (pure logic, no Streamlit dependencies)."""

from __future__ import annotations

import pandas as pd

from ui_window_presets import get_window_preset


def _to_trade_date(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    t = pd.Timestamp(ts)

    # Preferred: convert to America/New_York and take date.
    try:
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        t_ny = t.tz_convert("America/New_York")
        return t_ny.date()
    except Exception:
        pass

    # Fallback: UTC date.
    try:
        t_utc = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t_utc.date()
    except Exception:
        pass

    try:
        return t.date()
    except Exception:
        return None


def filter_df_by_trading_day_window(df, window_key):
    """Filter an equity DataFrame by trading-day window.

    Returns:
      (df_filtered, available_days, required_days)
    """
    preset = get_window_preset(window_key)
    required_days = int(preset.get("trading_days", 21) or 21)

    if not isinstance(df, pd.DataFrame) or df.empty or "time" not in df.columns:
        empty = pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else ["time", "equity"])
        if "trade_date" not in empty.columns:
            empty["trade_date"] = pd.Series(dtype="object")
        return empty, 0, required_days

    work = df.copy()
    work["trade_date"] = work["time"].map(_to_trade_date)
    work = work.dropna(subset=["trade_date"])
    if work.empty:
        return work, 0, required_days

    unique_dates = sorted(pd.unique(work["trade_date"]).tolist())
    available_days = int(len(unique_dates))
    keep_dates = set(unique_dates[-required_days:]) if required_days > 0 else set()
    if keep_dates:
        filtered = work[work["trade_date"].isin(keep_dates)].copy()
    else:
        filtered = work.iloc[0:0].copy()
    return filtered, available_days, required_days

