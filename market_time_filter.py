from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Iterable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

try:
    import pytz  # type: ignore
except Exception:  # pragma: no cover
    pytz = None  # type: ignore[assignment]


def _resolve_tz(tz_name: str):
    name = str(tz_name or "America/New_York").strip() or "America/New_York"
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    if pytz is not None:
        try:
            return pytz.timezone(name)
        except Exception:
            pass
    return timezone.utc


def parse_iso_to_utc(ts_any: Any) -> datetime:
    if isinstance(ts_any, datetime):
        dt = ts_any
    else:
        text = str(ts_any or "").strip()
        if not text:
            raise ValueError("empty timestamp")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_market_dt(dt_utc: datetime, market_tz: str) -> datetime:
    if not isinstance(dt_utc, datetime):
        raise TypeError("dt_utc must be datetime")
    if dt_utc.tzinfo is None or dt_utc.tzinfo.utcoffset(dt_utc) is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(_resolve_tz(market_tz))


def _parse_hhmm(value: Any, fallback: str) -> dt_time:
    text = str(value or fallback).strip()
    if ":" not in text:
        text = fallback
    try:
        hh_s, mm_s = text.split(":", 1)
        hh = max(0, min(23, int(hh_s)))
        mm = max(0, min(59, int(mm_s)))
        return dt_time(hour=hh, minute=mm)
    except Exception:
        fh, fm = fallback.split(":")
        return dt_time(hour=int(fh), minute=int(fm))


def in_market_window(
    dt_market: datetime,
    open_time_et: Any,
    close_time_et: Any,
    open_grace_min: int = 15,
    close_grace_min: int = 10,
) -> bool:
    if not isinstance(dt_market, datetime):
        return False
    open_t = _parse_hhmm(open_time_et, "09:30")
    close_t = _parse_hhmm(close_time_et, "16:00")
    open_dt = datetime.combine(dt_market.date(), open_t, tzinfo=dt_market.tzinfo)
    close_dt = datetime.combine(dt_market.date(), close_t, tzinfo=dt_market.tzinfo)
    open_dt = open_dt - timedelta(minutes=max(0, int(open_grace_min or 0)))
    close_dt = close_dt + timedelta(minutes=max(0, int(close_grace_min or 0)))
    return bool(open_dt <= dt_market <= close_dt)


def _row_ts_value(row: dict[str, Any]) -> Any:
    for key in ("ts", "time", "timestamp"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return val
    return None


def _parse_blackout_dates(values: Iterable[Any] | None) -> set[date]:
    out: set[date] = set()
    if values is None:
        values = {"2026-02-15"}
    for val in values:
        try:
            out.add(date.fromisoformat(str(val).strip()))
        except Exception:
            continue
    return out


def sanitize_equity_rows(
    rows,
    *,
    market_tz: str,
    open_time_et: Any,
    close_time_et: Any,
    open_grace_min: int = 15,
    close_grace_min: int = 10,
    drop_weekends: bool = True,
    drop_offhours: bool = True,
    blackout_dates_market: Iterable[Any] | None = None,
):
    clean_rows: list[dict[str, Any]] = []
    stats = {
        "total_in": 0,
        "kept": 0,
        "dropped_blackout": 0,
        "dropped_weekend": 0,
        "dropped_offhours": 0,
        "dropped_invalid_ts": 0,
    }
    if not isinstance(rows, list):
        return clean_rows, stats

    blackout_dates = _parse_blackout_dates(blackout_dates_market)

    for row in rows:
        if not isinstance(row, dict):
            continue
        stats["total_in"] += 1
        ts_val = _row_ts_value(row)
        try:
            dt_utc = parse_iso_to_utc(ts_val)
            dt_market = to_market_dt(dt_utc, market_tz)
        except Exception:
            stats["dropped_invalid_ts"] += 1
            continue

        m_date = dt_market.date()
        if m_date in blackout_dates:
            stats["dropped_blackout"] += 1
            continue
        if drop_weekends and dt_market.weekday() >= 5:
            stats["dropped_weekend"] += 1
            continue
        if drop_offhours and (not in_market_window(dt_market, open_time_et, close_time_et, open_grace_min, close_grace_min)):
            stats["dropped_offhours"] += 1
            continue

        clean_rows.append(dict(row))
        stats["kept"] += 1

    return clean_rows, stats


__all__ = [
    "in_market_window",
    "parse_iso_to_utc",
    "sanitize_equity_rows",
    "to_market_dt",
]
