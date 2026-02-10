from __future__ import annotations

from datetime import datetime, time, timedelta, date as date_cls, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _coerce_zone(tz_name: Optional[str]):
    if ZoneInfo is None:
        return datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(str(tz_name or "America/New_York"))
    except Exception:
        return ZoneInfo("America/New_York")


def _parse_datetime(value: Any, tz_name: Optional[str] = None) -> Optional[datetime]:
    tzinfo = _coerce_zone(tz_name)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except Exception:
                    continue
        if dt is None:
            return None
    else:
        return None

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        # Treat naive timestamps as UTC to avoid misinterpreting local wall clock as ET.
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(tzinfo)
    except Exception:
        return dt


def _coerce_time(value: Any, default: time) -> time:
    if isinstance(value, time):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt).time()
            return parsed.replace(microsecond=0)
        except Exception:
            continue
    return default


def _previous_trading_day(day: date_cls) -> date_cls:
    out = day - timedelta(days=1)
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def _next_trading_day(day: date_cls) -> date_cls:
    out = day + timedelta(days=1)
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def get_market_session_state(
    now_dt: Any,
    tz_market: str = "America/New_York",
    open_time_et: Any = "09:30",
    close_time_et: Any = "16:00",
    open_grace_min: int = 15,
    close_grace_min: int = 10,
) -> Dict[str, Any]:
    market_tz = _coerce_zone(tz_market)
    now_parsed = _parse_datetime(now_dt, tz_market) or datetime.now(market_tz)
    now_et = now_parsed.astimezone(market_tz)
    today = now_et.date()

    open_tm = _coerce_time(open_time_et, time(9, 30))
    close_tm = _coerce_time(close_time_et, time(16, 0))
    open_dt = datetime.combine(today, open_tm, tzinfo=market_tz)
    close_dt = datetime.combine(today, close_tm, tzinfo=market_tz)
    open_grace_dt = open_dt + timedelta(minutes=max(0, int(open_grace_min)))
    close_grace_dt = close_dt + timedelta(minutes=max(0, int(close_grace_min)))

    if today.weekday() >= 5:
        state = "WEEKEND"
        trading_date = _next_trading_day(today)
        last_completed = _previous_trading_day(trading_date)
    elif now_et < open_dt:
        state = "PRE_OPEN"
        trading_date = today
        last_completed = _previous_trading_day(today)
    elif now_et < close_dt:
        state = "OPEN"
        trading_date = today
        last_completed = _previous_trading_day(today)
    else:
        state = "POST_CLOSE"
        trading_date = today
        if now_et >= close_grace_dt:
            last_completed = today
        else:
            last_completed = _previous_trading_day(today)

    return {
        "state": state,
        "now_et": now_et.isoformat(),
        "now_utc": now_et.astimezone(timezone.utc).isoformat(),
        "trading_date_et": trading_date.isoformat(),
        "last_completed_trading_date_et": last_completed.isoformat(),
        "open_time_et": open_dt.isoformat(),
        "close_time_et": close_dt.isoformat(),
        "open_grace_min": int(max(0, int(open_grace_min))),
        "close_grace_min": int(max(0, int(close_grace_min))),
        "open_grace_passed": bool(now_et >= open_grace_dt) if state == "OPEN" else False,
        "close_grace_passed": bool(now_et >= close_grace_dt) if state == "POST_CLOSE" else False,
    }


def is_market_open_for_trading(session_dict: Dict[str, Any]) -> bool:
    if not isinstance(session_dict, dict):
        return False
    return str(session_dict.get("state", "")).upper() == "OPEN" and bool(session_dict.get("open_grace_passed"))
