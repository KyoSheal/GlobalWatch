"""Returns coverage diagnostics helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


REASON_PRICE_MISSING = "PRICE_MISSING"
REASON_TOO_FEW_POINTS = "TOO_FEW_POINTS"
REASON_ALL_NAN = "ALL_NAN"
REASON_CALENDAR_MISMATCH = "CALENDAR_MISMATCH"
REASON_SOURCE_UNSUPPORTED = "SOURCE_UNSUPPORTED"
REASON_UNKNOWN = "UNKNOWN"


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        try:
            return value.to_pydatetime().isoformat()
        except Exception:
            return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime().isoformat()
    except Exception:
        return None
    return None


def _extract_series(payload: Any) -> tuple[pd.Series, Optional[str], Optional[bool], str]:
    if payload is None:
        return pd.Series(dtype=float), None, None, ""

    if isinstance(payload, pd.DataFrame):
        if "Close" in payload.columns:
            return payload["Close"], None, True, ""
        if payload.shape[1] >= 1:
            return payload.iloc[:, 0], None, True, ""
        return pd.Series(dtype=float), None, True, ""

    if isinstance(payload, pd.Series):
        return payload, None, True, ""

    if isinstance(payload, dict):
        last_ts = _iso_or_none(payload.get("last_price_ts"))
        source_supported = payload.get("source_supported")
        if source_supported is not None:
            source_supported = bool(source_supported)
        note = str(payload.get("note", "") or "")

        if isinstance(payload.get("series"), pd.Series):
            return payload.get("series"), last_ts, source_supported, note

        if isinstance(payload.get("history"), pd.DataFrame):
            history = payload.get("history")
            if "Close" in history.columns:
                return history["Close"], last_ts, source_supported, note
            if history.shape[1] >= 1:
                return history.iloc[:, 0], last_ts, source_supported, note

        closes = payload.get("close")
        if isinstance(closes, (list, tuple)):
            try:
                return pd.Series(list(closes), dtype=float), last_ts, source_supported, note
            except Exception:
                return pd.Series(dtype=float), last_ts, source_supported, note

        return pd.Series(dtype=float), last_ts, source_supported, note

    return pd.Series(dtype=float), None, None, ""


def diagnose_returns_coverage(
    ticker: str,
    lookback_cfg: Dict[str, Any],
    price_provider: Callable[..., Any],
    calendar_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Diagnose why returns coverage is missing for one ticker."""
    ticker_u = str(ticker or "").upper().strip()
    expected_points = max(0, _to_int(lookback_cfg.get("expected_points", lookback_cfg.get("lookback_days", 0)), 0))
    min_obs = max(1, _to_int(lookback_cfg.get("min_obs", max(5, expected_points // 2)), max(5, expected_points // 2)))
    detect_calendar_mismatch = bool(calendar_cfg.get("detect_calendar_mismatch", True))

    out = {
        "ticker": ticker_u,
        "has_prices": False,
        "last_price_ts": None,
        "expected_points": int(expected_points),
        "actual_points": 0,
        "nan_ratio": 1.0,
        "reason_code": REASON_UNKNOWN,
        "note": "",
    }

    try:
        payload = price_provider(ticker=ticker_u, lookback_cfg=dict(lookback_cfg or {}), calendar_cfg=dict(calendar_cfg or {}))
    except NotImplementedError as e:
        out["reason_code"] = REASON_SOURCE_UNSUPPORTED
        out["note"] = str(e)[:200]
        return out
    except Exception as e:
        out["reason_code"] = REASON_UNKNOWN
        out["note"] = f"provider_error={e}"[:200]
        return out

    series, last_ts_hint, source_supported, note = _extract_series(payload)
    if isinstance(note, str) and note:
        out["note"] = note[:200]
    if source_supported is False:
        out["reason_code"] = REASON_SOURCE_UNSUPPORTED
        if not out["note"]:
            out["note"] = "source_supported=false"
        return out

    if series is None or len(series) == 0:
        out["reason_code"] = REASON_PRICE_MISSING
        out["last_price_ts"] = last_ts_hint
        return out

    out["has_prices"] = True
    try:
        s = pd.to_numeric(series, errors="coerce")
    except Exception:
        s = pd.Series(dtype=float)
    total_points = int(len(s))
    non_na_points = int(s.notna().sum()) if total_points > 0 else 0
    out["actual_points"] = int(non_na_points)
    if total_points > 0:
        out["nan_ratio"] = float(max(0.0, min(1.0, 1.0 - (non_na_points / float(total_points)))))
    else:
        out["nan_ratio"] = 1.0

    idx = getattr(series, "index", None)
    last_idx_ts = None
    if idx is not None and len(idx) > 0:
        try:
            last_idx_ts = _iso_or_none(idx[-1])
        except Exception:
            last_idx_ts = None
    out["last_price_ts"] = last_idx_ts or last_ts_hint

    if non_na_points <= 0:
        out["reason_code"] = REASON_ALL_NAN
        return out

    if detect_calendar_mismatch and idx is not None and len(idx) > 0:
        try:
            ts_index = pd.to_datetime(idx, errors="coerce")
            weekend_points = int(sum(1 for ts in ts_index if not pd.isna(ts) and int(ts.dayofweek) >= 5))
            if weekend_points > 0:
                out["reason_code"] = REASON_CALENDAR_MISMATCH
                if not out["note"]:
                    out["note"] = f"weekend_points={weekend_points}"
                return out
        except Exception:
            pass

    if non_na_points < int(min_obs):
        out["reason_code"] = REASON_TOO_FEW_POINTS
        return out

    if expected_points > 0 and non_na_points < int(max(1, expected_points * 0.5)):
        out["reason_code"] = REASON_TOO_FEW_POINTS
        return out

    out["reason_code"] = REASON_UNKNOWN
    if not out["note"]:
        out["note"] = "unclassified"
    return out

