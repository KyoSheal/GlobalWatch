"""Cooldown policy helpers for rebalance scheduling."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Optional
import random

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_COOLDOWN_CFG: Dict[str, Any] = {
    "success_cooldown_min": 90.0,
    "failure_backoff": {
        "default": {"base": 5.0, "cap": 30.0},
        "portfolio_cov_rc_limit": {"base": 5.0, "cap": 30.0},
        "broker_error": {"base": 2.0, "cap": 60.0},
        "hard_risk_limit": {"base": 10.0, "cap": 60.0},
    },
    "enable_jitter": False,
    "jitter_pct": 0.10,
}


def _coerce_zone(tz_name: Optional[str]):
    if ZoneInfo is None:
        return datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(str(tz_name or "America/New_York"))
    except Exception:
        return ZoneInfo("America/New_York")


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


def _coerce_aware_utc(now: Any) -> datetime:
    if isinstance(now, datetime):
        dt = now
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _next_business_day(day):
    out = day
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def next_market_open_time(
    now: Any,
    *,
    tz_market: str = "America/New_York",
    open_time_et: Any = "09:30",
) -> datetime:
    """Return next market-open timestamp (timezone-aware UTC)."""
    now_utc = _coerce_aware_utc(now)
    market_tz = _coerce_zone(tz_market)
    now_local = now_utc.astimezone(market_tz)
    open_tm = _coerce_time(open_time_et, time(9, 30))

    candidate_day = now_local.date()
    candidate_day = _next_business_day(candidate_day)
    candidate_local = datetime.combine(candidate_day, open_tm, tzinfo=market_tz)
    if candidate_local <= now_local:
        candidate_day = _next_business_day(candidate_day + timedelta(days=1))
        candidate_local = datetime.combine(candidate_day, open_tm, tzinfo=market_tz)
    return candidate_local.astimezone(timezone.utc)


def normalize_cooldown_reason(reason: Any) -> str:
    raw = str(reason or "").strip().lower()
    if not raw:
        return "other_fail"
    if raw in {"portfolio_cov_rc_limit"}:
        return "portfolio_cov_rc_limit"
    if raw in {"broker_error", "order_reject", "api_error"}:
        return "broker_error"
    if raw in {"hard_risk_limit", "portfolio_volatility", "diversity_hhi"}:
        return "hard_risk_limit"
    if raw in {
        "market_closed_gate",
        "state_pre_open",
        "state_post_close",
        "state_weekend",
        "pre_open",
        "post_close",
        "weekend",
        "open_grace_not_passed",
    }:
        return raw
    if raw.startswith("risk_gate:"):
        sub = raw.split(":", 1)[1].strip().lower()
        if sub == "portfolio_cov_rc_limit":
            return "portfolio_cov_rc_limit"
        if sub in {"portfolio_volatility", "diversity_hhi"}:
            return "hard_risk_limit"
        return "other_fail"
    return "other_fail"


def _coerce_float(value: Any, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if v < 0:
        return 0.0
    return v


def _merge_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "success_cooldown_min": float(DEFAULT_COOLDOWN_CFG["success_cooldown_min"]),
        "failure_backoff": {
            "default": dict(DEFAULT_COOLDOWN_CFG["failure_backoff"]["default"]),
            "portfolio_cov_rc_limit": dict(DEFAULT_COOLDOWN_CFG["failure_backoff"]["portfolio_cov_rc_limit"]),
            "broker_error": dict(DEFAULT_COOLDOWN_CFG["failure_backoff"]["broker_error"]),
            "hard_risk_limit": dict(DEFAULT_COOLDOWN_CFG["failure_backoff"]["hard_risk_limit"]),
        },
        "enable_jitter": bool(DEFAULT_COOLDOWN_CFG["enable_jitter"]),
        "jitter_pct": float(DEFAULT_COOLDOWN_CFG["jitter_pct"]),
    }
    user_cfg = cfg if isinstance(cfg, dict) else {}
    base["success_cooldown_min"] = _coerce_float(user_cfg.get("success_cooldown_min", base["success_cooldown_min"]), base["success_cooldown_min"])
    base["enable_jitter"] = bool(user_cfg.get("enable_jitter", base["enable_jitter"]))
    base["jitter_pct"] = _coerce_float(user_cfg.get("jitter_pct", base["jitter_pct"]), base["jitter_pct"])

    user_failure = user_cfg.get("failure_backoff", {})
    if isinstance(user_failure, dict):
        for key in ("default", "portfolio_cov_rc_limit", "broker_error", "hard_risk_limit"):
            item = user_failure.get(key, {})
            if not isinstance(item, dict):
                continue
            b = _coerce_float(item.get("base", base["failure_backoff"][key]["base"]), base["failure_backoff"][key]["base"])
            c = _coerce_float(item.get("cap", base["failure_backoff"][key]["cap"]), base["failure_backoff"][key]["cap"])
            if c < b:
                c = b
            base["failure_backoff"][key] = {"base": b, "cap": c}
    return base


def cooldown_policy(
    now: Any,
    outcome: str,
    reason: Any,
    state: Optional[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]],
    *,
    next_open_override: Optional[datetime] = None,
    tz_market: str = "America/New_York",
    open_time_et: Any = "09:30",
) -> Dict[str, Any]:
    """
    Compute cooldown decision.

    Returns:
      {
        "outcome": ...,
        "reason": normalized_reason,
        "backoff_min": float,
        "next_allowed_ts": datetime (UTC-aware),
        "next_allowed_ts_iso": str,
        "fail_count": int,
        "policy": str,
        "state": updated_state_dict,
      }
    """
    now_utc = _coerce_aware_utc(now)
    norm_cfg = _merge_cfg(cfg)
    reason_key = normalize_cooldown_reason(reason)
    outcome_u = str(outcome or "").strip().upper()
    st = dict(state or {})
    fail_map = st.get("consecutive_fail_by_reason", {})
    if not isinstance(fail_map, dict):
        fail_map = {}
    fail_map = {str(k): int(v) for k, v in fail_map.items() if str(k)}

    backoff_min = 0.0
    fail_count = 0
    policy_name = "none"
    enable_jitter = bool(norm_cfg.get("enable_jitter", False))
    jitter_pct = max(0.0, float(norm_cfg.get("jitter_pct", 0.10) or 0.10))

    if outcome_u == "SUCCESS_TRADE":
        fail_map = {}
        backoff_min = float(norm_cfg.get("success_cooldown_min", 0.0) or 0.0)
        if enable_jitter and backoff_min > 0 and jitter_pct > 0:
            backoff_min = max(0.0, backoff_min * (1.0 + random.uniform(-jitter_pct, jitter_pct)))
        policy_name = "success_cooldown"
        next_allowed = now_utc + timedelta(minutes=max(0.0, backoff_min))
    elif outcome_u == "SKIP_MARKET_CLOSED":
        # Do not escalate fail counters on market-state skip.
        fail_count = int(fail_map.get(reason_key, 0) or 0)
        if isinstance(next_open_override, datetime):
            next_allowed = _coerce_aware_utc(next_open_override)
        else:
            next_allowed = next_market_open_time(now_utc, tz_market=tz_market, open_time_et=open_time_et)
        backoff_min = max(0.0, (next_allowed - now_utc).total_seconds() / 60.0)
        policy_name = "next_market_open"
    else:
        policy_name = "failure_backoff"
        fail_count = int(fail_map.get(reason_key, 0) or 0) + 1
        fail_map[reason_key] = fail_count
        reason_policy_key = reason_key if reason_key in norm_cfg.get("failure_backoff", {}) else "default"
        reason_cfg = norm_cfg["failure_backoff"].get(reason_policy_key, norm_cfg["failure_backoff"]["default"])
        base = float(reason_cfg.get("base", 5.0) or 5.0)
        cap = float(reason_cfg.get("cap", 30.0) or 30.0)
        if cap < base:
            cap = base
        backoff_min = min(cap, base * (2 ** max(0, fail_count - 1)))
        if enable_jitter and backoff_min > 0 and jitter_pct > 0:
            backoff_min = min(cap, max(0.0, backoff_min * (1.0 + random.uniform(-jitter_pct, jitter_pct))))
        next_allowed = now_utc + timedelta(minutes=max(0.0, backoff_min))

    st["consecutive_fail_by_reason"] = fail_map
    st["last_outcome"] = outcome_u
    st["last_reason"] = reason_key
    st["last_backoff_min"] = float(backoff_min)
    st["last_policy"] = str(policy_name)
    st["last_next_allowed_ts"] = next_allowed.isoformat()

    return {
        "outcome": outcome_u,
        "reason": reason_key,
        "backoff_min": float(backoff_min),
        "next_allowed_ts": next_allowed,
        "next_allowed_ts_iso": next_allowed.isoformat(),
        "fail_count": int(fail_count),
        "policy": str(policy_name),
        "state": st,
    }
