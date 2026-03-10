"""Pure logic helpers for reports/statistics trading-day windows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def summarize_risk_model_health(report: Dict[str, Any]) -> Dict[str, Any]:
    report_obj = report if isinstance(report, dict) else {}
    health = report_obj.get("risk_model_health", {}) if isinstance(report_obj.get("risk_model_health"), dict) else {}
    gate = health.get("risk_gate", {}) if isinstance(health.get("risk_gate"), dict) else {}
    coverage = health.get("coverage", {}) if isinstance(health.get("coverage"), dict) else {}
    execution = health.get("execution", {}) if isinstance(health.get("execution"), dict) else {}
    cost = health.get("cost", {}) if isinstance(health.get("cost"), dict) else {}

    metric_name = str(gate.get("metric_name", "")).strip()
    metric_value = gate.get("metric_value")
    summary = {
        "date": str(report_obj.get("date", health.get("date", ""))),
        "triggered": bool(gate.get("triggered", False)),
        "reason": str(gate.get("reason", "")).strip(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_threshold": gate.get("threshold"),
        "stage": str(gate.get("stage", "unknown") or "unknown"),
        "returns_missing_count": int(_to_float(coverage.get("returns_missing_count"), 0.0)),
        "cov_missing_count": int(_to_float(coverage.get("cov_missing_count"), 0.0))
        if coverage.get("cov_missing_count", None) is not None
        else None,
        "orders_place": int(_to_float(execution.get("orders_place"), 0.0)),
        "orders_skip": int(_to_float(execution.get("orders_skip"), 0.0)),
        "cost_bps": _to_float(cost.get("cost_bps"), 0.0) if cost.get("cost_bps", None) is not None else None,
        "cost_total": _to_float(cost.get("cost_total"), 0.0),
    }
    return summary


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_price_freshness(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    price_debug = snapshot.get("price_debug", {}) if isinstance(snapshot.get("price_debug"), dict) else {}
    rows = [v for v in price_debug.values() if isinstance(v, dict)]
    if not rows:
        return {
            "price_age_min": None,
            "price_live_count": 0,
            "price_recent_count": 0,
            "price_stale_count": 0,
            "price_missing_count": 0,
        }
    ages = []
    live = 0
    recent = 0
    stale = 0
    missing = 0
    for row in rows:
        status = str(row.get("status", "")).strip().upper()
        if status == "LIVE":
            live += 1
        elif status == "RECENT":
            recent += 1
        elif status == "STALE":
            stale += 1
        else:
            missing += 1
        try:
            age_val = row.get("age_min")
            if age_val is not None:
                ages.append(float(age_val))
        except Exception:
            continue
    return {
        "price_age_min": max(ages) if ages else None,
        "price_live_count": int(live),
        "price_recent_count": int(recent),
        "price_stale_count": int(stale),
        "price_missing_count": int(missing),
    }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def summarize_live_cycle_health(
    snapshot: Dict[str, Any],
    trades: List[Dict[str, Any]] | None = None,
    latest_daily_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    snap = snapshot if isinstance(snapshot, dict) else {}
    trades_rows = trades if isinstance(trades, list) else []
    daily = latest_daily_report if isinstance(latest_daily_report, dict) else {}

    market_session = snap.get("market_session", {}) if isinstance(snap.get("market_session"), dict) else {}
    gate_payload = snap.get("risk_gate_decision", {}) if isinstance(snap.get("risk_gate_decision"), dict) else {}
    execution = snap.get("execution_summary", {}) if isinstance(snap.get("execution_summary"), dict) else {}
    rebalance_gate = snap.get("rebalance_gate", {}) if isinstance(snap.get("rebalance_gate"), dict) else {}
    skip_reason = str(snap.get("rebalance_skipped_reason", "")).strip()

    last_cycle_ts = (
        _clean_text(snap.get("timestamp"))
        or _clean_text(snap.get("timestamp_utc"))
        or _clean_text(market_session.get("now_utc"))
    )
    session_state = (
        str(market_session.get("state", "")).strip()
        or str(rebalance_gate.get("session_state", "")).strip()
        or "UNKNOWN"
    )

    gate_final = str(gate_payload.get("final_gate_decision", "")).strip().upper()
    if not gate_final:
        if skip_reason.startswith("risk_gate:"):
            gate_final = "ABORT"
        elif skip_reason:
            gate_final = "SKIP"
        elif bool(rebalance_gate.get("allowed", False)):
            gate_final = "ALLOW"
        else:
            gate_final = "UNKNOWN"

    gate_reason = (
        str(gate_payload.get("reason", "")).strip()
        or skip_reason
        or str(rebalance_gate.get("reason_detail", "")).strip()
        or str(rebalance_gate.get("reason", "")).strip()
    )

    returns_age_min = None
    cov_age_min = None
    for key in ("returns_age_min", "returns_freshness_min"):
        try:
            if snap.get(key, None) is not None:
                returns_age_min = float(snap.get(key))
                break
        except Exception:
            continue
    for key in ("cov_age_min", "cov_freshness_min"):
        try:
            if snap.get(key, None) is not None:
                cov_age_min = float(snap.get(key))
                break
        except Exception:
            continue

    price_fresh = _extract_price_freshness(snap)

    fills_count_today = None
    if snap.get("fills_count_today", None) is not None:
        try:
            fills_count_today = int(float(snap.get("fills_count_today", 0)))
        except Exception:
            fills_count_today = None
    last_trade_ts = (
        _clean_text(snap.get("last_trade_ts"))
        or _clean_text(snap.get("last_trade_time"))
        or ""
    )
    if fills_count_today is None or not last_trade_ts:
        try:
            et = ZoneInfo("America/New_York")
        except Exception:
            et = timezone.utc
        trading_date_et = str(market_session.get("trading_date_et", "")).strip()
        if not trading_date_et:
            cycle_dt = _parse_iso_utc(last_cycle_ts)
            trading_date_et = cycle_dt.astimezone(et).date().isoformat() if cycle_dt else ""
        matched = 0
        latest_trade_dt = None
        for row in trades_rows:
            if not isinstance(row, dict):
                continue
            t = _parse_iso_utc(row.get("timestamp") or row.get("time"))
            if t is None:
                continue
            if trading_date_et and t.astimezone(et).date().isoformat() != trading_date_et:
                continue
            matched += 1
            if latest_trade_dt is None or t > latest_trade_dt:
                latest_trade_dt = t
        if fills_count_today is None:
            fills_count_today = matched
        if not last_trade_ts and latest_trade_dt is not None:
            last_trade_ts = latest_trade_dt.isoformat()

    orders_place = int(_to_float(execution.get("orders_place"), 0.0))
    orders_skip = int(_to_float(execution.get("orders_skip"), 0.0))

    daily_generated_at = _clean_text(daily.get("generated_at_local")) or ""
    daily_date = _clean_text(daily.get("date")) or ""

    return {
        "source": "snapshot_live.json" if bool(snap) else "daily_fallback",
        "last_cycle_ts": last_cycle_ts or None,
        "market_session": session_state,
        "active_risk_profile": str(snap.get("active_risk_profile", "")).strip().lower() or None,
        "requested_risk_profile": str(snap.get("requested_risk_profile", "")).strip().lower() or None,
        "risk_profile_source": str(snap.get("risk_profile_source", "")).strip() or None,
        "final_gate_decision": gate_final or None,
        "gate_reason": gate_reason or None,
        "orders_place": orders_place,
        "orders_skip": orders_skip,
        "returns_age_min": returns_age_min,
        "cov_age_min": cov_age_min,
        "price_age_min": price_fresh.get("price_age_min"),
        "price_live_count": price_fresh.get("price_live_count"),
        "price_recent_count": price_fresh.get("price_recent_count"),
        "price_stale_count": price_fresh.get("price_stale_count"),
        "price_missing_count": price_fresh.get("price_missing_count"),
        "fills_count_today": int(fills_count_today or 0),
        "last_trade_ts": last_trade_ts or None,
        "daily_report_date": daily_date or None,
        "daily_report_generated_at": daily_generated_at or None,
        "fallback_to_daily": False if snap else True,
    }


def build_risk_health_trend_rows(reports: List[Dict[str, Any]], max_rows: int = 7) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reports_obj = [x for x in reports if isinstance(x, dict)]
    reports_obj.sort(key=lambda x: str(x.get("date", "")))
    for report in reports_obj[-max(1, int(max_rows)):]:
        summary = summarize_risk_model_health(report)
        rows.append(
            {
                "date": summary.get("date"),
                "triggered": bool(summary.get("triggered", False)),
                "reason": summary.get("reason", ""),
                "metric_value": summary.get("metric_value"),
                "stage": summary.get("stage", "unknown"),
                "returns_missing_count": int(_to_float(summary.get("returns_missing_count"), 0.0)),
                "orders_place": int(_to_float(summary.get("orders_place"), 0.0)),
                "orders_skip": int(_to_float(summary.get("orders_skip"), 0.0)),
                "cost_bps": summary.get("cost_bps"),
            }
        )
    return rows


def format_ui_health_preview(summary: Dict[str, Any]) -> str:
    date_str = str(summary.get("date", "")).strip() or "N/A"
    triggered = bool(summary.get("triggered", False))
    metric_name = str(summary.get("metric_name", "")).strip() or "N/A"
    metric_value = summary.get("metric_value")
    missing = int(_to_float(summary.get("returns_missing_count"), 0.0))
    place = int(_to_float(summary.get("orders_place"), 0.0))
    cost_bps = summary.get("cost_bps")
    return (
        f"[UI_HEALTH_PREVIEW] date={date_str} triggered={str(triggered).lower()} "
        f"metric={metric_name}:{metric_value} missing={missing} place={place} cost_bps={cost_bps}"
    )


def format_ui_health_preview_live(summary: Dict[str, Any]) -> str:
    s = summary if isinstance(summary, dict) else {}
    last_trade_ts = s.get("last_trade_ts") or "N/A"
    gate_reason = s.get("gate_reason") or "none"
    return (
        "[UI_HEALTH_PREVIEW] "
        f"mode=live source={s.get('source')} "
        f"last_cycle_ts={s.get('last_cycle_ts')} "
        f"session={s.get('market_session')} "
        f"active={s.get('active_risk_profile')} requested={s.get('requested_risk_profile')} "
        f"risk_source={s.get('risk_profile_source')} "
        f"gate={s.get('final_gate_decision')} reason={gate_reason} "
        f"price_age_min={s.get('price_age_min')} fills_today={s.get('fills_count_today')} "
        f"last_trade_ts={last_trade_ts}"
    )


def format_ui_health_preview_daily(summary: Dict[str, Any], generated_at: str | None = None) -> str:
    date_str = str(summary.get("date", "")).strip() or "N/A"
    triggered = bool(summary.get("triggered", False))
    metric_name = str(summary.get("metric_name", "")).strip() or "N/A"
    metric_value = summary.get("metric_value")
    missing = int(_to_float(summary.get("returns_missing_count"), 0.0))
    place = int(_to_float(summary.get("orders_place"), 0.0))
    return (
        "[UI_HEALTH_PREVIEW_DAILY] "
        f"date={date_str} generated_at={generated_at or 'N/A'} "
        f"triggered={str(triggered).lower()} "
        f"metric={metric_name}:{metric_value} missing={missing} place={place}"
    )
