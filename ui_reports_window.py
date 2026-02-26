"""Pure logic helpers for reports/statistics trading-day windows."""

from __future__ import annotations

from typing import Any, Dict, List

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
