from __future__ import annotations

from paper_trading import build_drift_report


def test_drift_rules_risk_gate_metric_is_critical():
    expected = {
        "risk_gate_decision": {
            "metric_name": "known_weight",
            "metric_value": 0.40,
            "threshold": 0.70,
            "reason": "portfolio_cov_rc_limit",
        },
        "execution_summary": {"orders_place": 0, "orders_skip": 1},
        "asset_policy_mode": "ALLOW_ORIGINAL",
        "no_trade_summary": {
            "has_trade": False,
            "top_blockers": [{"reason": "RISK_GATE", "count": 1}],
        },
    }
    actual = {
        "risk_gate_decision": {
            "metric_name": "known_weight",
            "metric_value": 1.00,
            "threshold": 0.70,
            "reason": "",
        },
        "execution_summary": {"orders_place": 0, "orders_skip": 1},
        "asset_policy_mode": "ALLOW_ORIGINAL",
        "no_trade_summary": {
            "has_trade": False,
            "top_blockers": [{"reason": "RISK_GATE", "count": 1}],
        },
    }
    report = build_drift_report(expected, actual)
    diffs = report.get("diffs", [])
    assert isinstance(diffs, list) and diffs
    assert any(str(d.get("severity")) == "CRITICAL" for d in diffs)
    summary = report.get("summary", {})
    assert int((summary.get("severity_counts", {}) or {}).get("CRITICAL", 0)) >= 1
    assert bool(summary.get("pass", True)) is False

