from __future__ import annotations

import daily_reporter


def test_risk_model_health_force_proxy_has_trade():
    report = {
        "date": "2026-02-10",
        "no_trade_summary": {
            "top_blockers": [],
            "gate_reason": "",
            "policy": {"asset_policy_mode": "FORCE_PROXY", "execution_proxy_used": True},
        },
        "risk_gate_decision": {
            "reason": "",
            "metric_name": "max_rc_fraction",
            "metric_value": 0.22,
            "threshold": 0.35,
            "stage": "returns",
        },
        "cov_coverage": {"known_weight": 0.95, "missing_weight_total": 0.05, "missing_count": 1},
        "returns_coverage_diag": {"schema_version": 1, "items": []},
        "execution_summary": {"orders_place": 2, "orders_skip": 0, "skip_reasons": {}},
        "cost_summary": {"enabled": True, "totals": {"total": 2.0}, "cost_bps": 4.0, "trades_count": 2},
        "asset_policy_mode": "FORCE_PROXY",
        "execution_proxy_used": True,
    }
    out = daily_reporter.build_risk_model_health(report=report, snapshot=report, daily_fields=None)
    assert out["schema_version"] == 1
    assert out["risk_gate"]["triggered"] is False
    assert out["execution"]["orders_place"] == 2
    assert out["coverage"]["returns_missing_count"] == 0
    assert out["policy"]["asset_policy_mode"] == "FORCE_PROXY"
    assert out["policy"]["execution_proxy_used"] is True


def test_risk_model_health_allow_original_triggered_missing():
    report = {
        "date": "2026-02-10",
        "no_trade_summary": {
            "top_blockers": [{"reason": "PRICE_MISSING", "count": 2}, {"reason": "RISK_GATE", "count": 1}],
            "gate_reason": "portfolio_cov_rc_limit",
            "policy": {"asset_policy_mode": "ALLOW_ORIGINAL", "execution_proxy_used": False},
        },
        "risk_gate_decision": {
            "reason": "portfolio_cov_rc_limit",
            "metric_name": "max_rc_fraction",
            "metric_value": 0.92,
            "threshold": 0.35,
            "stage": "returns",
        },
        "cov_coverage": {"known_weight": 0.49, "missing_weight_total": 0.20, "missing_count": 2},
        "returns_coverage_diag": {
            "schema_version": 1,
            "items": [
                {"ticker": "XIU.TO", "reason_code": "PRICE_MISSING"},
                {"ticker": "FTS.TO", "reason_code": "PRICE_MISSING"},
            ],
        },
        "execution_summary": {
            "orders_place": 0,
            "orders_skip": 3,
            "skip_reasons": {"PRICE_MISSING": 2, "RISK_GATE": 1},
        },
        "cost_summary": {"enabled": True, "totals": {"total": 0.0}, "cost_bps": 0.0, "trades_count": 0},
        "asset_policy_mode": "ALLOW_ORIGINAL",
        "execution_proxy_used": False,
    }
    out = daily_reporter.build_risk_model_health(report=report, snapshot=report, daily_fields=None)
    assert out["risk_gate"]["triggered"] is True
    assert out["risk_gate"]["reason"] == "portfolio_cov_rc_limit"
    assert out["coverage"]["returns_missing_count"] == 2
    assert out["coverage"]["returns_missing_top"][0]["ticker"] == "XIU.TO"
    assert out["execution"]["orders_place"] == 0
    assert out["execution"]["orders_skip"] == 3
    assert out["execution"]["top_skip_reasons"][0]["reason"] == "PRICE_MISSING"


def test_risk_model_health_cost_enabled_fields():
    report = {
        "date": "2026-02-10",
        "no_trade_summary": {"top_blockers": []},
        "risk_gate_decision": {},
        "cov_coverage": {},
        "returns_coverage_diag": {"schema_version": 1, "items": []},
        "execution_summary": {"orders_place": 1, "orders_skip": 0, "skip_reasons": {}},
        "cost_summary": {
            "enabled": True,
            "totals": {"total": 3.5},
            "cost_bps": 5.5,
            "trades_count": 1,
        },
        "asset_policy_mode": "FORCE_PROXY",
        "execution_proxy_used": True,
    }
    out = daily_reporter.build_risk_model_health(report=report, snapshot=report, daily_fields=None)
    assert out["cost"]["enabled"] is True
    assert abs(float(out["cost"]["cost_total"]) - 3.5) < 1e-9
    assert abs(float(out["cost"]["cost_bps"]) - 5.5) < 1e-9
    assert int(out["cost"]["trades_count"]) == 1
