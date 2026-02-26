from __future__ import annotations

import daily_reporter


def test_build_no_trade_summary_allow_original_blockers():
    trades = {"trade_count": 0}
    snapshot = {
        "execution_summary": {
            "orders_place": 0,
            "orders_skip": 3,
            "skip_reasons": {"PRICE_MISSING": 2, "RISK_GATE": 1},
        },
        "rebalance_skipped_reason": "risk_gate:portfolio_cov_rc_limit",
        "asset_policy_mode": "ALLOW_ORIGINAL",
        "execution_proxy_used": False,
        "ticker_proxy_scope": "off",
    }
    gate = {
        "metric_name": "max_rc_fraction",
        "metric_value": 0.92,
        "threshold": 0.35,
        "stage": "returns",
        "reason": "portfolio_cov_rc_limit",
    }
    cov = {"known_weight": 0.49, "missing_weight_total": 0.20}
    returns_diag = {"schema_version": 1, "items": [{"ticker": "XIU.TO", "reason_code": "PRICE_MISSING"}]}

    out = daily_reporter.build_no_trade_summary(
        trades=trades,
        snapshot=snapshot,
        risk_gate_decision=gate,
        cov_coverage=cov,
        returns_coverage_diag=returns_diag,
    )
    assert out["schema_version"] == 1
    assert out["has_trade"] is False
    assert int(out["orders_place"]) == 0
    assert int(out["orders_skip"]) == 3
    assert out["top_blockers"][0]["reason"] == "PRICE_MISSING"
    assert out["gate_reason"] == "portfolio_cov_rc_limit"
    assert out["data_issues"]["returns_missing_top"][0]["ticker"] == "XIU.TO"
    assert out["policy"]["asset_policy_mode"] == "ALLOW_ORIGINAL"


def test_build_no_trade_summary_force_proxy_has_trade():
    trades = {"trade_count": 2}
    snapshot = {
        "execution_summary": {
            "orders_place": 2,
            "orders_skip": 0,
            "skip_reasons": {},
        },
        "asset_policy_mode": "FORCE_PROXY",
        "execution_proxy_used": True,
        "ticker_proxy_scope": "risk_and_execution",
    }
    out = daily_reporter.build_no_trade_summary(
        trades=trades,
        snapshot=snapshot,
        risk_gate_decision={},
        cov_coverage={},
        returns_coverage_diag={},
    )
    assert out["has_trade"] is True
    assert int(out["orders_place"]) == 2
    assert out["policy"]["execution_proxy_used"] is True
    assert out["policy"]["proxy_scope"] == "risk_and_execution"


def test_aggregate_reports_no_trade_summary_top_blockers_order():
    reports = [
        {
            "date": "2026-02-10",
            "trades": {"buy_notional": 0, "sell_notional": 0, "net_flow": 0, "trade_count": 0},
            "equity": {"pnl": 0.0, "pnl_pct": 0.0},
            "no_trade_summary": {
                "schema_version": 1,
                "has_trade": False,
                "orders_place": 0,
                "orders_skip": 3,
                "top_blockers": [{"reason": "PRICE_MISSING", "count": 2}, {"reason": "RISK_GATE", "count": 1}],
                "gate_reason": "portfolio_cov_rc_limit",
                "gate_metric": {"metric_name": "max_rc_fraction", "metric_value": 0.92, "threshold": 0.35, "stage": "returns"},
                "data_issues": {"returns_missing_top": [{"ticker": "XIU.TO", "reason": "PRICE_MISSING"}], "cov_known_weight": 0.49, "cov_missing_weight_total": 0.2},
                "policy": {"asset_policy_mode": "ALLOW_ORIGINAL", "execution_proxy_used": False, "proxy_scope": "off"},
            },
        },
        {
            "date": "2026-02-11",
            "trades": {"buy_notional": 0, "sell_notional": 0, "net_flow": 0, "trade_count": 0},
            "equity": {"pnl": 0.0, "pnl_pct": 0.0},
            "no_trade_summary": {
                "schema_version": 1,
                "has_trade": False,
                "orders_place": 0,
                "orders_skip": 2,
                "top_blockers": [{"reason": "PRICE_MISSING", "count": 1}, {"reason": "RISK_GATE", "count": 1}],
                "gate_reason": "portfolio_cov_rc_limit",
                "gate_metric": {"metric_name": "max_rc_fraction", "metric_value": 0.91, "threshold": 0.35, "stage": "returns"},
                "data_issues": {"returns_missing_top": [{"ticker": "FTS.TO", "reason": "PRICE_MISSING"}], "cov_known_weight": 0.50, "cov_missing_weight_total": 0.19},
                "policy": {"asset_policy_mode": "ALLOW_ORIGINAL", "execution_proxy_used": False, "proxy_scope": "off"},
            },
        },
    ]
    agg = daily_reporter.aggregate_reports(reports, window_days=2)
    assert agg.get("status") == "ok"
    nt = agg.get("no_trade_summary", {})
    assert isinstance(nt, dict)
    assert nt.get("has_trade") is False
    assert int(nt.get("orders_place", 0) or 0) == 0
    assert int(nt.get("orders_skip", 0) or 0) == 5
    blockers = nt.get("top_blockers", [])
    assert isinstance(blockers, list) and blockers
    assert blockers[0]["reason"] == "PRICE_MISSING"
    assert blockers[0]["count"] == 3
