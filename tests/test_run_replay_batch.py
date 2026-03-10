from __future__ import annotations

from tools.run_replay_batch import aggregate_batch_results, choose_primary_reason


def test_choose_primary_reason_priority_chain():
    snapshot = {
        "rebalance_skipped_reason": "",
        "risk_gate_decision": {"reason": "portfolio_cov_rc_limit"},
        "no_trade_summary": {"gate_reason": "market_closed_gate"},
        "execution_summary": {"orders_place": 0, "orders_skip": 2, "skip_reasons": {"PRICE_MISSING": 2}},
    }
    assert choose_primary_reason(snapshot) == "portfolio_cov_rc_limit"

    snapshot2 = {
        "rebalance_skipped_reason": "",
        "risk_gate_decision": {},
        "no_trade_summary": {"top_blockers": [{"reason": "COOLDOWN", "count": 3}]},
        "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"PRICE_STALE": 1}},
    }
    assert choose_primary_reason(snapshot2) == "PRICE_STALE"


def test_aggregate_batch_results_basic_counts():
    rows = [
        {
            "replay_status": "PASS",
            "primary_reason": "traded",
            "orders_place": 2,
            "orders_skip": 1,
            "fills_count": 2,
            "estimated_cost": 1.25,
            "turnover": 1000.0,
            "config_metadata_compare_status": "ok",
            "scenario_metadata_compare_status": "ok",
            "scenario_comparable_day": True,
        },
        {
            "replay_status": "FAIL",
            "primary_reason": "portfolio_cov_rc_limit",
            "orders_place": 0,
            "orders_skip": 1,
            "fills_count": 0,
            "estimated_cost": 0.0,
            "turnover": 0.0,
            "config_metadata_compare_status": "legacy_snapshot_missing_metadata",
            "scenario_metadata_compare_status": "scenario_effective_risk_model_fingerprint_mismatch",
            "scenario_comparable_day": False,
        },
    ]
    summary = aggregate_batch_results(rows)
    assert summary["days_total"] == 2
    assert summary["days_pass"] == 1
    assert summary["days_fail"] == 1
    assert summary["days_with_trades"] == 1
    assert summary["days_without_trades"] == 1
    assert summary["fills_total"] == 2
    assert summary["orders_place_total"] == 2
    assert summary["orders_skip_total"] == 2
    assert abs(float(summary["estimated_cost_total"]) - 1.25) < 1e-12
    assert summary["reason_counts"]["traded"] == 1
    assert summary["reason_counts"]["portfolio_cov_rc_limit"] == 1
    assert summary["config_metadata_status_counts"]["ok"] == 1
    assert summary["config_metadata_status_counts"]["legacy_snapshot_missing_metadata"] == 1
    assert summary["comparable_days_count"] == 1
    assert summary["non_comparable_days_count"] == 1
    assert summary["scenario_metadata_status_counts"]["ok"] == 1
    assert summary["scenario_metadata_status_counts"]["scenario_effective_risk_model_fingerprint_mismatch"] == 1
    assert summary["scenario_comparable_days_count"] == 1
    assert summary["scenario_non_comparable_days_count"] == 1
