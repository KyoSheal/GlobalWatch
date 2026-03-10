from __future__ import annotations

import math

import paper_trading


def test_portfolio_cov_rc_hysteresis_band_zero_backward_compatible():
    d_abort = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.31,
        rc_limit=0.30,
        hysteresis_band=0.0,
        previous_gate_decision="ALLOW",
    )
    d_allow = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.30,
        rc_limit=0.30,
        hysteresis_band=0.0,
        previous_gate_decision="ABORT",
    )
    assert d_abort["final_gate_decision"] == "ABORT"
    assert d_allow["final_gate_decision"] == "ALLOW"
    assert d_abort["sticky_zone"] is False
    assert d_allow["sticky_zone"] is False


def test_portfolio_cov_rc_hysteresis_threshold_edges():
    d_pass = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.26,
        rc_limit=0.30,
        hysteresis_band=0.03,
        previous_gate_decision=None,
    )
    d_fail = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.34,
        rc_limit=0.30,
        hysteresis_band=0.03,
        previous_gate_decision=None,
    )
    assert math.isclose(float(d_pass["pass_threshold"]), 0.27, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(d_pass["fail_threshold"]), 0.33, rel_tol=0.0, abs_tol=1e-12)
    assert d_pass["final_gate_decision"] == "ALLOW"
    assert d_fail["final_gate_decision"] == "ABORT"


def test_portfolio_cov_rc_hysteresis_sticky_and_fallback():
    d_prev_abort = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.31,
        rc_limit=0.30,
        hysteresis_band=0.03,
        previous_gate_decision="ABORT",
    )
    d_prev_allow = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.31,
        rc_limit=0.30,
        hysteresis_band=0.03,
        previous_gate_decision="ALLOW",
    )
    d_no_prev = paper_trading.resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.31,
        rc_limit=0.30,
        hysteresis_band=0.03,
        previous_gate_decision=None,
    )
    assert d_prev_abort["sticky_zone"] is True
    assert d_prev_allow["sticky_zone"] is True
    assert d_prev_abort["final_gate_decision"] == "ABORT"
    assert d_prev_allow["final_gate_decision"] == "ALLOW"
    assert d_no_prev["sticky_zone"] is True
    assert d_no_prev["fallback_used"] is True
    assert d_no_prev["final_gate_decision"] == "ABORT"


def _build_engine_with_cov(max_rc_fraction: float, prev_decision: str | None):
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.current_cycle = 999
    engine.config = {
        "strategy": {"lookback_days": 40},
        "execution": {
            "max_portfolio_volatility": 0.25,
            "portfolio_vol_min_coverage": 0.70,
            "enable_diversity_check": False,
            "enable_target_cov_gate": True,
            "target_cov_gate_min_coverage": 0.60,
            "target_cov_gate_require_ok": True,
            "cov_coverage_top_n": 20,
            "cov_coverage_max_list": 200,
        },
        "risk_model": {
            "use_cov_vol_for_gate": True,
            "rc_limit": 0.30,
            "portfolio_cov_rc_hysteresis_band": 0.03,
            "min_cov_gate_coverage": 0.60,
            "cov_gate_fallback_to_weighted": True,
        },
        "asset_data_policy": {
            "mode": "ALLOW_ORIGINAL",
            "match_rules": [{"suffix": ".TO"}],
            "proxy_map": {},
            "allow_execution_proxy": False,
            "allow_risk_proxy": False,
        },
    }
    engine.portfolio_snapshots = []
    engine.positions = {}
    engine.cash = 30000.0
    engine.current_asset_policy_decisions = []
    engine.current_asset_policy_summary = {
        "counts": {"ALLOW_ORIGINAL": 0, "USE_PROXY": 0, "DISABLE": 0},
        "top_reasons": [],
    }
    engine._last_cov_coverage_dump_meta = {}
    engine._last_portfolio_cov_rc_gate_decision = None
    engine.current_risk_check_info = {
        "risk_gate_decision": {
            "metric_name": "max_rc_fraction",
            "final_gate_decision": prev_decision,
            "reason": "portfolio_cov_rc_limit" if prev_decision == "ABORT" else "",
        }
    }
    engine._get_asset_volatility_optional = lambda *_args, **_kwargs: None
    engine.get_current_price = lambda *_args, **_kwargs: (100.0, 0.0, "LIVE")

    def _cov_diag_stub(_reason_tag, _weights_map, cycle_id=None):  # noqa: ARG001
        return {
            "enabled": True,
            "status": "ok",
            "returns_meta": {
                "overall_row_coverage": 1.0,
                "cols": 1,
                "used_tickers": ["SPY"],
                "missing_tickers": [],
                "dropped_tickers": [],
            },
            "portfolio_vol_annualized": 0.10,
            "max_rc_fraction": float(max_rc_fraction),
            "max_rc_ticker": "SPY",
            "avg_pairwise_corr": 0.1,
            "rc_fraction": {"SPY": float(max_rc_fraction)},
        }

    engine._compute_cov_diag_cached = _cov_diag_stub
    return engine


def test_portfolio_cov_rc_hysteresis_integration_sticky_uses_previous_decision():
    engine_abort = _build_engine_with_cov(max_rc_fraction=0.31, prev_decision="ABORT")
    result_abort = paper_trading.PaperTradingEngine._evaluate_portfolio_risk_gate(engine_abort, {"SPY": 0.4, "CASH": 0.6})
    decision_abort = result_abort.get("risk_gate_decision", {})
    assert bool(result_abort.get("abort")) is True
    assert str(result_abort.get("abort_reason", "")) == "portfolio_cov_rc_limit"
    assert decision_abort.get("sticky_zone") is True
    assert decision_abort.get("previous_gate_decision") == "ABORT"
    assert decision_abort.get("final_gate_decision") == "ABORT"

    engine_allow = _build_engine_with_cov(max_rc_fraction=0.31, prev_decision="ALLOW")
    result_allow = paper_trading.PaperTradingEngine._evaluate_portfolio_risk_gate(engine_allow, {"SPY": 0.4, "CASH": 0.6})
    decision_allow = result_allow.get("risk_gate_decision", {})
    assert bool(result_allow.get("abort")) is False
    assert str(result_allow.get("abort_reason", "")) == ""
    assert decision_allow.get("sticky_zone") is True
    assert decision_allow.get("previous_gate_decision") == "ALLOW"
    assert decision_allow.get("final_gate_decision") == "ALLOW"
