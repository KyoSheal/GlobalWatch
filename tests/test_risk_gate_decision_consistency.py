import math

from paper_trading import PaperTradingEngine


def test_risk_gate_decision_consistency():
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
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
            "rc_limit": 0.35,
            "min_cov_gate_coverage": 0.60,
            "cov_gate_fallback_to_weighted": True,
        },
    }
    engine.portfolio_snapshots = []
    engine.positions = {}
    engine.cash = 30000.0
    engine._get_asset_volatility_optional = lambda *_args, **_kwargs: None
    engine.get_current_price = lambda *_args, **_kwargs: (100.0, 0.0, "LIVE")

    def _cov_diag_stub(_reason_tag, weights_map, cycle_id=None):  # noqa: ARG001
        risky = [
            str(k).upper().strip()
            for k, v in (weights_map or {}).items()
            if str(k).upper().strip() != "CASH" and float(v or 0.0) > 0.0
        ]
        used = [t for t in ("SPY", "XLP") if t in risky]
        missing = [t for t in risky if t not in used]
        return {
            "enabled": True,
            "status": "ok",
            "returns_meta": {
                "overall_row_coverage": 1.0,
                "cols": max(1, len(used)),
                "used_tickers": used,
                "missing_tickers": missing,
                "dropped_tickers": [],
            },
            "portfolio_vol_annualized": 0.10,
            "max_rc_fraction": 0.92,
            "max_rc_ticker": "XIU.TO",
            "avg_pairwise_corr": 0.12,
            "rc_fraction": {"XIU.TO": 0.6, "SPY": 0.2, "XLP": 0.2},
        }

    engine._compute_cov_diag_cached = _cov_diag_stub

    target_weights = {"XIU.TO": 0.1107, "FTS.TO": 0.0932, "SPY": 0.1, "XLP": 0.1, "CASH": 0.5961}
    result = PaperTradingEngine._evaluate_portfolio_risk_gate(engine, target_weights)

    assert bool(result.get("abort")) is True
    assert str(result.get("abort_reason", "")) == "portfolio_cov_rc_limit"

    decision = result.get("risk_gate_decision", {})
    cov = result.get("cov_coverage", {})
    cov_dbg = result.get("cov_coverage_debug_inputs", {})

    assert isinstance(decision, dict)
    assert decision.get("reason") == "portfolio_cov_rc_limit"
    assert isinstance(cov_dbg, dict)
    assert cov_dbg.get("known_weight_gate_value") == decision.get("metric_value")

    metric_name = str(decision.get("metric_name", ""))
    metric_value = decision.get("metric_value")
    cov_known_weight = cov.get("known_weight")
    if metric_name == "known_weight":
        assert metric_value is not None and cov_known_weight is not None
        assert math.isclose(float(metric_value), float(cov_known_weight), rel_tol=0.0, abs_tol=1e-9)
    else:
        assert metric_name == "max_rc_fraction"
        assert str(decision.get("basis", "")) in {"target_weights", "target_weights_abs"}
        assert metric_value is not None and float(metric_value) > 0.35
