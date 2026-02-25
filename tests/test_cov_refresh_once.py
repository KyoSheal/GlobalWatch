from paper_trading import PaperTradingEngine


def test_cov_refresh_attempted_once_per_cycle():
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine.current_cycle = 99
    engine.config = {"execution": {"cov_refresh_on_rc_limit": True}}
    engine.returns_cache = {
        ("cov_diag_current", 99, (("AAA", 0.5),)): {"meta": {}},
        ("cov_diag_target", 99, (("AAA", 0.5),)): {"meta": {}},
    }
    engine._cov_refresh_attempted_cycle = None
    engine.current_cov_refresh_info = {}

    calls = {"n": 0}

    def _stub_eval(_target_weights):
        calls["n"] += 1
        return {"abort": True, "abort_reason": "portfolio_cov_rc_limit"}

    engine._evaluate_portfolio_risk_gate = _stub_eval

    out1 = PaperTradingEngine._attempt_cov_refresh_once(
        engine,
        target_weights={"AAA": 0.5, "CASH": 0.5},
        reason="portfolio_cov_rc_limit",
    )
    out2 = PaperTradingEngine._attempt_cov_refresh_once(
        engine,
        target_weights={"AAA": 0.5, "CASH": 0.5},
        reason="portfolio_cov_rc_limit",
    )

    assert bool(out1.get("attempted")) is True
    assert str(out1.get("status", "")) in {"ok", "error"}
    assert bool(out2.get("attempted")) is False
    assert str(out2.get("status", "")) == "already_attempted_this_cycle"
    assert int(calls["n"]) == 1

