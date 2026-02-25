from paper_trading import _bind_debug_risk_gate, _build_debug_risk_gate_stub_payload


class _DummyEngine:
    pass


def test_dryrun_real_gate_flag_uses_real_gate_not_stub():
    calls = {"real": 0, "stub": 0}

    def _real_gate(_target_weights):
        calls["real"] += 1
        return {"abort": True, "abort_reason": "portfolio_cov_rc_limit", "risk_gate_stub_used": False}

    def _stub_gate(_target_weights):
        calls["stub"] += 1
        return _build_debug_risk_gate_stub_payload(
            target_weights={"AAA": 0.5, "CASH": 0.5},
            requested_abort_reason="portfolio_cov_rc_limit",
            allow_portfolio_cov_reason=False,
        )

    engine = _DummyEngine()
    engine._evaluate_portfolio_risk_gate = _real_gate
    mode = _bind_debug_risk_gate(engine, stub_fn=_stub_gate, dryrun_real_risk_gate=True)
    assert mode == "real"
    payload = engine._evaluate_portfolio_risk_gate({"AAA": 0.5, "CASH": 0.5})
    assert calls["real"] == 1
    assert calls["stub"] == 0
    assert bool(payload.get("risk_gate_stub_used", False)) is False


def test_stub_payload_default_does_not_fake_portfolio_cov_rc_limit():
    payload = _build_debug_risk_gate_stub_payload(
        target_weights={"AAA": 0.5, "CASH": 0.5},
        requested_abort_reason="portfolio_cov_rc_limit",
        allow_portfolio_cov_reason=False,
    )
    assert str(payload.get("abort_reason", "")) != "portfolio_cov_rc_limit"
    assert bool(payload.get("risk_gate_stub_used", False)) is True
