from paper_trading import _build_debug_risk_gate_stub_payload


def test_risk_gate_stub_marking_default_no_portfolio_cov_reason():
    payload = _build_debug_risk_gate_stub_payload(
        target_weights={"AAA": 0.5, "CASH": 0.5},
        requested_abort_reason="portfolio_cov_rc_limit",
        allow_portfolio_cov_reason=False,
    )

    assert bool(payload.get("risk_gate_stub_used")) is True
    assert str(payload.get("risk_gate_stub_name")) == "_risk_gate_stub"
    assert str(payload.get("abort_reason", "")) != "portfolio_cov_rc_limit"

    debug_inputs = payload.get("cov_coverage_debug_inputs", {})
    assert isinstance(debug_inputs, dict)
    assert bool(debug_inputs.get("stub_used")) is True
    assert "stubbed gate path" in str(debug_inputs.get("stub_reason", ""))
