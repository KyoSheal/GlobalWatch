from __future__ import annotations

from cost_model import compute_trade_cost


def test_compute_trade_cost_buy_direction_and_components():
    out = compute_trade_cost(
        side="BUY",
        qty=10,
        price=100.0,
        notional=1000.0,
        slippage_bps=10.0,
        fee_per_trade=1.0,
        fee_bps=5.0,
        min_fee=0.0,
    )
    assert out["side"] == "BUY"
    assert abs(float(out["effective_price"]) - 100.1) < 1e-9
    assert abs(float(out["slippage_cost"]) - 1.0) < 1e-9
    assert abs(float(out["fee_cost"]) - 1.5) < 1e-9
    assert abs(float(out["total_cost"]) - 2.5) < 1e-9


def test_compute_trade_cost_sell_direction_and_components():
    out = compute_trade_cost(
        side="SELL",
        qty=10,
        price=100.0,
        notional=1000.0,
        slippage_bps=10.0,
        fee_per_trade=1.0,
        fee_bps=5.0,
        min_fee=0.0,
    )
    assert out["side"] == "SELL"
    assert abs(float(out["effective_price"]) - 99.9) < 1e-9
    assert abs(float(out["slippage_cost"]) - 1.0) < 1e-9
    assert abs(float(out["fee_cost"]) - 1.5) < 1e-9
    assert abs(float(out["total_cost"]) - 2.5) < 1e-9


def test_compute_trade_cost_min_fee_applies():
    out = compute_trade_cost(
        side="BUY",
        qty=1,
        price=10.0,
        notional=10.0,
        slippage_bps=0.0,
        fee_per_trade=0.0,
        fee_bps=0.0,
        min_fee=2.0,
    )
    assert abs(float(out["slippage_cost"]) - 0.0) < 1e-9
    assert abs(float(out["fee_cost"]) - 2.0) < 1e-9
    assert abs(float(out["total_cost"]) - 2.0) < 1e-9
