"""Simple deterministic trade-cost model helpers."""

from __future__ import annotations

from typing import Any, Dict


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def compute_trade_cost(
    side: str,
    qty: float,
    price: float,
    notional: float,
    slippage_bps: float,
    fee_per_trade: float,
    fee_bps: float,
    min_fee: float,
) -> Dict[str, Any]:
    """
    Compute simple fee+slippage cost breakdown.

    Slippage convention:
    - BUY effective_price = price * (1 + slip)
    - SELL effective_price = price * (1 - slip)
    Cost is always returned as positive USD dollars.
    """
    side_u = str(side or "").strip().upper() or "BUY"
    qty_f = abs(_to_float(qty, 0.0))
    price_f = _to_float(price, 0.0)
    notional_f = abs(_to_float(notional, 0.0))
    slip_bps_f = max(0.0, _to_float(slippage_bps, 0.0))
    fee_trade_f = max(0.0, _to_float(fee_per_trade, 0.0))
    fee_bps_f = max(0.0, _to_float(fee_bps, 0.0))
    min_fee_f = max(0.0, _to_float(min_fee, 0.0))

    slip_rate = slip_bps_f / 10000.0
    fee_bps_rate = fee_bps_f / 10000.0
    slippage_cost = notional_f * slip_rate
    fee_cost = fee_trade_f + (notional_f * fee_bps_rate)
    if fee_cost < min_fee_f:
        fee_cost = min_fee_f
    total_cost = slippage_cost + fee_cost

    if side_u == "SELL":
        effective_price = price_f * (1.0 - slip_rate)
    else:
        effective_price = price_f * (1.0 + slip_rate)

    return {
        "side": side_u,
        "qty": float(qty_f),
        "price": float(price_f),
        "notional": float(notional_f),
        "slippage_bps": float(slip_bps_f),
        "fee_per_trade": float(fee_trade_f),
        "fee_bps": float(fee_bps_f),
        "min_fee": float(min_fee_f),
        "slippage_cost": float(max(0.0, slippage_cost)),
        "fee_cost": float(max(0.0, fee_cost)),
        "total_cost": float(max(0.0, total_cost)),
        "effective_price": float(effective_price),
    }

