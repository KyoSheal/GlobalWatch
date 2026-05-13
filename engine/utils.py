"""Engine utility functions — standalone covariance / risk-gate helpers.

These functions live here as the canonical source; paper_trading.py re-exports
them for backward compatibility.
"""
from __future__ import annotations

from paper_trading import (
    _normalize_cov_rc_gate_decision,
    resolve_portfolio_cov_rc_hysteresis_decision,
    resolve_portfolio_cov_rc_abort_buffer_decision,
)

__all__ = [
    "_normalize_cov_rc_gate_decision",
    "resolve_portfolio_cov_rc_hysteresis_decision",
    "resolve_portfolio_cov_rc_abort_buffer_decision",
]
