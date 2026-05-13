"""Engine package — paper trading simulation engine.

Public API:

    from engine import PaperTradingEngine
    from engine.constants import LIVE_SCHEMA_VERSION, DEFAULT_RISK_PROFILES
    from engine.macro_signal import MacroSignalAdapter
    from engine.utils import _normalize_cov_rc_gate_decision

Backward-compatible imports continue to work from paper_trading directly.
"""
from __future__ import annotations

from paper_trading import (
    PaperTradingEngine,
    MacroSignalAdapter,
    LIVE_SCHEMA_VERSION,
    RISK_PROFILE_TEMPLATE_VERSION,
    EFFECTIVE_RISK_MODEL_CONFIG_SCHEMA_VERSION,
    DEFAULT_RISK_PROFILES,
    RISK_PROFILE_ALLOWED_KEYS,
    RISK_PROFILE_CHOICES,
    RISK_PROFILE_DEFAULT,
    _normalize_cov_rc_gate_decision,
    resolve_portfolio_cov_rc_hysteresis_decision,
    resolve_portfolio_cov_rc_abort_buffer_decision,
)

__all__ = [
    "PaperTradingEngine",
    "MacroSignalAdapter",
    "LIVE_SCHEMA_VERSION",
    "RISK_PROFILE_TEMPLATE_VERSION",
    "EFFECTIVE_RISK_MODEL_CONFIG_SCHEMA_VERSION",
    "DEFAULT_RISK_PROFILES",
    "RISK_PROFILE_ALLOWED_KEYS",
    "RISK_PROFILE_CHOICES",
    "RISK_PROFILE_DEFAULT",
    "_normalize_cov_rc_gate_decision",
    "resolve_portfolio_cov_rc_hysteresis_decision",
    "resolve_portfolio_cov_rc_abort_buffer_decision",
]
