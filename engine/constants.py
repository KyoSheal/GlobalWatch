"""Engine constants — schema versions, risk profile definitions.

These constants live here as the canonical source; paper_trading.py re-exports
them for backward compatibility.
"""
from __future__ import annotations

from paper_trading import (
    LIVE_SCHEMA_VERSION,
    RISK_PROFILE_TEMPLATE_VERSION,
    EFFECTIVE_RISK_MODEL_CONFIG_SCHEMA_VERSION,
    DEFAULT_RISK_PROFILES,
    RISK_PROFILE_ALLOWED_KEYS,
    RISK_PROFILE_CHOICES,
    RISK_PROFILE_DEFAULT,
)

__all__ = [
    "LIVE_SCHEMA_VERSION",
    "RISK_PROFILE_TEMPLATE_VERSION",
    "EFFECTIVE_RISK_MODEL_CONFIG_SCHEMA_VERSION",
    "DEFAULT_RISK_PROFILES",
    "RISK_PROFILE_ALLOWED_KEYS",
    "RISK_PROFILE_CHOICES",
    "RISK_PROFILE_DEFAULT",
]
