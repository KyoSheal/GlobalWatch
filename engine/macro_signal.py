"""MacroSignalAdapter — ChromaDB-backed macro signal integration.

This module lives here as the canonical source; paper_trading.py re-exports
MacroSignalAdapter for backward compatibility.
"""
from __future__ import annotations

from paper_trading import MacroSignalAdapter

__all__ = ["MacroSignalAdapter"]
