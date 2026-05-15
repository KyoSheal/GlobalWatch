"""Tests for P1-P3: vector retrieval interface, query mode, similarity weighting."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import paper_trading


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_engine():
    cfg = {
        "news_overlay": {
            "enabled": True,
            "alpha": 0.08,
            "max_abs_delta": 0.10,
            "min_confidence": 0.30,
            "max_age_hours": 48.0,
            "decay_lambda_per_hour": 0.04,
            "industry_collection": "industry_signals",
            "industry_stale_warn_hours": 2.0,
            "mode": "risk_only",
            "enable_confidence_scaling": True,
            "semantic_query_n_results": 20,
        },
        "macro_integration": {"chroma_path": "./memory_db"},
    }
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.config = cfg
    engine.logger = MagicMock()
    return engine


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _make_meta(l2: str, confidence: float = 0.8, risk_delta: float = 0.2,
               hours_ago: float = 1.0) -> dict:
    return {
        "L2": l2,
        "confidence": confidence,
        "risk_delta": risk_delta,
        "timestamp": _iso(hours_ago),
        "scope": "industry",
        "direction": "bullish",
        "horizon": "1d",
    }


# ── P1: tickers parameter forwarded ──────────────────────────────────────────

def test_p1_tickers_forwarded_to_read(tmp_path):
    """apply_news_overlay_to_cash_target passes tickers to _read_recent_industry_signals."""
    engine = _make_engine()
    captured = {}

    def _mock_read(tickers=None):
        captured["tickers"] = tickers
        return []

    engine._read_recent_industry_signals = _mock_read
    tickers = ["QQQ", "SMH", "NVDA"]
    engine.apply_news_overlay_to_cash_target(tickers, 100_000.0)
    assert captured.get("tickers") == tickers


# ── P2: query mode vs fallback ────────────────────────────────────────────────

def test_p2_query_mode_used_when_tickers_provided(tmp_path):
    """When tickers provided, collection.query() is called instead of collection.get()."""
    engine = _make_engine()
    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "metadatas": [[_make_meta("Technology")]],
        "documents": [["{}"]],
        "distances": [[0.3]],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=["QQQ", "SMH"])

    mock_coll.query.assert_called_once()
    mock_coll.get.assert_not_called()
    assert len(rows) == 1
    assert rows[0]["L2"] == "Technology"


def test_p2_fallback_to_get_when_no_tickers(tmp_path):
    """When tickers is empty/None, collection.get() is called."""
    engine = _make_engine()
    mock_coll = MagicMock()
    mock_coll.get.return_value = {
        "ids": ["id1"],
        "metadatas": [_make_meta("Technology")],
        "documents": ["{}"],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=None)

    mock_coll.get.assert_called_once()
    mock_coll.query.assert_not_called()


def test_p2_distance_field_in_returned_rows(tmp_path):
    """Rows returned from query mode include a 'distance' field."""
    engine = _make_engine()
    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "ids": [["id1", "id2"]],
        "metadatas": [[_make_meta("Technology"), _make_meta("Energy")]],
        "documents": [["{}","{}"]],
        "distances": [[0.1, 0.9]],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=["QQQ"])

    by_l2 = {r["L2"]: r for r in rows}
    assert "distance" in by_l2["Technology"]
    assert "distance" in by_l2["Energy"]
    assert abs(by_l2["Technology"]["distance"] - 0.1) < 1e-9
    assert abs(by_l2["Energy"]["distance"] - 0.9) < 1e-9


# ── P3: similarity weighting ──────────────────────────────────────────────────

def test_p3_similarity_weight_reduces_far_signal():
    """A signal with high distance (far from holdings) gets lower effective weight."""
    engine = _make_engine()

    # distance=0.0 → similarity=1.0 → effective_weight = decay × 1.0
    # distance=2.0 → similarity=0.0 → effective_weight = decay × 0.5
    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "ids": [["id_near", "id_far"]],
        "metadatas": [[
            _make_meta("Technology", confidence=1.0, risk_delta=1.0, hours_ago=0.0),
            _make_meta("Energy",     confidence=1.0, risk_delta=1.0, hours_ago=0.0),
        ]],
        "documents": [["{}","{}"]],
        "distances": [[0.0, 2.0]],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=["QQQ"])

    by_l2 = {r["L2"]: r for r in rows}
    near_conf = by_l2["Technology"]["confidence"]
    far_conf  = by_l2["Energy"]["confidence"]
    # near (dist=0): effective_weight = decay(0h) × (0.5 + 0.5×1.0) = 1.0 × 1.0 = 1.0
    # far  (dist=2): effective_weight = decay(0h) × (0.5 + 0.5×0.0) = 1.0 × 0.5 = 0.5
    assert near_conf > far_conf
    assert abs(near_conf / far_conf - 2.0) < 1e-6


def test_p3_no_distance_in_get_mode_uses_full_weight():
    """In get() fallback mode (no distances), signal uses full decay weight."""
    engine = _make_engine()
    mock_coll = MagicMock()
    mock_coll.get.return_value = {
        "ids": ["id1"],
        "metadatas": [_make_meta("Technology", confidence=1.0, risk_delta=1.0, hours_ago=0.0)],
        "documents": ["{}"],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=None)

    assert len(rows) == 1
    # No distances → similarity=1.0 → effective_weight = decay × 1.0 = 1.0
    row = rows[0]
    assert abs(row["confidence"] - 1.0) < 1e-6
    assert abs(row["raw_fields_snapshot"]["similarity"] - 1.0) < 1e-6


def test_p3_effective_weight_in_snapshot():
    """raw_fields_snapshot exposes similarity and effective_weight fields."""
    engine = _make_engine()
    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "metadatas": [[_make_meta("Technology", confidence=0.8, risk_delta=0.3, hours_ago=2.0)]],
        "documents": [["{}"]],
        "distances": [[0.6]],
    }

    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        rows = engine._read_recent_industry_signals(tickers=["QQQ"])

    snap = rows[0]["raw_fields_snapshot"]
    assert "similarity" in snap
    assert "effective_weight" in snap
    # distance=0.6 → similarity = max(0, 1 - 0.6/2) = 0.7
    assert abs(snap["similarity"] - 0.7) < 1e-6
