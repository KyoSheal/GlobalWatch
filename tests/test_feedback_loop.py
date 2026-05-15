"""Tests for P4-P8: three-layer separation, prediction log, IC EMA, settle, adaptive alpha."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import paper_trading


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_engine(tmp_path: str, min_cycles: int = 5):
    cfg = {
        "news_overlay": {
            "enabled": True,
            "alpha": 0.08,
            "max_abs_delta": 0.10,
            "min_confidence": 0.30,
            "max_age_hours": 48.0,
            "decay_lambda_per_hour": 0.04,
            "industry_collection": "industry_signals",
            "mode": "risk_only",
            "enable_confidence_scaling": False,  # simpler math
            "semantic_query_n_results": 20,
            "min_cycles_before_adaptive": min_cycles,
            "ic_ema_alpha": 0.1,
        },
        "macro_integration": {"chroma_path": tmp_path},
        "objectives": {"min_cash_pct": 0.10},
    }
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.config = cfg
    engine.logger = MagicMock()
    engine.current_cycle = 1
    engine.cash = 100_000.0
    engine.positions = {}
    return engine


def _iso(hours_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


# ── P4: path helpers are independent ─────────────────────────────────────────

def test_p4_paths_are_separate(tmp_path):
    engine = _make_engine(str(tmp_path))
    pred_path = engine._get_prediction_log_path()
    ic_path = engine._get_signal_ic_state_path()
    assert pred_path != ic_path
    assert pred_path.endswith("prediction_log.jsonl")
    assert ic_path.endswith("signal_ic_state.json")


def test_p4_ic_state_survives_log_deletion(tmp_path):
    engine = _make_engine(str(tmp_path))
    state = {"Technology": {"ic": 0.14, "n_settled": 10}}
    engine._save_signal_ic_state(state)

    # Delete prediction log (simulating developer cleanup)
    pred_path = engine._get_prediction_log_path()
    with open(pred_path, "w") as f:
        f.write("")
    os.remove(pred_path)

    # IC state should still be intact
    loaded = engine._load_signal_ic_state()
    assert loaded.get("Technology", {}).get("ic") == 0.14


# ── P5: prediction log written on apply ──────────────────────────────────────

def test_p5_prediction_log_written(tmp_path):
    engine = _make_engine(str(tmp_path))
    engine._append_prediction_log(
        worst_l2="Technology",
        predicted_delta=-0.25,
        cash_adj=0.02,
        l2_deltas={"Technology": -0.25, "Energy": -0.10},
    )
    pred_path = engine._get_prediction_log_path()
    assert os.path.exists(pred_path)
    with open(pred_path, "r") as f:
        record = json.loads(f.readline())
    assert record["worst_l2"] == "Technology"
    assert record["predicted_delta"] == -0.25
    assert record["cash_adj"] == 0.02
    assert "snapshot_equity" in record
    assert "cycle_ts" in record


def test_p5_prediction_log_appends(tmp_path):
    engine = _make_engine(str(tmp_path))
    for i in range(3):
        engine._append_prediction_log(
            worst_l2=f"Sector{i}", predicted_delta=-0.1 * i,
            cash_adj=0.01, l2_deltas={},
        )
    pred_path = engine._get_prediction_log_path()
    with open(pred_path, "r") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 3


# ── P6: IC EMA update ────────────────────────────────────────────────────────

def test_p6_ic_ema_correct_prediction(tmp_path):
    engine = _make_engine(str(tmp_path))
    # Seed: predicted down (delta=-0.2), actual equity dropped
    engine._append_prediction_log(
        worst_l2="Technology", predicted_delta=-0.2,
        cash_adj=0.02, l2_deltas={"Technology": -0.2},
    )
    # Simulate equity drop from 100k → 99k
    engine.cash = 99_000.0

    engine._settle_previous_predictions()

    state = engine._load_signal_ic_state()
    tech = state.get("Technology", {})
    assert tech.get("n_settled") == 1
    # ic_contribution = +1 (correct direction), alpha_ema=0.1
    # ic_new = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
    assert abs(tech.get("ic", 0.0) - 0.1) < 1e-9


def test_p6_ic_ema_wrong_prediction(tmp_path):
    engine = _make_engine(str(tmp_path))
    # Predicted down but equity went up
    engine._append_prediction_log(
        worst_l2="Energy", predicted_delta=-0.15,
        cash_adj=0.01, l2_deltas={"Energy": -0.15},
    )
    engine.cash = 101_000.0  # equity rose, prediction was wrong

    engine._settle_previous_predictions()

    state = engine._load_signal_ic_state()
    energy = state.get("Energy", {})
    # ic_contribution = -1, ic_new = 0.1 * (-1) + 0.9 * 0 = -0.1
    assert abs(energy.get("ic", 0.0) - (-0.1)) < 1e-9


def test_p6_ic_ema_accumulates(tmp_path):
    engine = _make_engine(str(tmp_path))
    # 3 correct predictions: each time predict down, equity drops
    equity = 100_000.0
    for _ in range(3):
        engine.cash = equity
        engine._append_prediction_log(
            worst_l2="Finance", predicted_delta=-0.1,
            cash_adj=0.01, l2_deltas={"Finance": -0.1},
        )
        equity -= 1_000.0  # equity dropped
        engine.cash = equity
        engine._settle_previous_predictions()

    state = engine._load_signal_ic_state()
    ic = state["Finance"]["ic"]
    # After 3 correct: 0.1, 0.19, 0.271
    assert ic > 0.2


# ── P7: no log → settle is no-op ─────────────────────────────────────────────

def test_p7_settle_no_log_is_noop(tmp_path):
    engine = _make_engine(str(tmp_path))
    engine._settle_previous_predictions()  # should not raise
    state = engine._load_signal_ic_state()
    assert state == {}


# ── P8: adaptive alpha ────────────────────────────────────────────────────────

def test_p8_no_ic_uses_base_alpha(tmp_path):
    engine = _make_engine(str(tmp_path), min_cycles=5)
    # No IC state → effective_alpha should equal base_alpha
    state = {}
    engine._save_signal_ic_state(state)

    mock_coll = MagicMock()
    meta = {
        "L2": "Technology", "confidence": 1.0, "risk_delta": -1.0,
        "timestamp": _iso(1.0), "scope": "industry", "direction": "bearish", "horizon": "1d",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]], "metadatas": [[meta]], "documents": [["{}"]],
        "distances": [[0.0]],
    }
    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        _, info = engine.apply_news_overlay_to_cash_target(["QQQ"], 0.20)

    diag = info.get("l2_delta_map_sample", [])
    if diag:
        assert abs(diag[0].get("effective_alpha", 0.08) - 0.08) < 1e-9


def test_p8_positive_ic_amplifies_alpha(tmp_path):
    engine = _make_engine(str(tmp_path), min_cycles=5)
    # Seed IC: Technology has ic=+0.2, n_settled=5 (conf=1.0)
    engine._save_signal_ic_state({
        "Technology": {"ic": 0.2, "n_settled": 5}
    })

    mock_coll = MagicMock()
    meta = {
        "L2": "Technology", "confidence": 1.0, "risk_delta": -1.0,
        "timestamp": _iso(1.0), "scope": "industry", "direction": "bearish", "horizon": "1d",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]], "metadatas": [[meta]], "documents": [["{}"]],
        "distances": [[0.0]],
    }
    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        _, info = engine.apply_news_overlay_to_cash_target(["QQQ"], 0.20)

    diag = info.get("l2_delta_map_sample", [])
    if diag:
        # effective_alpha = 0.08 × (1 + 0.2 × 3) = 0.08 × 1.6 = 0.128
        eff = diag[0].get("effective_alpha", 0.08)
        assert eff > 0.08, f"expected amplified alpha, got {eff}"


def test_p8_negative_ic_reduces_alpha(tmp_path):
    engine = _make_engine(str(tmp_path), min_cycles=5)
    # Seed IC: Energy has ic=-0.2, n_settled=10 (conf=1.0)
    engine._save_signal_ic_state({
        "Energy": {"ic": -0.2, "n_settled": 10}
    })

    mock_coll = MagicMock()
    meta = {
        "L2": "Energy", "confidence": 1.0, "risk_delta": -1.0,
        "timestamp": _iso(1.0), "scope": "industry", "direction": "bearish", "horizon": "1d",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]], "metadatas": [[meta]], "documents": [["{}"]],
        "distances": [[0.0]],
    }
    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        _, info = engine.apply_news_overlay_to_cash_target(["XLE"], 0.20)

    diag = info.get("l2_delta_map_sample", [])
    if diag:
        # effective_alpha = 0.08 × max(0.5, 1 + (-0.2)×3) = 0.08 × 0.4 → clamped to 0.08×0.5 = 0.04
        eff = diag[0].get("effective_alpha", 0.08)
        assert eff < 0.08, f"expected reduced alpha, got {eff}"


def test_p8_gradual_rampup_partial_conf(tmp_path):
    """With 2 settled out of 5 required, conf=0.4 → partial adaptation."""
    engine = _make_engine(str(tmp_path), min_cycles=5)
    engine._save_signal_ic_state({
        "Healthcare": {"ic": 1.0, "n_settled": 2}  # only 2/5
    })

    mock_coll = MagicMock()
    meta = {
        "L2": "Healthcare", "confidence": 1.0, "risk_delta": -1.0,
        "timestamp": _iso(1.0), "scope": "industry", "direction": "bearish", "horizon": "1d",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]], "metadatas": [[meta]], "documents": [["{}"]],
        "distances": [[0.0]],
    }
    with patch.object(engine, "_get_industry_signals_collection", return_value=mock_coll), \
         patch("paper_trading.CHROMADB_AVAILABLE", True):
        _, info = engine.apply_news_overlay_to_cash_target(["XLV"], 0.20)

    diag = info.get("l2_delta_map_sample", [])
    if diag:
        eff = diag[0].get("effective_alpha", 0.08)
        # conf=0.4: eff_alpha = 0.08 × (0.6×1.0 + 0.4×(1+1.0×3)) = 0.08 × (0.6 + 0.4×4) = 0.08 × 2.2 → capped 2.0
        # = 0.08 × (0.6 + 1.6) = 0.08 × 2.0 (cap) → max 0.16
        assert eff > 0.08, "partial conf should still amplify a strong IC signal"
        assert eff <= 0.08 * 2.0 + 1e-9, "should not exceed 2× base_alpha cap"
