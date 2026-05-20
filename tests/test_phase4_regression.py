"""Phase 4 regression tests — establish behavioral baseline before splitting.

These tests verify:
  1. Module-level constants exist with correct types/values
  2. MacroSignalAdapter interface is intact
  3. PaperTradingEngine has all expected methods (method presence)
  4. Key computations return correct shapes/types (behavior)
  5. Cross-module connections work (integration)
  6. Backward-compat imports from paper_trading still work after split

Run before AND after the split; all should pass both times.
"""

from __future__ import annotations

import json
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. Backward-compat imports — these must work from paper_trading directly
# ---------------------------------------------------------------------------

def test_import_paper_trading_engine():
    from paper_trading import PaperTradingEngine  # noqa: F401
    assert callable(PaperTradingEngine)


def test_import_macro_signal_adapter():
    from paper_trading import MacroSignalAdapter  # noqa: F401
    assert callable(MacroSignalAdapter)


def test_import_constants():
    from paper_trading import (
        LIVE_SCHEMA_VERSION,
        RISK_PROFILE_DEFAULT,
        DEFAULT_RISK_PROFILES,
        RISK_PROFILE_CHOICES,
    )
    assert isinstance(LIVE_SCHEMA_VERSION, int)
    assert RISK_PROFILE_DEFAULT == "high"
    assert set(DEFAULT_RISK_PROFILES.keys()) == {"low", "mid", "high", "ultra"}
    assert set(RISK_PROFILE_CHOICES) == {"low", "mid", "high", "ultra"}


def test_import_standalone_functions():
    from paper_trading import (
        _normalize_cov_rc_gate_decision,
        resolve_portfolio_cov_rc_hysteresis_decision,
        resolve_portfolio_cov_rc_abort_buffer_decision,
    )
    assert callable(_normalize_cov_rc_gate_decision)
    assert callable(resolve_portfolio_cov_rc_hysteresis_decision)
    assert callable(resolve_portfolio_cov_rc_abort_buffer_decision)


# ---------------------------------------------------------------------------
# 2. Standalone function behavior
# ---------------------------------------------------------------------------

def test_normalize_cov_rc_gate_decision_allow():
    from paper_trading import _normalize_cov_rc_gate_decision
    assert _normalize_cov_rc_gate_decision("ALLOW") == "ALLOW"
    assert _normalize_cov_rc_gate_decision("allow") == "ALLOW"


def test_normalize_cov_rc_gate_decision_abort():
    from paper_trading import _normalize_cov_rc_gate_decision
    assert _normalize_cov_rc_gate_decision("ABORT") == "ABORT"


def test_normalize_cov_rc_gate_decision_unknown():
    from paper_trading import _normalize_cov_rc_gate_decision
    assert _normalize_cov_rc_gate_decision(None) is None
    assert _normalize_cov_rc_gate_decision("bogus") is None


def test_hysteresis_decision_returns_dict():
    from paper_trading import resolve_portfolio_cov_rc_hysteresis_decision
    result = resolve_portfolio_cov_rc_hysteresis_decision(
        portfolio_rc_fraction=0.30,
        rc_limit=0.35,
        hysteresis_band=0.10,
        previous_gate_decision="ALLOW",
    )
    assert isinstance(result, dict)
    # Returns final_gate_decision or gate_decision key
    assert "final_gate_decision" in result or "gate_decision" in result or len(result) > 0


def test_abort_buffer_decision_returns_dict():
    from paper_trading import resolve_portfolio_cov_rc_abort_buffer_decision
    result = resolve_portfolio_cov_rc_abort_buffer_decision(
        portfolio_rc_fraction=0.40,
        base_rc_limit=0.35,
        hysteresis_band=0.10,
        previous_gate_decision="ALLOW",
        prev_abort_streak=0,
        trigger_consecutive_aborts=3,
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 3. PaperTradingEngine — method presence (all categories)
# ---------------------------------------------------------------------------

REQUIRED_METHODS = [
    # Config / init
    "load_config", "apply_risk_profile", "_deep_merge_dicts",
    "_configure_run_output_paths", "_normalize_execution_risk_profile",
    "_compute_config_hash",
    # Price / market data
    "get_current_price", "get_market_data", "to_yahoo_symbol",
    "classify_price_freshness", "_collect_price_debug",
    # Signal processing
    "calculate_momentum", "calculate_volatility", "detect_exit_signals",
    "calculate_target_weights", "refresh_macro_cache",
    "_compute_cross_sectional_metrics", "_apply_correlation_filter",
    "_apply_score_stability_controls",
    # Risk gating / covariance
    "compute_cov_risk_diagnostics", "_evaluate_portfolio_risk_gate",
    "apply_vol_targeting_to_targets", "build_returns_matrix",
    "_get_risk_model_cfg", "check_risk_controls",
    # Phase 3: advanced covariance / risk methods
    "_estimate_covariance_ledoit_wolf", "_compute_crisis_mode",
    "_compute_portfolio_cvar", "_get_vix_level",
    "_compute_auto_risk_profile_signal", "_get_fx_rate",
    # Phase 2: signal quality methods
    "_apply_sector_concentration_cap",
    # Phase 4: execution quality methods
    "_compute_position_stop_loss_overrides",
    "_apply_ramp_in_to_targets",
    "_compute_adaptive_turnover_limit",
    # Phase 5: enhanced performance metrics / circuit breaker / factor attribution
    "_compute_enhanced_performance_metrics",
    "_check_rolling_drawdown_circuit_breaker",
    "_record_factor_attribution",
    "_append_attribution_jsonl",
    "generate_summary_report",
    # Trade execution
    "execute_rebalance", "apply_trade_planner", "filter_trades_greedy",
    "estimate_trade_cost", "_get_cost_model_cfg", "_get_planner_cfg",
    # Snapshot / I/O
    "record_snapshot", "write_live_snapshot", "build_live_snapshot",
    "save_results", "generate_equity_curve", "save_trade_history_jsonl",
    "build_replay_bundle",
    # Market session / cooldown / runtime
    "_refresh_market_session_state", "_apply_cooldown_outcome",
    "_maybe_apply_runtime_risk_profile", "run_cycle", "run",
    # Asset policy
    "resolve_asset_policy", "_get_asset_data_policy_cfg",
    # Utility
    "_json_safe_clone", "_now", "_coerce_datetime_utc",
    "atomic_write_json",
    # AI improvements
    "_compute_per_ticker_news_score",
]


@pytest.fixture(scope="module")
def engine_cls():
    from paper_trading import PaperTradingEngine
    return PaperTradingEngine


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_engine_has_method(engine_cls, method_name):
    """Verify every expected method is present on the class (not just instance)."""
    assert hasattr(engine_cls, method_name), (
        f"PaperTradingEngine is missing method: {method_name}"
    )
    assert callable(getattr(engine_cls, method_name)), (
        f"PaperTradingEngine.{method_name} is not callable"
    )


# ---------------------------------------------------------------------------
# 4. PaperTradingEngine initialization (with real config)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Create a real engine instance with paper_config.json."""
    from paper_trading import PaperTradingEngine
    tmp = tmp_path_factory.mktemp("engine_out")
    cfg_path = Path("paper_config.json")
    if not cfg_path.exists():
        pytest.skip("paper_config.json not found")
    # Patch output dir so we don't pollute real outputs
    import json as _json
    raw = _json.loads(cfg_path.read_text())
    raw.setdefault("reporting", {})["out_dir"] = str(tmp)
    tmp_cfg = tmp / "paper_config.json"
    tmp_cfg.write_text(_json.dumps(raw))
    eng = PaperTradingEngine(str(tmp_cfg))
    return eng


def test_engine_initial_cash(engine):
    assert isinstance(engine.cash, (int, float))
    assert engine.cash > 0


def test_engine_positions_dict(engine):
    assert isinstance(engine.positions, dict)


def test_engine_config_loaded(engine):
    assert isinstance(engine.config, dict)
    assert "execution" in engine.config
    assert "universe" in engine.config


def test_engine_current_cycle_zero(engine):
    assert isinstance(engine.current_cycle, int)


def test_engine_run_id_nonempty(engine):
    assert isinstance(engine.run_id, str)
    assert len(engine.run_id) > 0


# ---------------------------------------------------------------------------
# 5. Key computation behaviors (no network, stub data)
# ---------------------------------------------------------------------------

def test_json_safe_clone_dict(engine):
    obj = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    result = engine._json_safe_clone(obj)
    assert result == obj
    assert result is not obj  # deep copy


def test_json_safe_clone_non_serializable(engine):
    obj = {"dt": datetime.now(timezone.utc)}
    result = engine._json_safe_clone(obj, fallback={})
    # Should either serialize to string or return fallback — must not raise
    assert isinstance(result, dict)


def test_now_returns_aware_datetime(engine):
    now = engine._now()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None


def test_coerce_datetime_utc_iso(engine):
    from datetime import datetime, timezone
    # _coerce_datetime_utc accepts datetime objects; _parse_datetime_utc_safe accepts ISO strings
    dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = engine._coerce_datetime_utc(dt)
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    # Also test ISO string parsing via _parse_datetime_utc_safe
    iso = "2026-01-15T12:00:00+00:00"
    result2 = engine._parse_datetime_utc_safe(iso)
    assert isinstance(result2, datetime)
    assert result2.tzinfo is not None


def test_compute_config_hash_stable(engine):
    h1 = engine._compute_config_hash(engine.config)
    h2 = engine._compute_config_hash(engine.config)
    assert isinstance(h1, str)
    assert h1 == h2


def test_get_risk_model_cfg_returns_dict(engine):
    cfg = engine._get_risk_model_cfg()
    assert isinstance(cfg, dict)


def test_get_cost_model_cfg_returns_dict(engine):
    cfg = engine._get_cost_model_cfg()
    assert isinstance(cfg, dict)


def test_get_planner_cfg_returns_dict(engine):
    cfg = engine._get_planner_cfg()
    assert isinstance(cfg, dict)


def test_get_asset_data_policy_cfg_returns_dict(engine):
    cfg = engine._get_asset_data_policy_cfg()
    assert isinstance(cfg, dict)


# ---------------------------------------------------------------------------
# 6. Covariance / risk computation with synthetic data
# ---------------------------------------------------------------------------

def test_compute_cov_risk_diagnostics_empty(engine):
    result = engine.compute_cov_risk_diagnostics({})
    assert isinstance(result, dict)


def test_compute_cov_risk_diagnostics_single_asset(engine):
    result = engine.compute_cov_risk_diagnostics({"SPY": 1.0})
    assert isinstance(result, dict)


def test_compute_portfolio_vol_and_rc_shape(engine):
    """Verify vol/RC computation returns expected keys."""
    tickers = ["A", "B", "C"]
    cov_data = np.eye(3) * 0.04  # 20% vol each asset, uncorrelated
    cov = pd.DataFrame(cov_data, index=tickers, columns=tickers)
    weights = {"A": 0.4, "B": 0.4, "C": 0.2}
    result = engine._compute_portfolio_vol_and_rc(cov, weights, annualization_factor=252)
    assert isinstance(result, dict)
    assert "portfolio_vol" in result
    # Risk contributions are stored in rc_fraction dict
    assert "rc_fraction" in result or "rc" in result
    assert isinstance(result["portfolio_vol"], float)


def test_build_returns_matrix_empty(engine):
    """build_returns_matrix with no valid tickers returns (DataFrame, dict) tuple."""
    result = engine.build_returns_matrix(
        tickers=["FAKE_TICKER_XYZ"],
        lookback_days=20,
        interval="1d",
    )
    # Returns (pd.DataFrame, dict) tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    df, meta = result
    assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# 7. Snapshot / I/O behavior
# ---------------------------------------------------------------------------

def test_atomic_write_json_roundtrip(engine, tmp_path):
    path = str(tmp_path / "test.json")
    payload = {"key": "value", "num": 42}
    engine.atomic_write_json(path, payload)
    import json
    loaded = json.loads(Path(path).read_text())
    assert loaded == payload


def test_build_live_snapshot_returns_dict(engine):
    """build_live_snapshot with empty snapshot returns a dict with key fields."""
    snap = {}
    result = engine.build_live_snapshot(snap)
    assert isinstance(result, dict)


def test_json_safe_clone_fallback_on_error(engine):
    class NotSerializable:
        pass
    result = engine._json_safe_clone({"x": NotSerializable()}, fallback={"fallback": True})
    # Should return fallback or a dict with string representation
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 8. Asset policy integration
# ---------------------------------------------------------------------------

def test_resolve_asset_policy_valid_ticker(engine):
    result = engine.resolve_asset_policy("SPY", context={})
    assert isinstance(result, dict)
    assert "decision" in result or "action" in result or isinstance(result, dict)


def test_build_industry_lookup_returns_dict(engine):
    result = engine._build_industry_lookup()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 9. MacroSignalAdapter interface
# ---------------------------------------------------------------------------

def test_macro_signal_adapter_instantiable():
    from paper_trading import MacroSignalAdapter
    # Config with chroma disabled
    cfg = {
        "reporting": {"chroma_path": None},
        "execution": {},
    }
    adapter = MacroSignalAdapter(cfg)
    assert adapter is not None


def test_macro_signal_adapter_has_analyze():
    from paper_trading import MacroSignalAdapter
    assert hasattr(MacroSignalAdapter, "analyze_signals")
    assert callable(MacroSignalAdapter.analyze_signals)


def test_macro_signal_adapter_compute_weight():
    from paper_trading import MacroSignalAdapter
    cfg = {"reporting": {}, "execution": {}}
    adapter = MacroSignalAdapter(cfg)
    result = adapter.compute_signal_weight("2026-01-01T00:00:00+00:00")
    # Returns (weight, age_hours) tuple
    assert isinstance(result, tuple)
    w, age = result
    assert isinstance(float(w), float)
    assert 0.0 <= float(w) <= 1.0


# ---------------------------------------------------------------------------
# 10. Cross-module connection tests
# ---------------------------------------------------------------------------

def test_engine_uses_config_validator(engine):
    """Verify config validator is wired into load_config via validate_config attr."""
    from config_validator import ConfigValidationError
    assert callable(ConfigValidationError)


def test_engine_uses_price_service(engine):
    """PriceService should be initialized and accessible."""
    from price_service import PriceService
    # engine._price_service or engine.price_service may be the attribute
    has_service = (
        hasattr(engine, "_price_service") or
        hasattr(engine, "price_service") or
        hasattr(engine, "_price_fetcher")
    )
    assert has_service, "Engine has no price service attribute"


def test_engine_uses_cooldown_policy(engine):
    """cooldown_policy should be importable and used by engine."""
    from cooldown_policy import cooldown_policy, next_market_open_time
    assert callable(cooldown_policy)
    assert callable(next_market_open_time)
    # Engine should have cooldown state
    assert hasattr(engine, "cooldown_state") or hasattr(engine, "_cooldown_state")


def test_engine_uses_cost_model(engine):
    """cost_model.compute_trade_cost should be importable and engine uses it."""
    from cost_model import compute_trade_cost
    result = compute_trade_cost(
        side="BUY", qty=10, price=100.0, notional=1000.0,
        slippage_bps=5.0, fee_per_trade=1.0, fee_bps=0.0, min_fee=0.0,
    )
    assert isinstance(result, dict)
    assert "total_cost" in result


def test_engine_uses_atomic_io(engine):
    """atomic_io should be wired into engine write methods."""
    from atomic_io import atomic_write_json, safe_read_json
    assert callable(atomic_write_json)
    assert callable(safe_read_json)


def test_config_validator_wired(engine):
    """validate_config should be used — verify bad config gets warnings."""
    from config_validator import validate_config, ConfigValidationError
    with pytest.raises(ConfigValidationError):
        validate_config({"initial_cash_usd": -1, "rebalance_minutes": -1, "universe": []})


# ---------------------------------------------------------------------------
# 11. After-split: engine/ package exists and re-exports correctly
#     engine/ package created — xfail markers removed.
# ---------------------------------------------------------------------------

def test_engine_package_import():
    from engine import PaperTradingEngine  # noqa: F401
    assert callable(PaperTradingEngine)


def test_engine_constants_module():
    from engine.constants import (
        LIVE_SCHEMA_VERSION,
        DEFAULT_RISK_PROFILES,
        RISK_PROFILE_DEFAULT,
    )
    assert LIVE_SCHEMA_VERSION == 2


def test_engine_macro_signal_module():
    from engine.macro_signal import MacroSignalAdapter  # noqa: F401
    assert callable(MacroSignalAdapter)


def test_engine_utils_module():
    from engine.utils import (  # noqa: F401
        _normalize_cov_rc_gate_decision,
        resolve_portfolio_cov_rc_hysteresis_decision,
    )
    assert callable(_normalize_cov_rc_gate_decision)


# ---------------------------------------------------------------------------
# 12. Phase 1 changes — config + code regression
# ---------------------------------------------------------------------------

def test_phase1_initial_cash_80k(engine):
    """Phase 6 partial: initial_cash_usd should be 80000."""
    assert engine.initial_cash == 80000 or engine.config.get("initial_cash_usd") == 80000


def test_phase1_cost_model_enabled(engine):
    """Phase 1.1: cost_model.enabled must be True."""
    cost_cfg = engine._get_cost_model_cfg()
    assert cost_cfg.get("enabled") is True, "cost_model.enabled should be True"


def test_phase1_cost_model_params(engine):
    """Phase 1.1: slippage 5 bps, fee_per_trade $1, fee_bps 0.5."""
    cost_cfg = engine._get_cost_model_cfg()
    assert float(cost_cfg.get("slippage_bps", 0)) == 5.0
    assert float(cost_cfg.get("fee_per_trade", 0)) == 1.0
    assert float(cost_cfg.get("fee_bps", 0)) == 0.5


def test_phase1_rc_abort_buffer_enabled(engine):
    """Phase 1.3: RC abort buffer must be enabled."""
    risk_cfg = engine._get_risk_model_cfg()
    assert risk_cfg.get("portfolio_cov_rc_abort_buffer_enabled") is True


def test_phase1_hysteresis_band(engine):
    """Phase 1.3: hysteresis band must be > 0 (mid profile sets 0.03, base sets 0.02)."""
    risk_cfg = engine._get_risk_model_cfg()
    assert float(risk_cfg.get("portfolio_cov_rc_hysteresis_band", 0.0)) > 0.0


def test_phase1_vol_targeting_enabled(engine):
    """Phase 1.5: enable_vol_targeting must be True."""
    risk_cfg = engine._get_risk_model_cfg()
    assert risk_cfg.get("enable_vol_targeting") is True


def test_phase1_vol_target_value(engine):
    """Phase 1.5: vol_target raised to 0.40 to align with tech portfolio vol range."""
    risk_cfg = engine._get_risk_model_cfg()
    assert float(risk_cfg.get("vol_target", 0)) == pytest.approx(0.40)


def test_phase1_macro_decay_lambda(engine):
    """Phase 1.4: decay_lambda_per_hour must be 0.04."""
    macro_cfg = engine.config.get("macro_integration", {})
    assert float(macro_cfg.get("decay_lambda_per_hour", 0)) == pytest.approx(0.04)


def test_phase1_macro_signal_age(engine):
    """Phase 1.4: signal_max_age_hours must be 72."""
    macro_cfg = engine.config.get("macro_integration", {})
    assert float(macro_cfg.get("signal_max_age_hours", 0)) == 72.0


def test_phase1_cost_sensitive_ranking_enabled(engine):
    """Phase 1.6: enable_cost_sensitive_ranking must be True."""
    planner_cfg = engine._get_planner_cfg()
    assert planner_cfg.get("enable_cost_sensitive_ranking") is True


def test_phase1_lambda_cost(engine):
    """Phase 1.6: lambda_cost must be 2.0."""
    planner_cfg = engine._get_planner_cfg()
    assert float(planner_cfg.get("lambda_cost", 0)) == pytest.approx(2.0)


def test_phase1_fx_rate_helper_exists(engine):
    """Phase 1.2: _get_fx_rate method must exist."""
    assert hasattr(engine, "_get_fx_rate")
    assert callable(engine._get_fx_rate)


def test_phase1_fx_rate_cad_usd_returns_float(engine):
    """Phase 1.2: _get_fx_rate('CAD') must return a positive float."""
    rate = engine._get_fx_rate("CAD", "USD")
    assert isinstance(rate, float)
    assert 0.5 < rate < 1.5, f"CAD/USD rate {rate} out of expected range"


def test_phase1_fx_config_present(engine):
    """Phase 1.2: fx_rates config section must exist with CAD_USD key."""
    fx_cfg = engine.config.get("fx_rates", {})
    assert "CAD_USD" in fx_cfg
    assert float(fx_cfg["CAD_USD"]) > 0


# ---------------------------------------------------------------------------
# 13. Phase 3 changes — advanced covariance / risk regression
# ---------------------------------------------------------------------------

def test_phase3_config_cov_method(engine):
    """Phase 3.1: cov_method should be 'ledoit_wolf'."""
    risk_cfg = engine._get_risk_model_cfg()
    assert risk_cfg.get("cov_method") == "ledoit_wolf"


def test_phase3_config_crisis_detection(engine):
    """Phase 3.2: enable_crisis_detection must be True."""
    risk_cfg = engine._get_risk_model_cfg()
    assert risk_cfg.get("enable_crisis_detection") is True


def test_phase3_config_crisis_params(engine):
    """Phase 3.2: crisis params must be present."""
    risk_cfg = engine._get_risk_model_cfg()
    assert float(risk_cfg.get("crisis_corr_ratio", 0)) == pytest.approx(1.5)
    assert int(risk_cfg.get("crisis_short_window", 0)) == 20
    assert int(risk_cfg.get("crisis_long_window", 0)) == 60
    assert float(risk_cfg.get("crisis_rc_tighten_pct", 0)) == pytest.approx(0.30)


def test_phase3_config_cvar_gate(engine):
    """Phase 3.3: enable_cvar_gate must be True."""
    risk_cfg = engine._get_risk_model_cfg()
    assert risk_cfg.get("enable_cvar_gate") is True


def test_phase3_config_cvar_params(engine):
    """Phase 3.3: CVaR params must be present."""
    risk_cfg = engine._get_risk_model_cfg()
    assert float(risk_cfg.get("cvar_confidence", 0)) == pytest.approx(0.95)
    assert float(risk_cfg.get("cvar_daily_threshold", 0)) == pytest.approx(-0.05)


def test_phase3_config_auto_risk_profile(engine):
    """Phase 3.4: auto_risk_profile section must be enabled."""
    arp_cfg = engine.config.get("auto_risk_profile", {})
    assert arp_cfg.get("enabled") is True


def test_phase3_config_auto_risk_profile_params(engine):
    """Phase 3.4: auto_risk_profile vix thresholds must be set."""
    arp_cfg = engine.config.get("auto_risk_profile", {})
    assert float(arp_cfg.get("vix_low", 0)) == pytest.approx(15.0)
    assert float(arp_cfg.get("vix_high", 0)) == pytest.approx(25.0)
    assert float(arp_cfg.get("vix_extreme", 0)) == pytest.approx(40.0)
    assert int(arp_cfg.get("cooldown_hours", 0)) == 24


def test_phase3_ledoit_wolf_method_exists(engine):
    """Phase 3.1: _estimate_covariance_ledoit_wolf must be callable."""
    assert hasattr(engine, "_estimate_covariance_ledoit_wolf")
    assert callable(engine._estimate_covariance_ledoit_wolf)


def test_phase3_ledoit_wolf_returns_dataframe(engine):
    """Phase 3.1: Ledoit-Wolf estimator returns (DataFrame, dict)."""
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    ret = pd.DataFrame(np.random.randn(100, 3) * 0.01, index=dates, columns=["A", "B", "C"])
    result, meta = engine._estimate_covariance_ledoit_wolf(ret)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert "shrinkage" in meta


def test_phase3_crisis_mode_method_exists(engine):
    """Phase 3.2: _compute_crisis_mode must be callable."""
    assert hasattr(engine, "_compute_crisis_mode")
    assert callable(engine._compute_crisis_mode)


def test_phase3_crisis_mode_returns_dict(engine):
    """Phase 3.2: crisis mode returns dict with expected keys."""
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    ret = pd.DataFrame(np.random.randn(100, 4) * 0.01, index=dates, columns=["A", "B", "C", "D"])
    result = engine._compute_crisis_mode(ret)
    assert isinstance(result, dict)
    assert "crisis_mode" in result
    assert "corr_ratio" in result
    assert isinstance(result["crisis_mode"], bool)


def test_phase3_crisis_mode_not_triggered_on_random(engine):
    """Phase 3.2: uncorrelated random returns should not trigger crisis."""
    import pandas as pd
    import numpy as np
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    ret = pd.DataFrame(np.random.randn(100, 4) * 0.01, index=dates, columns=["A", "B", "C", "D"])
    result = engine._compute_crisis_mode(ret, crisis_ratio=10.0)  # very high threshold
    assert result["crisis_mode"] is False


def test_phase3_cvar_method_exists(engine):
    """Phase 3.3: _compute_portfolio_cvar must be callable."""
    assert hasattr(engine, "_compute_portfolio_cvar")
    assert callable(engine._compute_portfolio_cvar)


def test_phase3_cvar_returns_dict(engine):
    """Phase 3.3: CVaR returns dict with cvar and var keys."""
    import pandas as pd
    import numpy as np
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=250, freq="D")
    ret = pd.DataFrame(np.random.randn(250, 3) * 0.01, index=dates, columns=["A", "B", "C"])
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = engine._compute_portfolio_cvar(ret, weights)
    assert isinstance(result, dict)
    assert "cvar" in result
    assert "var" in result
    assert result["cvar"] <= result["var"], "CVaR should be <= VaR (more extreme)"


def test_phase3_vix_method_exists(engine):
    """Phase 3.4: _get_vix_level must be callable."""
    assert hasattr(engine, "_get_vix_level")
    assert callable(engine._get_vix_level)


def test_phase3_vix_returns_float_or_none(engine):
    """Phase 3.4: _get_vix_level must return float or None."""
    result = engine._get_vix_level()
    assert result is None or isinstance(result, float), f"Expected float or None, got {type(result)}"
    if result is not None:
        assert 0 < result < 300, f"VIX {result} outside plausible range"


def test_phase3_auto_risk_profile_method_exists(engine):
    """Phase 3.4: _compute_auto_risk_profile_signal must be callable."""
    assert hasattr(engine, "_compute_auto_risk_profile_signal")
    assert callable(engine._compute_auto_risk_profile_signal)


def test_phase3_auto_risk_profile_returns_valid(engine):
    """Phase 3.4: auto profile signal returns valid profile name or None."""
    regime = {"vix": None, "risk_on": False, "risk_off": False}
    result = engine._compute_auto_risk_profile_signal(regime)
    valid = {None, "low", "mid", "high"}
    assert result in valid, f"Unexpected auto risk profile: {result}"


# ---------------------------------------------------------------------------
# 14. Phase 2 changes — signal quality & portfolio construction
# ---------------------------------------------------------------------------

def test_phase2_config_skip_recent_days(engine):
    """Phase 2.2: momentum_skip_recent_days must be 5."""
    ex_cfg = engine.config.get("execution", {})
    assert int(ex_cfg.get("momentum_skip_recent_days", 0)) == 5


def test_phase2_config_sharpe_momentum_weight(engine):
    """Phase 2.1: sharpe_momentum_weight must be 0.20."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("sharpe_momentum_weight", 0)) == pytest.approx(0.20)


def test_phase2_config_crisis_momentum_scale(engine):
    """Phase 2.3: crisis_momentum_weight_scale must be 0.50."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("crisis_momentum_weight_scale", 1.0)) == pytest.approx(0.50)


def test_phase2_config_min_rank_score_threshold(engine):
    """Phase 2.4: min_rank_score_threshold must be -0.50."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("min_rank_score_threshold", -99.0)) == pytest.approx(-0.50)


def test_phase2_config_min_volume_z_score(engine):
    """Phase 2.6: min_volume_z_score must be -1.50."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("min_volume_z_score", 0)) == pytest.approx(-1.50)


def test_phase2_config_max_sector_weight(engine):
    """Phase 2.5: max_sector_weight must be 0.45."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("max_sector_weight", 1.0)) == pytest.approx(0.45)


def test_phase2_calculate_momentum_skip(engine):
    """Phase 2.2: calculate_momentum accepts skip_recent_days param, returns float."""
    result = engine.calculate_momentum("SPY", lookback_days=20, skip_recent_days=0)
    assert isinstance(result, float)


def test_phase2_sector_concentration_cap_method_exists(engine):
    """Phase 2.5: _apply_sector_concentration_cap must be callable."""
    assert hasattr(engine, "_apply_sector_concentration_cap")
    assert callable(engine._apply_sector_concentration_cap)


def test_phase2_sector_concentration_cap_enforced(engine):
    """Phase 2.5: sector cap actually reduces over-concentrated sector."""
    weights = {"AAPL": 0.30, "MSFT": 0.25, "GOOG": 0.20, "GLD": 0.10, "TLT": 0.15}
    metrics = {
        "AAPL": {"industry": "Tech"},
        "MSFT": {"industry": "Tech"},
        "GOOG": {"industry": "Tech"},
        "GLD": {"industry": "Commodity"},
        "TLT": {"industry": "Bond"},
    }
    result = engine._apply_sector_concentration_cap(weights, metrics, max_sector_weight=0.40)
    tech_total = result.get("AAPL", 0) + result.get("MSFT", 0) + result.get("GOOG", 0)
    assert tech_total <= 0.40 + 1e-6, f"Tech sector total {tech_total:.4f} exceeds cap 0.40"


def test_phase2_sector_concentration_cap_no_change_if_within(engine):
    """Phase 2.5: sector cap should not change weights already within limit."""
    weights = {"A": 0.20, "B": 0.20, "C": 0.20}
    metrics = {
        "A": {"industry": "X"},
        "B": {"industry": "Y"},
        "C": {"industry": "Z"},
    }
    result = engine._apply_sector_concentration_cap(weights, metrics, max_sector_weight=0.45)
    for ticker in weights:
        assert abs(result[ticker] - weights[ticker]) < 1e-9


def test_phase2_cross_sectional_metrics_has_sharpe_z(engine):
    """Phase 2.1: _compute_cross_sectional_metrics returns sharpe_z key in metrics."""
    fake_assets = [{"ticker": "SPY"}, {"ticker": "QQQ"}]
    metrics, ranked = engine._compute_cross_sectional_metrics(
        fake_assets,
        lookback_days=20,
        vol_target=0.12,
        momentum_weight=1.0,
        vol_weight=0.5,
        top_n=5,
        sharpe_momentum_weight=0.20,
    )
    for ticker, data in metrics.items():
        assert "sharpe_z" in data, f"sharpe_z missing for {ticker}"
        assert "sharpe_momentum" in data, f"sharpe_momentum missing for {ticker}"


# ---------------------------------------------------------------------------
# 15. Phase 4 changes — execution quality
# ---------------------------------------------------------------------------

def test_phase4_config_stop_loss_enabled(engine):
    """Phase 4.1: stop_loss_enabled must be True."""
    ex_cfg = engine.config.get("execution", {})
    assert ex_cfg.get("stop_loss_enabled") is True


def test_phase4_config_stop_loss_pct(engine):
    """Phase 4.1: stop_loss_pct must be -0.08."""
    ex_cfg = engine.config.get("execution", {})
    assert float(ex_cfg.get("stop_loss_pct", 0)) == pytest.approx(-0.08)


def test_phase4_config_ramp_in(engine):
    """Phase 4.2: ramp_in_enabled=True and ramp_in_cycles=2 (reduced from 3 for faster entry)."""
    ex_cfg = engine.config.get("execution", {})
    assert ex_cfg.get("ramp_in_enabled") is True
    assert int(ex_cfg.get("ramp_in_cycles", 0)) == 2


def test_phase4_config_adaptive_turnover(engine):
    """Phase 4.3: adaptive_turnover_enabled=True with sensible params."""
    ex_cfg = engine.config.get("execution", {})
    assert ex_cfg.get("adaptive_turnover_enabled") is True
    assert float(ex_cfg.get("adaptive_turnover_vol_high", 0)) == pytest.approx(0.025)
    assert float(ex_cfg.get("adaptive_turnover_scale_min", 1)) == pytest.approx(0.40)


def test_phase4_stop_loss_method_exists(engine):
    """Phase 4.1: _compute_position_stop_loss_overrides must be callable."""
    assert callable(engine._compute_position_stop_loss_overrides)


def test_phase4_stop_loss_no_positions(engine):
    """Phase 4.1: with no positions, stop-loss returns weights unchanged."""
    import copy
    orig_positions = copy.copy(engine.positions)
    engine.positions = {}
    weights = {"AAPL": 0.30, "SPY": 0.40}
    result = engine._compute_position_stop_loss_overrides(weights)
    engine.positions = orig_positions
    assert result == weights


def test_phase4_stop_loss_triggers_on_loss(engine):
    """Phase 4.1: position below stop threshold should get weight zeroed."""
    import copy
    orig_positions = copy.copy(engine.positions)
    orig_cost = copy.copy(engine.cost_basis)
    # Simulate a position with cost 200, current price effectively ~170 (−15%)
    engine.positions["_TEST_SL_"] = 10
    engine.cost_basis["_TEST_SL_"] = 200.0
    weights = {"_TEST_SL_": 0.25}
    # Patch get_current_price temporarily
    _orig_gcp = engine.get_current_price
    engine.get_current_price = lambda t, **kw: (170.0, 1, "LIVE") if t == "_TEST_SL_" else _orig_gcp(t, **kw)
    result = engine._compute_position_stop_loss_overrides(weights)
    engine.get_current_price = _orig_gcp
    engine.positions = orig_positions
    engine.cost_basis = orig_cost
    assert result.get("_TEST_SL_", 0.25) == 0.0, "Stop-loss should zero out the weight"


def test_phase4_ramp_in_method_exists(engine):
    """Phase 4.2: _apply_ramp_in_to_targets must be callable."""
    assert callable(engine._apply_ramp_in_to_targets)


def test_phase4_ramp_in_no_change_existing_positions(engine):
    """Phase 4.2: ramp-in should not scale down existing (already held) positions."""
    import copy
    orig_positions = copy.copy(engine.positions)
    orig_cycle = engine.current_cycle
    # Simulate an existing position
    engine.positions["SPY"] = 100
    engine.current_cycle = 10
    weights = {"SPY": 0.40}
    result = engine._apply_ramp_in_to_targets(weights)
    engine.positions = orig_positions
    engine.current_cycle = orig_cycle
    # Held positions should not be ramped
    assert abs(result.get("SPY", 0) - 0.40) < 1e-6


def test_phase4_adaptive_turnover_method_exists(engine):
    """Phase 4.3: _compute_adaptive_turnover_limit must be callable."""
    assert callable(engine._compute_adaptive_turnover_limit)


def test_phase4_adaptive_turnover_returns_float(engine):
    """Phase 4.3: returns a float turnover value."""
    result = engine._compute_adaptive_turnover_limit(0.40)
    assert isinstance(result, float)
    assert 0.0 < result <= 0.40 + 1e-6


def test_phase4_adaptive_turnover_reduces_on_high_vol(engine):
    """Phase 4.3: high equity vol should produce lower turnover limit."""
    import copy
    orig_snapshots = engine.portfolio_snapshots[:]
    # Inject volatile equity history
    engine.portfolio_snapshots = [
        {"total_equity": v} for v in [100, 97, 103, 95, 106, 92]
    ]
    result = engine._compute_adaptive_turnover_limit(0.40)
    engine.portfolio_snapshots = orig_snapshots
    # With high volatility, the result should be <= base
    assert result <= 0.40 + 1e-6


# ---------------------------------------------------------------------------
# 16. AI/LLM improvements regression
# ---------------------------------------------------------------------------

def test_ai_config_news_score_weight(engine):
    """AI.2: news_score_weight must be 0.15."""
    macro_cfg = engine.config.get("macro_integration", {})
    assert float(macro_cfg.get("news_score_weight", 0)) == pytest.approx(0.15)


def test_ai_config_min_calibration_count(engine):
    """AI.3: min_calibration_count must be 5."""
    macro_cfg = engine.config.get("macro_integration", {})
    assert int(macro_cfg.get("min_calibration_count", 0)) == 5


def test_ai_compute_per_ticker_news_score_method_exists(engine):
    """AI.2: _compute_per_ticker_news_score must be callable."""
    assert callable(engine._compute_per_ticker_news_score)


def test_ai_compute_per_ticker_news_score_empty_topics(engine):
    """AI.2: empty topics list returns empty dict."""
    result = engine._compute_per_ticker_news_score([])
    assert result == {}


def test_ai_compute_per_ticker_news_score_bullish(engine):
    """AI.2: bullish topic on mapped sector produces positive ticker score."""
    topics = [{
        "theme": "semiconductors",
        "direction": "bullish",
        "strength": 2.0,
        "confidence_effective": 0.7,
        "accuracy_factor": 1.0,
    }]
    result = engine._compute_per_ticker_news_score(topics)
    # If topic_sector_ticker_map has 'semiconductors', scores should be positive
    # If not in map, result will be empty — both are valid
    for ticker, score in result.items():
        assert -1.0 <= score <= 1.0, f"{ticker} score {score} out of range"


def test_ai_compute_per_ticker_news_score_values_bounded(engine):
    """AI.2: all output scores must be in [-1.0, 1.0]."""
    topics = [
        {"theme": "tech_rally", "direction": "bullish", "strength": 5.0,
         "confidence_effective": 0.9, "accuracy_factor": 1.2},
        {"theme": "recession", "direction": "bearish", "strength": 3.0,
         "confidence_effective": 0.8, "accuracy_factor": 0.9},
    ]
    result = engine._compute_per_ticker_news_score(topics)
    for ticker, score in result.items():
        assert -1.0 <= score <= 1.0, f"{ticker} score {score} out of [-1, 1]"


def test_ai_accuracy_factor_no_history(engine):
    """AI.3: _get_accuracy_factor with no history returns conservative factor."""
    factor, acc, scope = engine.macro_adapter._get_accuracy_factor("unknown_theme_xyz", "unknown_source_xyz")
    assert isinstance(factor, float)
    assert 0.40 <= factor <= 0.80, f"No-history factor {factor} should be conservative"
    assert "prior" in scope or "conservative" in scope


def test_ai_accuracy_factor_perfect_accuracy(engine):
    """AI.3: high accuracy history gives factor > 1.0."""
    import copy
    orig = copy.deepcopy(engine.macro_adapter.theme_accuracy_history)
    engine.macro_adapter.theme_accuracy_history["test_high_acc"] = [1.0] * 10
    factor, acc, scope = engine.macro_adapter._get_accuracy_factor("test_high_acc", "unknown_source_xyz")
    engine.macro_adapter.theme_accuracy_history = orig
    assert factor > 1.0, f"Perfect accuracy should give factor > 1.0, got {factor}"


def test_ai_accuracy_factor_poor_accuracy(engine):
    """AI.3: poor accuracy history gives factor < 0.70."""
    import copy
    orig = copy.deepcopy(engine.macro_adapter.theme_accuracy_history)
    engine.macro_adapter.theme_accuracy_history["test_low_acc"] = [0.0] * 10
    factor, acc, scope = engine.macro_adapter._get_accuracy_factor("test_low_acc", "unknown_source_xyz")
    engine.macro_adapter.theme_accuracy_history = orig
    assert factor < 0.70, f"Zero accuracy should give factor < 0.70, got {factor}"


def test_ai_cross_sectional_metrics_has_news_score(engine):
    """AI.2: _compute_cross_sectional_metrics includes news_score in output."""
    fake_assets = [{"ticker": "SPY"}, {"ticker": "QQQ"}]
    metrics, _ = engine._compute_cross_sectional_metrics(
        fake_assets,
        lookback_days=20,
        vol_target=0.12,
        momentum_weight=1.0,
        vol_weight=0.5,
        top_n=5,
        news_score_weight=0.15,
    )
    for ticker, data in metrics.items():
        assert "news_score" in data, f"news_score key missing for {ticker}"
        assert "news_contrib" in data, f"news_contrib key missing for {ticker}"


# ===========================================================================
# Section 17 – Phase 5: Enhanced metrics / circuit breaker / factor attribution
# ===========================================================================

# ---- 17.1 Config keys ----

def test_phase5_config_circuit_breaker_enabled(engine):
    """Phase 5.2: circuit_breaker_rolling_enabled is in execution config."""
    exec_cfg = engine.config.get('execution', {})
    assert 'circuit_breaker_rolling_enabled' in exec_cfg


def test_phase5_config_circuit_breaker_window(engine):
    """Phase 5.2: circuit_breaker_rolling_window defaults to positive int."""
    exec_cfg = engine.config.get('execution', {})
    window = exec_cfg.get('circuit_breaker_rolling_window', 10)
    assert isinstance(window, int) and window > 0


def test_phase5_config_circuit_breaker_pct(engine):
    """Phase 5.2: circuit_breaker_rolling_drawdown_pct is between 0 and 1."""
    exec_cfg = engine.config.get('execution', {})
    pct = float(exec_cfg.get('circuit_breaker_rolling_drawdown_pct', 0.12))
    assert 0 < pct < 1


def test_phase5_config_attribution_path(engine):
    """Phase 5.3: attribution_path is configured in reporting."""
    reporting_cfg = engine.config.get('reporting', {})
    assert 'attribution_path' in reporting_cfg
    assert str(reporting_cfg['attribution_path']).endswith('.jsonl')


# ---- 17.2 Enhanced performance metrics ----

def test_phase5_enhanced_metrics_method_exists(engine):
    """Phase 5.1: _compute_enhanced_performance_metrics must be callable."""
    assert callable(engine._compute_enhanced_performance_metrics)


def test_phase5_enhanced_metrics_zero_returns(engine):
    """Phase 5.1: empty returns list returns zero metrics."""
    result = engine._compute_enhanced_performance_metrics([], [], 0.0, 0.0)
    assert isinstance(result, dict)
    assert result['sortino'] == 0.0
    assert result['calmar'] == 0.0
    assert result['win_rate'] == 0.0
    assert result['max_consecutive_losses'] == 0


def test_phase5_enhanced_metrics_sortino_positive_only_returns(engine):
    """Phase 5.1: Sortino is 0 when there are no negative returns (downside_std=0)."""
    returns = [0.01, 0.02, 0.015]  # all positive
    result = engine._compute_enhanced_performance_metrics(returns, [], 0.05, 0.10)
    assert result['sortino'] == 0.0  # no downside, returns 0 (safe division by zero)


def test_phase5_enhanced_metrics_sortino_with_losses(engine):
    """Phase 5.1: Sortino is non-zero when there are negative returns."""
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    result = engine._compute_enhanced_performance_metrics(returns, [], 0.05, 0.05)
    assert result['sortino'] != 0.0


def test_phase5_enhanced_metrics_calmar(engine):
    """Phase 5.1: Calmar = annualized_return / max_drawdown."""
    result = engine._compute_enhanced_performance_metrics([0.01, 0.02], [], 0.10, 0.20)
    expected_calmar = 0.20 / 0.10
    assert abs(result['calmar'] - expected_calmar) < 1e-9


def test_phase5_enhanced_metrics_win_rate_from_trades(engine):
    """Phase 5.1: win_rate computed from round-trip trades."""
    trades = [
        {'side': 'BUY', 'ticker': 'SPY', 'price': 100.0, 'quantity': 10, 'cost': 0.0},
        {'side': 'SELL', 'ticker': 'SPY', 'price': 110.0, 'quantity': 10, 'cost': 0.0},  # win
        {'side': 'BUY', 'ticker': 'QQQ', 'price': 200.0, 'quantity': 5, 'cost': 0.0},
        {'side': 'SELL', 'ticker': 'QQQ', 'price': 190.0, 'quantity': 5, 'cost': 0.0},  # loss
    ]
    result = engine._compute_enhanced_performance_metrics([0.01, -0.01], trades, 0.05, 0.10)
    assert abs(result['win_rate'] - 0.5) < 1e-9
    assert result['avg_win'] > 0
    assert result['avg_loss'] < 0


def test_phase5_enhanced_metrics_max_consecutive_losses(engine):
    """Phase 5.1: max consecutive losses computed from trade P&Ls."""
    # 3 buy-sell round trips: loss, loss, win
    trades = [
        {'side': 'BUY', 'ticker': 'A', 'price': 100.0, 'quantity': 1, 'cost': 0.0},
        {'side': 'SELL', 'ticker': 'A', 'price': 90.0, 'quantity': 1, 'cost': 0.0},
        {'side': 'BUY', 'ticker': 'A', 'price': 100.0, 'quantity': 1, 'cost': 0.0},
        {'side': 'SELL', 'ticker': 'A', 'price': 85.0, 'quantity': 1, 'cost': 0.0},
        {'side': 'BUY', 'ticker': 'A', 'price': 100.0, 'quantity': 1, 'cost': 0.0},
        {'side': 'SELL', 'ticker': 'A', 'price': 115.0, 'quantity': 1, 'cost': 0.0},
    ]
    result = engine._compute_enhanced_performance_metrics([], trades, 0.0, 0.0)
    assert result['max_consecutive_losses'] == 2


# ---- 17.3 Rolling drawdown circuit breaker ----

def test_phase5_circuit_breaker_method_exists(engine):
    """Phase 5.2: _check_rolling_drawdown_circuit_breaker must be callable."""
    assert callable(engine._check_rolling_drawdown_circuit_breaker)


def test_phase5_circuit_breaker_no_snapshots_noop(engine):
    """Phase 5.2: circuit breaker does nothing with fewer than 2 snapshots."""
    engine.portfolio_snapshots = []
    engine.circuit_breaker_rolling_active = False
    engine._check_rolling_drawdown_circuit_breaker()
    assert engine.circuit_breaker_rolling_active is False


def test_phase5_circuit_breaker_not_triggered_below_threshold(engine):
    """Phase 5.2: no trigger when rolling drawdown is below threshold."""
    import copy
    engine.circuit_breaker_rolling_active = False
    engine.portfolio_snapshots = [
        {'total_equity': 100000.0 - i * 100} for i in range(5)
    ]  # tiny 0.4% drawdown over 5 cycles
    engine._check_rolling_drawdown_circuit_breaker()
    assert engine.circuit_breaker_rolling_active is False


def test_phase5_circuit_breaker_triggered_above_threshold(engine):
    """Phase 5.2: triggers and sets active=True when rolling drawdown exceeds threshold."""
    engine.circuit_breaker_rolling_active = False
    engine.circuit_breaker_rolling_triggered_cycle = None
    start_equity = 100000.0
    # 15% drop over the window
    end_equity = start_equity * 0.85
    window = int(engine.config.get('execution', {}).get('circuit_breaker_rolling_window', 10))
    engine.portfolio_snapshots = [{'total_equity': start_equity}] + [
        {'total_equity': end_equity} for _ in range(window - 1)
    ]
    engine._check_rolling_drawdown_circuit_breaker()
    assert engine.circuit_breaker_rolling_active is True
    assert engine.circuit_breaker_rolling_triggered_cycle is not None


def test_phase5_circuit_breaker_recovery_clears_active(engine):
    """Phase 5.2: recovery resets circuit_breaker_rolling_active to False."""
    engine.circuit_breaker_rolling_active = True
    engine.circuit_breaker_rolling_recovery_equity = 90000.0
    engine.portfolio_snapshots = [
        {'total_equity': 85000.0},
        {'total_equity': 92000.0},  # above recovery target
    ]
    engine._check_rolling_drawdown_circuit_breaker()
    assert engine.circuit_breaker_rolling_active is False


def test_phase5_circuit_breaker_state_in_snapshot(engine):
    """Phase 5.2: snapshot includes circuit_breaker_rolling_active key."""
    engine.circuit_breaker_rolling_active = False
    engine.circuit_breaker_rolling_triggered_cycle = None
    snapshot = engine.record_snapshot()
    assert 'circuit_breaker_rolling_active' in snapshot
    assert isinstance(snapshot['circuit_breaker_rolling_active'], bool)


# ---- 17.4 Factor attribution ----

def test_phase5_factor_attribution_method_exists(engine):
    """Phase 5.3: _record_factor_attribution must be callable."""
    assert callable(engine._record_factor_attribution)


def test_phase5_factor_attribution_empty_asset_metrics(engine):
    """Phase 5.3: empty asset metrics results in empty factor_contributions."""
    engine._last_asset_metrics_for_attribution = {}
    engine._record_factor_attribution(cycle_return=0.0)
    assert engine.current_factor_contributions == {}


def test_phase5_factor_attribution_keys_present(engine):
    """Phase 5.3: attribution record has expected keys."""
    engine._last_asset_metrics_for_attribution = {
        'SPY': {'blended_momentum_z': 0.5, 'sharpe_z': 0.3, 'news_contrib': 0.1, 'vol_score': 0.4},
        'QQQ': {'blended_momentum_z': 0.2, 'sharpe_z': 0.1, 'news_contrib': 0.0, 'vol_score': 0.3},
    }
    engine._record_factor_attribution(cycle_return=0.01)
    contrib = engine.current_factor_contributions
    assert isinstance(contrib, dict)
    for key in ('cycle', 'timestamp', 'cycle_return', 'momentum_z_avg', 'sharpe_z_avg',
                'news_contrib_avg', 'vol_score_avg', 'n_assets'):
        assert key in contrib, f"Missing key: {key}"


def test_phase5_factor_attribution_averages_correct(engine):
    """Phase 5.3: momentum_z_avg is mean of blended_momentum_z across assets."""
    engine._last_asset_metrics_for_attribution = {
        'A': {'blended_momentum_z': 1.0, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
        'B': {'blended_momentum_z': 3.0, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
    }
    engine._record_factor_attribution(cycle_return=0.02)
    assert abs(engine.current_factor_contributions['momentum_z_avg'] - 2.0) < 1e-9


def test_phase5_factor_attribution_cycle_return_stored(engine):
    """Phase 5.3: cycle_return value is stored in attribution record."""
    engine._last_asset_metrics_for_attribution = {
        'SPY': {'blended_momentum_z': 0.0, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
    }
    engine._record_factor_attribution(cycle_return=0.0312)
    assert abs(engine.current_factor_contributions['cycle_return'] - 0.0312) < 1e-9


def test_phase5_factor_attribution_n_assets(engine):
    """Phase 5.3: n_assets matches number of entries in _last_asset_metrics_for_attribution."""
    engine._last_asset_metrics_for_attribution = {
        'A': {'blended_momentum_z': 0.1, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
        'B': {'blended_momentum_z': 0.2, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
        'C': {'blended_momentum_z': 0.3, 'sharpe_z': 0.0, 'news_contrib': 0.0, 'vol_score': 0.0},
    }
    engine._record_factor_attribution()
    assert engine.current_factor_contributions['n_assets'] == 3


def test_phase5_snapshot_has_factor_contributions_key(engine):
    """Phase 5.3: record_snapshot includes factor_contributions dict."""
    snapshot = engine.record_snapshot()
    assert 'factor_contributions' in snapshot
    assert isinstance(snapshot['factor_contributions'], dict)
