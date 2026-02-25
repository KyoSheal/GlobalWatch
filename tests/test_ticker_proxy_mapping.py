from __future__ import annotations

import paper_trading


def test_ticker_proxy_mapping_lifts_returns_coverage():
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.config = {
        "execution": {"cov_coverage_top_n": 20, "cov_coverage_max_list": 200},
        "risk_model": {
            "enable_ticker_proxy_for_returns": True,
            "ticker_proxy_map": {"XIU.TO": "EWC", "FTS.TO": "FTS"},
        },
    }
    engine._last_cov_coverage_dump_meta = {}

    target_weights = {
        "XIU.TO": 0.1107,
        "FTS.TO": 0.0932,
        "SPY": 0.1,
        "XLP": 0.1,
    }
    diag = {
        "status": "ok",
        "returns_meta": {
            "used_tickers": ["SPY", "XLP"],
            "used_tickers_mapped": ["SPY", "XLP", "EWC", "FTS"],
            "missing_tickers": [],
            "dropped_tickers": [],
            "ticker_proxy_map_used": [
                {"from": "XIU.TO", "to": "EWC", "reason": "returns_missing"},
                {"from": "FTS.TO", "to": "FTS", "reason": "returns_missing"},
            ],
        },
    }

    coverage = paper_trading.PaperTradingEngine._build_cov_coverage_dump(
        engine,
        target_weights,
        diag,
        basis="target_weights",
    )

    assert float(coverage.get("known_weight", 0.0) or 0.0) >= 0.9
    top_missing = coverage.get("top_missing", [])
    top_missing_tickers = {str(row.get("ticker", "")).upper() for row in top_missing if isinstance(row, dict)}
    assert "XIU.TO" not in top_missing_tickers
    assert "FTS.TO" not in top_missing_tickers

    meta = engine._last_cov_coverage_dump_meta
    assert bool(meta.get("ticker_proxy_used", False)) is True
    proxy_rows = meta.get("ticker_proxy_map_used", [])
    assert isinstance(proxy_rows, list) and len(proxy_rows) >= 2


def test_risk_gate_result_records_ticker_proxy_map_used():
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.current_cycle = 999
    engine.config = {
        "strategy": {"lookback_days": 40},
        "execution": {
            "max_portfolio_volatility": 0.25,
            "portfolio_vol_min_coverage": 0.70,
            "enable_diversity_check": False,
            "enable_target_cov_gate": True,
            "target_cov_gate_min_coverage": 0.60,
            "target_cov_gate_require_ok": True,
            "cov_coverage_top_n": 20,
            "cov_coverage_max_list": 200,
        },
        "risk_model": {
            "use_cov_vol_for_gate": True,
            "rc_limit": 0.35,
            "min_cov_gate_coverage": 0.60,
            "cov_gate_fallback_to_weighted": True,
            "enable_ticker_proxy_for_returns": True,
            "ticker_proxy_map": {"XIU.TO": "EWC", "FTS.TO": "FTS"},
        },
    }
    engine.portfolio_snapshots = []
    engine.positions = {}
    engine.cash = 30000.0
    engine._get_asset_volatility_optional = lambda *_args, **_kwargs: None
    engine.get_current_price = lambda *_args, **_kwargs: (100.0, 0.0, "LIVE")

    def _cov_diag_stub(_reason_tag, _weights_map, cycle_id=None):  # noqa: ARG001
        return {
            "enabled": True,
            "status": "ok",
            "returns_meta": {
                "overall_row_coverage": 1.0,
                "cols": 4,
                "used_tickers": ["SPY", "XLP", "EWC", "FTS"],
                "used_tickers_mapped": ["SPY", "XLP", "EWC", "FTS"],
                "missing_tickers": [],
                "dropped_tickers": [],
                "ticker_proxy_used": True,
                "ticker_proxy_map_used": [
                    {"from": "XIU.TO", "to": "EWC", "reason": "returns_missing"},
                    {"from": "FTS.TO", "to": "FTS", "reason": "returns_missing"},
                ],
            },
            "portfolio_vol_annualized": 0.08,
            "max_rc_fraction": 0.22,
            "max_rc_ticker": "SPY",
            "avg_pairwise_corr": 0.12,
            "rc_fraction": {"SPY": 0.22, "XLP": 0.20, "XIU.TO": 0.19, "FTS.TO": 0.19},
        }

    engine._compute_cov_diag_cached = _cov_diag_stub

    target_weights = {"XIU.TO": 0.1107, "FTS.TO": 0.0932, "SPY": 0.1, "XLP": 0.1, "CASH": 0.5961}
    result = paper_trading.PaperTradingEngine._evaluate_portfolio_risk_gate(engine, target_weights)

    assert bool(result.get("abort", True)) is False
    assert bool(result.get("ticker_proxy_used", False)) is True
    proxy_rows = result.get("ticker_proxy_map_used", [])
    assert isinstance(proxy_rows, list) and len(proxy_rows) >= 2
