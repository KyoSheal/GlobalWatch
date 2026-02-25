from __future__ import annotations

import paper_trading


def _make_engine(scope: str):
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.config = {
        "risk_model": {
            "enable_ticker_proxy_for_returns": True,
            "ticker_proxy_scope": scope,
            "ticker_proxy_map": {"XIU.TO": "EWC", "FTS.TO": "FTS"},
        }
    }
    return engine


def test_apply_target_proxy_for_execution_respects_risk_only_scope():
    engine = _make_engine("risk_only")
    target = {"XIU.TO": 0.11, "FTS.TO": 0.09, "SPY": 0.10, "CASH": 0.70}
    mapped, rows = paper_trading.PaperTradingEngine._apply_target_proxy_for_execution(engine, target)
    assert mapped == target
    assert rows == []


def test_apply_target_proxy_for_execution_maps_under_risk_and_execution_scope():
    engine = _make_engine("risk_and_execution")
    target = {"XIU.TO": 0.11, "FTS.TO": 0.09, "SPY": 0.10, "CASH": 0.70}
    mapped, rows = paper_trading.PaperTradingEngine._apply_target_proxy_for_execution(engine, target)
    assert abs(float(mapped.get("EWC", 0.0)) - 0.11) < 1e-12
    assert abs(float(mapped.get("FTS", 0.0)) - 0.09) < 1e-12
    assert abs(float(mapped.get("SPY", 0.0)) - 0.10) < 1e-12
    assert abs(float(mapped.get("CASH", 0.0)) - 0.70) < 1e-12
    assert "XIU.TO" not in mapped
    assert "FTS.TO" not in mapped
    assert isinstance(rows, list) and len(rows) == 2
