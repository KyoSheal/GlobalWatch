from __future__ import annotations

import paper_trading


def _make_engine(mode: str, proxy_map: dict[str, str] | None = None):
    engine = paper_trading.PaperTradingEngine.__new__(paper_trading.PaperTradingEngine)
    engine.config = {
        "asset_data_policy": {
            "mode": str(mode).upper(),
            "match_rules": [{"suffix": ".TO"}],
            "proxy_map": dict(proxy_map or {"XIU.TO": "EWC", "FTS.TO": "FTS"}),
            "allow_execution_proxy": True,
            "allow_risk_proxy": True,
        }
    }
    engine.current_asset_policy_decisions = []
    engine.current_asset_policy_summary = {"counts": {"ALLOW_ORIGINAL": 0, "USE_PROXY": 0, "DISABLE": 0}, "top_reasons": []}
    return engine


def test_asset_policy_disable_asset_for_to_suffix():
    engine = _make_engine("DISABLE_ASSET")
    decision = paper_trading.PaperTradingEngine.resolve_asset_policy(
        engine,
        "XIU.TO",
        context={"stage": "execution", "price_status": "LIVE"},
    )
    assert decision["action"] == "DISABLE"
    assert decision["risk_ticker"] is None
    assert decision["exec_ticker"] is None
    assert decision["reason"] == "POLICY_DISABLE"


def test_asset_policy_force_proxy_maps_to_proxy():
    engine = _make_engine("FORCE_PROXY")
    decision = paper_trading.PaperTradingEngine.resolve_asset_policy(
        engine,
        "XIU.TO",
        context={"stage": "execution", "price_status": "MISSING"},
    )
    assert decision["action"] == "USE_PROXY"
    assert decision["risk_ticker"] == "EWC"
    assert decision["exec_ticker"] == "EWC"
    assert "PRICE_MISSING" in decision["reason"]


def test_asset_policy_allow_original_and_price_missing_reason():
    engine = _make_engine("ALLOW_ORIGINAL")
    decision = paper_trading.PaperTradingEngine.resolve_asset_policy(
        engine,
        "XIU.TO",
        context={"stage": "execution", "price_status": "MISSING"},
    )
    assert decision["action"] == "ALLOW_ORIGINAL"
    assert decision["risk_ticker"] == "XIU.TO"
    assert decision["exec_ticker"] == "XIU.TO"
    assert decision["reason"] == "PRICE_MISSING"


def test_asset_policy_force_proxy_without_mapping_disables():
    engine = _make_engine("FORCE_PROXY", proxy_map={"FTS.TO": "FTS"})
    decision = paper_trading.PaperTradingEngine.resolve_asset_policy(
        engine,
        "XIU.TO",
        context={"stage": "execution", "price_status": "MISSING"},
    )
    assert decision["action"] == "DISABLE"
    assert decision["risk_ticker"] is None
    assert decision["exec_ticker"] is None
    assert decision["reason"] == "NO_PROXY_MAPPING"
