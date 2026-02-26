from __future__ import annotations

import json
from pathlib import Path

from paper_trading import PaperTradingEngine


def test_replay_bundle_manifest_includes_l1_matrices(tmp_path: Path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine.replay_bundle_enabled = True
    engine.replay_bundle_level = "L1"
    engine.run_id = "run_test_l1"
    engine.current_cycle = 7
    engine.cash = 10000.0
    engine.positions = {}
    engine.current_macro = {}
    engine.current_execution_proxy_info = {}
    engine.last_target_weights = {"AAA": 1.0, "CASH": 0.0}
    engine.current_risk_check_info = {
        "risk_gate_decision": {
            "metric_name": "max_rc_fraction",
            "metric_value": 0.5,
            "threshold": 0.6,
            "stage": "cov",
            "basis": "target_weights",
            "reason": "",
        },
        "risk_gate_weights": {"AAA": 1.0},
        "cov_risk_diag_target": {
            "max_rc_ticker": "AAA",
            "returns_meta": {"used_tickers": ["AAA"]},
        },
        "ticker_proxy_map_used": [],
    }
    engine._last_replay_bundle_cycle_id = None
    engine._last_replay_bundle_source = None
    engine._last_replay_bundle_path = ""
    engine.config = {
        "strategy": {"lookback_days": 40},
        "risk_model": {"returns_interval": "1d", "returns_lookback_days": 40, "rc_limit": 0.35},
        "asset_data_policy": {"mode": "FORCE_PROXY", "match_rules": [{"suffix": ".TO"}], "proxy_map": {}},
        "reporting": {"out_dir": str(out_dir), "replay_bundle_level": "L1"},
    }

    snapshot_payload = {
        "timestamp": "2026-02-26T12:00:00+00:00",
        "cycle": 7,
        "cycle_id": 7,
        "source": "unit_test",
        "cash": 10000.0,
        "total_equity": 10000.0,
        "positions_detail": {},
        "price_debug": {
            "AAA": {
                "ticker": "AAA",
                "now_ts": "2026-02-26T12:00:00+00:00",
                "status": "LIVE",
                "age_min": 1.0,
                "source": "stub",
                "price_ts": "2026-02-26T11:59:00+00:00",
                "tz_ok": True,
                "thresholds": {"live_max_min": 10.0, "recent_max_min": 60.0},
                "notes": None,
            }
        },
        "risk_gate_decision": engine.current_risk_check_info["risk_gate_decision"],
        "cov_coverage": {},
        "returns_coverage_diag": {"schema_version": 1, "items": []},
        "execution_summary": {"orders_total": 1, "orders_place": 1, "orders_skip": 0, "skip_reasons": {}},
        "cost_summary": {"enabled": False, "totals": {"total": 0.0}},
        "cost_model": {"enabled": False},
        "no_trade_summary": {},
        "risk_model_health": {},
        "asset_policy_summary": {},
        "asset_policy_mode": "FORCE_PROXY",
        "execution_proxy_used": False,
        "execution_proxy_map_used": [],
    }

    result = PaperTradingEngine.build_replay_bundle(engine, snapshot_payload, daily_report={})
    assert isinstance(result, dict)

    manifest_path = out_dir / "replay_bundle" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert str(manifest.get("replay_level")) == "L1"
    entries = {str((row or {}).get("path")): row for row in manifest.get("bundle_contents", []) if isinstance(row, dict)}
    assert "inputs/returns_matrix.npz" in entries
    assert "inputs/cov_matrix.npz" in entries
    assert bool(entries["inputs/returns_matrix.npz"].get("sha256"))
    assert bool(entries["inputs/cov_matrix.npz"].get("sha256"))
