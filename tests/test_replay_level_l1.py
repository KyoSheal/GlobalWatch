from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_trading import replay_bundle_once


def test_replay_bundle_l1_pass(tmp_path: Path):
    bundle_dir = tmp_path / "replay_bundle_l1"
    (bundle_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "expected").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "outputs").mkdir(parents=True, exist_ok=True)

    expected_fields = {
        "schema_version": 1,
        "risk_gate_decision": {
            "metric_name": "max_rc_fraction",
            "metric_value": 1.0,
            "threshold": 2.0,
            "reason": "",
            "stage": "cov",
            "basis": "target_weights",
        },
        "execution_summary": {"orders_place": 1, "orders_skip": 0, "skip_reasons": {}},
        "cost_model": {"enabled": True},
        "cost_summary": {"totals": {"total": 0.5}},
        "asset_policy_mode": "FORCE_PROXY",
        "no_trade_summary": {
            "schema_version": 1,
            "has_trade": True,
            "orders_place": 1,
            "orders_skip": 0,
            "top_blockers": [],
        },
        "risk_model_health": {
            "schema_version": 1,
            "risk_gate": {"triggered": False, "reason": ""},
            "coverage": {"returns_missing_count": 0},
            "execution": {"orders_place": 1, "orders_skip": 0},
        },
    }
    prices = {
        "schema_version": 1,
        "tickers": {
            "AAA": {
                "price": 100.0,
                "ts": "2026-02-26T10:30:00-05:00",
                "status": "LIVE",
                "source": "stub",
                "tz_ok": True,
            }
        },
    }
    risk_meta = {
        "schema_version": 1,
        "tickers_order": ["AAA"],
        "target_weights_mapped": {"AAA": 1.0},
        "risk_gate_decision_seed": {
            "metric_name": "max_rc_fraction",
            "threshold": 2.0,
            "stage": "cov",
            "basis": "target_weights",
        },
    }
    execution_seed = {
        "schema_version": 1,
        "execution_summary": {"orders_place": 1, "orders_skip": 0, "skip_reasons": {}},
        "cost_summary": {"totals": {"total": 0.5}},
        "no_trade_summary": {
            "schema_version": 1,
            "has_trade": True,
            "orders_place": 1,
            "orders_skip": 0,
            "top_blockers": [],
        },
        "risk_model_health": {
            "schema_version": 1,
            "risk_gate": {"triggered": False, "reason": ""},
            "coverage": {"returns_missing_count": 0},
            "execution": {"orders_place": 1, "orders_skip": 0},
        },
    }
    manifest = {"schema_version": 1, "code_version": "test", "replay_level": "L1"}

    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(bundle_dir / "inputs" / "prices.json", "w", encoding="utf-8") as f:
        json.dump(prices, f)
    with open(bundle_dir / "inputs" / "risk_input_meta.json", "w", encoding="utf-8") as f:
        json.dump(risk_meta, f)
    with open(bundle_dir / "inputs" / "execution_seed.json", "w", encoding="utf-8") as f:
        json.dump(execution_seed, f)
    with open(bundle_dir / "expected" / "snapshot_key_fields.json", "w", encoding="utf-8") as f:
        json.dump(expected_fields, f)

    np.savez_compressed(bundle_dir / "inputs" / "returns_matrix.npz", matrix=np.ones((1, 10), dtype=np.float32))
    np.savez_compressed(bundle_dir / "inputs" / "cov_matrix.npz", matrix=np.array([[1.0]], dtype=np.float32))

    report = replay_bundle_once(str(bundle_dir), replay_level="L1")
    assert bool(report.get("summary", {}).get("pass", False)) is True

    replay_snapshot_path = bundle_dir / "outputs" / "replay_snapshot.json"
    assert replay_snapshot_path.exists()
    replay_payload = json.loads(replay_snapshot_path.read_text(encoding="utf-8"))
    assert str((replay_payload.get("replay", {}) or {}).get("level", "")) == "L1"
