from __future__ import annotations

import json
from pathlib import Path

from paper_trading import replay_bundle_once


def test_replay_bundle_l0_pass(tmp_path: Path):
    bundle_dir = tmp_path / "replay_bundle"
    (bundle_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "expected").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "outputs").mkdir(parents=True, exist_ok=True)

    expected_fields = {
        "schema_version": 1,
        "risk_gate_decision": {
            "metric_name": "max_rc_fraction",
            "metric_value": 0.30,
            "threshold": 0.35,
            "reason": "",
        },
        "execution_summary": {"orders_place": 1, "orders_skip": 0},
        "asset_policy_mode": "FORCE_PROXY",
        "no_trade_summary": {
            "schema_version": 1,
            "has_trade": True,
            "top_blockers": [],
        },
    }
    prices = {
        "schema_version": 1,
        "tickers": {
            "SPY": {
                "price": 500.0,
                "ts": "2026-02-26T10:30:00-05:00",
                "status": "LIVE",
                "source": "stub",
                "tz_ok": True,
            }
        },
    }
    manifest = {"schema_version": 1, "code_version": "test"}

    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(bundle_dir / "inputs" / "prices.json", "w", encoding="utf-8") as f:
        json.dump(prices, f)
    with open(bundle_dir / "expected" / "snapshot_key_fields.json", "w", encoding="utf-8") as f:
        json.dump(expected_fields, f)

    report = replay_bundle_once(str(bundle_dir))
    assert bool(report.get("summary", {}).get("pass", False)) is True

    replay_snapshot_path = bundle_dir / "outputs" / "replay_snapshot.json"
    drift_report_path = bundle_dir / "outputs" / "drift_report.json"
    assert replay_snapshot_path.exists()
    assert drift_report_path.exists()

