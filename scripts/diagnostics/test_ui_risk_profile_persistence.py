#!/usr/bin/env python3
"""Regression test for UI risk profile source-of-truth priority."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json
from risk_profile_ui_utils import get_active_risk_profile


def main() -> int:
    root = Path("outputs") / "test_ui_risk_profile_persistence"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    runtime_control_path = root / "runtime_control.json"
    snapshot_live_path = root / "snapshot_live.json"
    config_path = root / "paper_config.json"

    atomic_write_json(
        snapshot_live_path,
        {
            "active_risk_profile": "mid",
        },
        indent=2,
    )
    atomic_write_json(
        runtime_control_path,
        {
            "schema_version": 1,
            "updated_at_utc": "2026-02-20T00:00:00+00:00",
            "request_id": "ut-ui-risk-persist",
            "requested_risk_profile": "high",
        },
        indent=2,
    )
    atomic_write_json(
        config_path,
        {
            "execution": {
                # intentionally invalid, runtime control must still win
                "risk_profile": "manual",
            },
            "reporting": {
                "runtime_control_path": str(runtime_control_path),
                "snapshot_live_path": str(snapshot_live_path),
            },
        },
        indent=2,
    )

    selected = get_active_risk_profile(config_path=str(config_path))
    assert selected == "high", f"expected runtime requested profile 'high', got {selected!r}"

    # Further corrupt config to ensure runtime priority remains unchanged.
    atomic_write_json(
        config_path,
        {
            "execution": {
                "risk_profile": "###invalid###",
            },
            "reporting": {
                "runtime_control_path": str(runtime_control_path),
                "snapshot_live_path": str(snapshot_live_path),
            },
        },
        indent=2,
    )
    selected2 = get_active_risk_profile(config_path=str(config_path))
    assert selected2 == "high", f"expected runtime requested profile 'high' after config corruption, got {selected2!r}"

    print("[PASS] ui_risk_profile_persistence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
