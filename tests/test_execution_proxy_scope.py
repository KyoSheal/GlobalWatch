from __future__ import annotations

import json
import os
from pathlib import Path

import paper_trading


def _run_scope(tmp_path: Path, scope: str):
    outdir = tmp_path / f"dryrun_{scope}"
    old_enable = os.environ.get("GW_DRYRUN_ENABLE_PROXY")
    old_scope = os.environ.get("GW_PROXY_SCOPE")
    try:
        os.environ["GW_DRYRUN_ENABLE_PROXY"] = "1"
        os.environ["GW_PROXY_SCOPE"] = scope
        rc = paper_trading.debug_run_system_s1_s5(
            config_path="paper_config.json",
            outdir=str(outdir),
            dryrun_real_risk_gate=True,
            proxy_scope=scope,
        )
        assert rc == 0
    finally:
        if old_enable is None:
            os.environ.pop("GW_DRYRUN_ENABLE_PROXY", None)
        else:
            os.environ["GW_DRYRUN_ENABLE_PROXY"] = old_enable
        if old_scope is None:
            os.environ.pop("GW_PROXY_SCOPE", None)
        else:
            os.environ["GW_PROXY_SCOPE"] = old_scope

    snap_path = outdir / "snapshot_live.json"
    assert snap_path.exists()
    with open(snap_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    return snap


def test_execution_proxy_scope_risk_only_skips_price_missing(tmp_path: Path):
    snap = _run_scope(tmp_path, "risk_only")
    summary = snap.get("execution_summary", {}) if isinstance(snap, dict) else {}
    skip_reasons = summary.get("skip_reasons", {}) if isinstance(summary, dict) else {}
    assert int(skip_reasons.get("PRICE_MISSING", 0) or 0) >= 1
    assert bool(snap.get("execution_proxy_used", False)) is False
    assert (snap.get("execution_proxy_map_used") or []) == []


def test_execution_proxy_scope_risk_and_execution_places_mapped(tmp_path: Path):
    snap = _run_scope(tmp_path, "risk_and_execution")
    summary = snap.get("execution_summary", {}) if isinstance(snap, dict) else {}
    assert int(summary.get("orders_place", 0) or 0) >= 1
    assert bool(snap.get("execution_proxy_used", False)) is True
    rows = snap.get("execution_proxy_map_used", [])
    assert isinstance(rows, list) and len(rows) >= 1
    mapped = {(str(r.get("from", "")).upper(), str(r.get("to", "")).upper()) for r in rows if isinstance(r, dict)}
    assert ("XIU.TO", "EWC") in mapped
