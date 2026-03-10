from __future__ import annotations

from paper_trading import build_drift_report


def _base_payload():
    return {
        "effective_risk_model_config_schema_version": 1,
        "effective_risk_model_config_fingerprint": "abc123",
        "execution_summary": {"orders_place": 0, "orders_skip": 0},
        "no_trade_summary": {"has_trade": False, "top_blockers": []},
        "cost_model": {"enabled": False},
        "cost_summary": {"totals": {"total": 0.0}},
        "risk_model_health": {
            "risk_gate": {"triggered": False, "reason": ""},
            "execution": {"orders_place": 0},
            "coverage": {"returns_missing_count": 0},
        },
        "asset_policy_mode": "FORCE_PROXY",
    }


def test_schema_version_mismatch_is_major():
    expected = _base_payload()
    actual = _base_payload()
    actual["effective_risk_model_config_schema_version"] = 2
    report = build_drift_report(expected, actual)
    diffs = report.get("diffs", [])
    assert any(
        d.get("path") == "$.effective_risk_model_config_schema_version"
        and str(d.get("severity")) == "MAJOR"
        for d in diffs
    )
    assert int(report.get("summary", {}).get("severity_counts", {}).get("MAJOR", 0)) >= 1
    assert report.get("config_metadata_compare", {}).get("status") == "effective_risk_model_config_schema_version_mismatch"


def test_fingerprint_mismatch_same_schema_is_major():
    expected = _base_payload()
    actual = _base_payload()
    actual["effective_risk_model_config_fingerprint"] = "xyz987"
    report = build_drift_report(expected, actual)
    diffs = report.get("diffs", [])
    assert any(
        d.get("path") == "$.effective_risk_model_config_fingerprint"
        and str(d.get("severity")) == "MAJOR"
        for d in diffs
    )
    meta = report.get("config_metadata_compare", {})
    assert meta.get("schema_version_match") is True
    assert meta.get("fingerprint_match") is False


def test_legacy_missing_metadata_is_compatible():
    expected = _base_payload()
    actual = _base_payload()
    actual.pop("effective_risk_model_config_schema_version", None)
    actual.pop("effective_risk_model_config_fingerprint", None)
    report = build_drift_report(expected, actual)
    summary = report.get("summary", {})
    sev = summary.get("severity_counts", {})
    assert int((sev or {}).get("CRITICAL", 0)) == 0
    assert int((sev or {}).get("MAJOR", 0)) == 0
    meta = report.get("config_metadata_compare", {})
    assert meta.get("status") == "legacy_snapshot_missing_metadata"
