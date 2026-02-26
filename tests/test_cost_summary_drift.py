from __future__ import annotations

from paper_trading import build_drift_report


def _base_payload():
    return {
        "risk_gate_decision": {
            "metric_name": "max_rc_fraction",
            "metric_value": 0.30,
            "threshold": 0.35,
            "reason": "",
        },
        "execution_summary": {"orders_place": 1, "orders_skip": 0},
        "asset_policy_mode": "FORCE_PROXY",
        "cost_model": {"enabled": True},
        "cost_summary": {"totals": {"total": 1.25}},
        "no_trade_summary": {
            "schema_version": 1,
            "has_trade": True,
            "top_blockers": [],
        },
    }


def test_cost_summary_drift_pass_when_matching():
    expected = _base_payload()
    actual = _base_payload()
    report = build_drift_report(expected, actual)
    summary = report.get("summary", {})
    assert bool(summary.get("pass", False)) is True
    sev = summary.get("severity_counts", {})
    assert int((sev or {}).get("CRITICAL", 0)) == 0
    assert int((sev or {}).get("MAJOR", 0)) == 0


def test_cost_summary_drift_major_when_total_changes():
    expected = _base_payload()
    actual = _base_payload()
    actual["cost_summary"] = {"totals": {"total": 9.99}}
    report = build_drift_report(expected, actual)
    summary = report.get("summary", {})
    assert bool(summary.get("pass", True)) is False
    diffs = report.get("diffs", [])
    assert any(
        str(d.get("path", "")) == "$.cost_summary.totals.total"
        and str(d.get("severity", "")) == "MAJOR"
        for d in diffs
    )
