from __future__ import annotations

import json
import copy
from pathlib import Path

from tools.run_replay_batch import BundleRecord, ScenarioSpec, extract_daily_result, load_scenarios
from paper_trading import _apply_replay_risk_overrides


def test_load_scenarios_filters_overrides(tmp_path: Path):
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "baseline_mid",
                        "risk_profile": "mid",
                        "risk_model_overrides": {"unknown_key": 1, "rc_limit": 0.3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scenarios = load_scenarios(str(scenario_file))
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.scenario_id == "baseline_mid"
    assert s.risk_profile == "mid"
    assert s.risk_model_overrides == {"rc_limit": 0.3}


def test_extract_daily_result_contains_scenario_fields(tmp_path: Path):
    bundle_dir = tmp_path / "run" / "replay_bundle"
    out_dir = bundle_dir / "outputs"
    expected_dir = bundle_dir / "expected"
    out_dir.mkdir(parents=True)
    expected_dir.mkdir(parents=True)
    expected_seed = {
        "effective_risk_profile": "mid",
        "effective_risk_model_config": {
            "rc_limit": 0.31,
            "portfolio_cov_rc_hysteresis_band": 0.05,
            "portfolio_cov_rc_abort_buffer_enabled": True,
            "portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts": 4,
            "portfolio_cov_rc_abort_buffer_relax_delta": 0.01,
            "portfolio_cov_rc_abort_buffer_active_cycles": 2,
        },
        "effective_risk_model_config_schema_version": 1,
    }
    scenario = ScenarioSpec(
        scenario_id="mid_rc030_hyst003",
        risk_profile="mid",
        risk_model_overrides={"rc_limit": 0.31, "portfolio_cov_rc_hysteresis_band": 0.05},
    )
    expected_for_scenario = _apply_replay_risk_overrides(
        copy.deepcopy(expected_seed),
        scenario_id=scenario.scenario_id,
        risk_profile_override=scenario.risk_profile,
        risk_model_overrides=scenario.risk_model_overrides,
    )
    fp = expected_for_scenario.get("effective_risk_model_config_fingerprint")
    schema_v = expected_for_scenario.get("effective_risk_model_config_schema_version")

    (out_dir / "replay_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-09T22:24:25.320703+00:00",
                "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"MARKET_CLOSED": 1}},
                "cost_summary": {"totals": {"total": 0.0}},
                "effective_hysteresis_band": 0.03,
                "effective_rc_limit": 0.3,
                "effective_risk_profile": "mid",
                "effective_risk_model_source": "snapshot_base+replay_override",
                "effective_risk_model_config_schema_version": schema_v,
                "effective_risk_model_config_fingerprint": fp,
                "effective_risk_model_config": {
                    "rc_limit": 0.31,
                    "portfolio_cov_rc_hysteresis_band": 0.05,
                    "portfolio_cov_rc_abort_buffer_enabled": True,
                    "portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts": 4,
                    "portfolio_cov_rc_abort_buffer_relax_delta": 0.01,
                    "portfolio_cov_rc_abort_buffer_active_cycles": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "drift_report.json").write_text(
        json.dumps(
            {
                    "summary": {"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
                    "config_metadata_compare": {
                        "effective_risk_model_config_schema_version_expected": schema_v,
                        "effective_risk_model_config_schema_version_actual": schema_v,
                        "effective_risk_model_config_fingerprint_expected": fp,
                        "effective_risk_model_config_fingerprint_actual": fp,
                        "schema_version_match": True,
                        "fingerprint_match": True,
                        "status": "ok",
                    },
                }
        ),
        encoding="utf-8",
    )
    (expected_dir / "snapshot_key_fields.json").write_text(
        json.dumps(expected_seed),
        encoding="utf-8",
    )

    rec = BundleRecord(
        date_et="2026-03-09",
        bundle_dir=str(bundle_dir),
        run_dir=str(bundle_dir.parent),
        manifest_path=str(bundle_dir / "manifest.json"),
        run_id="demo",
        created_ts="2026-03-09T22:24:25+00:00",
    )
    row = extract_daily_result(
        target_date="2026-03-09",
        bundle_rec=rec,
        scenario=scenario,
        replay_status="PASS",
        drift_summary={"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
        error=None,
    )
    assert row["scenario_id"] == "mid_rc030_hyst003"
    assert row["risk_profile_requested"] == "mid"
    assert row["effective_risk_profile"] == "mid"
    assert row["effective_risk_model_config_schema_version"] == schema_v
    assert row["effective_risk_model_config_fingerprint"] == fp
    assert row["effective_risk_model_metadata_source"] == "replay_snapshot_metadata"
    assert row["effective_rc_limit"] == 0.31
    assert row["effective_hysteresis_band"] == 0.05
    assert row["effective_abort_buffer_enabled"] is True
    assert row["effective_abort_buffer_trigger_consecutive_aborts"] == 4
    assert row["effective_abort_buffer_relax_delta"] == 0.01
    assert row["effective_abort_buffer_active_cycles"] == 2
    assert row["effective_param_source"] == "replay_snapshot_risk_model_config"
    assert row["config_metadata_compare_status"] == "ok"
    assert row["config_metadata_schema_version_expected"] == schema_v
    assert row["config_metadata_schema_version_actual"] == schema_v
    assert row["config_metadata_schema_version_match"] is True
    assert row["config_metadata_fingerprint_match"] is True
    assert row["scenario_metadata_compare_status"] == "ok"
    assert row["scenario_comparable_day"] is True


def test_extract_daily_result_fallback_to_scenario_override(tmp_path: Path):
    bundle_dir = tmp_path / "run" / "replay_bundle"
    out_dir = bundle_dir / "outputs"
    expected_dir = bundle_dir / "expected"
    out_dir.mkdir(parents=True)
    expected_dir.mkdir(parents=True)
    (out_dir / "replay_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-09T22:24:25.320703+00:00",
                "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"MARKET_CLOSED": 1}},
                "cost_summary": {"totals": {"total": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "drift_report.json").write_text(
        json.dumps(
            {
                "summary": {"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
                "config_metadata_compare": {
                    "effective_risk_model_config_schema_version_expected": None,
                    "effective_risk_model_config_schema_version_actual": None,
                    "effective_risk_model_config_fingerprint_expected": None,
                    "effective_risk_model_config_fingerprint_actual": None,
                    "schema_version_match": None,
                    "fingerprint_match": None,
                    "status": "legacy_snapshot_missing_metadata",
                },
            }
        ),
        encoding="utf-8",
    )
    (expected_dir / "snapshot_key_fields.json").write_text(
        json.dumps(
            {
                "effective_risk_profile": "mid",
                "effective_risk_model_config": {
                    "rc_limit": 0.29,
                    "portfolio_cov_rc_hysteresis_band": 0.0,
                    "portfolio_cov_rc_abort_buffer_enabled": False,
                    "portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts": 3,
                    "portfolio_cov_rc_abort_buffer_relax_delta": 0.02,
                    "portfolio_cov_rc_abort_buffer_active_cycles": 3,
                },
                "effective_risk_model_config_schema_version": 1,
                "effective_risk_model_config_fingerprint": "baseline_fp",
            }
        ),
        encoding="utf-8",
    )
    rec = BundleRecord(
        date_et="2026-03-09",
        bundle_dir=str(bundle_dir),
        run_dir=str(bundle_dir.parent),
        manifest_path=str(bundle_dir / "manifest.json"),
        run_id="demo",
        created_ts="2026-03-09T22:24:25+00:00",
    )
    scenario = ScenarioSpec(
        scenario_id="fallback_case",
        risk_profile="mid",
        risk_model_overrides={
            "rc_limit": 0.3,
            "portfolio_cov_rc_hysteresis_band": 0.03,
            "portfolio_cov_rc_abort_buffer_enabled": True,
            "portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts": 3,
            "portfolio_cov_rc_abort_buffer_relax_delta": 0.02,
            "portfolio_cov_rc_abort_buffer_active_cycles": 3,
        },
    )
    row = extract_daily_result(
        target_date="2026-03-09",
        bundle_rec=rec,
        scenario=scenario,
        replay_status="PASS",
        drift_summary={"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
        error=None,
    )
    assert row["effective_rc_limit"] == 0.3
    assert row["effective_hysteresis_band"] == 0.03
    assert row["effective_abort_buffer_enabled"] is True
    assert row["effective_param_source"] == "scenario_override_fallback"
    assert row["effective_risk_model_config_schema_version"] is None
    assert row["effective_risk_model_config_fingerprint"] is None
    assert row["effective_risk_model_metadata_source"] == "unavailable"
    assert row["config_metadata_compare_status"] == "legacy_snapshot_missing_metadata"
    assert row["scenario_metadata_compare_status"] == "scenario_metadata_missing"
    assert row["scenario_comparable_day"] is False


def test_extract_daily_result_legacy_snapshot_missing_metadata(tmp_path: Path):
    bundle_dir = tmp_path / "run" / "replay_bundle"
    out_dir = bundle_dir / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "replay_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-09T22:24:25.320703+00:00",
                "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"MARKET_CLOSED": 1}},
                "cost_summary": {"totals": {"total": 0.0}},
                "effective_risk_model_config": {"rc_limit": 0.29},
            }
        ),
        encoding="utf-8",
    )
    rec = BundleRecord(
        date_et="2026-03-09",
        bundle_dir=str(bundle_dir),
        run_dir=str(bundle_dir.parent),
        manifest_path=str(bundle_dir / "manifest.json"),
        run_id="demo",
        created_ts="2026-03-09T22:24:25+00:00",
    )
    scenario = ScenarioSpec(
        scenario_id="legacy_case",
        risk_profile="mid",
        risk_model_overrides={},
    )
    row = extract_daily_result(
        target_date="2026-03-09",
        bundle_rec=rec,
        scenario=scenario,
        replay_status="PASS",
        drift_summary={"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
        error=None,
    )
    assert row["effective_rc_limit"] == 0.29
    assert row["effective_risk_model_config_schema_version"] is None
    assert row["effective_risk_model_config_fingerprint"] is None
    assert row["effective_risk_model_metadata_source"] == "legacy_snapshot_missing_metadata"
    assert row["config_metadata_compare_status"] == "drift_report_missing"
    assert row["scenario_metadata_compare_status"] == "scenario_metadata_missing"
    assert row["scenario_comparable_day"] is False


def test_extract_daily_result_metadata_compare_missing(tmp_path: Path):
    bundle_dir = tmp_path / "run" / "replay_bundle"
    out_dir = bundle_dir / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "replay_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-09T22:24:25.320703+00:00",
                "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"MARKET_CLOSED": 1}},
                "cost_summary": {"totals": {"total": 0.0}},
                "effective_risk_model_config": {"rc_limit": 0.29},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "drift_report.json").write_text(
        json.dumps({"summary": {"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}}}),
        encoding="utf-8",
    )
    rec = BundleRecord(
        date_et="2026-03-09",
        bundle_dir=str(bundle_dir),
        run_dir=str(bundle_dir.parent),
        manifest_path=str(bundle_dir / "manifest.json"),
        run_id="demo",
        created_ts="2026-03-09T22:24:25+00:00",
    )
    scenario = ScenarioSpec(
        scenario_id="meta_missing_case",
        risk_profile="mid",
        risk_model_overrides={},
    )
    row = extract_daily_result(
        target_date="2026-03-09",
        bundle_rec=rec,
        scenario=scenario,
        replay_status="PASS",
        drift_summary={"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
        error=None,
    )
    assert row["config_metadata_compare_status"] == "metadata_compare_missing"


def test_extract_daily_result_scenario_expected_unavailable(tmp_path: Path):
    bundle_dir = tmp_path / "run" / "replay_bundle"
    out_dir = bundle_dir / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "replay_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-09T22:24:25.320703+00:00",
                "execution_summary": {"orders_place": 0, "orders_skip": 1, "skip_reasons": {"MARKET_CLOSED": 1}},
                "cost_summary": {"totals": {"total": 0.0}},
                "effective_risk_model_config_schema_version": 1,
                "effective_risk_model_config_fingerprint": "abc123",
                "effective_risk_model_config": {"rc_limit": 0.3},
            }
        ),
        encoding="utf-8",
    )
    rec = BundleRecord(
        date_et="2026-03-09",
        bundle_dir=str(bundle_dir),
        run_dir=str(bundle_dir.parent),
        manifest_path=str(bundle_dir / "manifest.json"),
        run_id="demo",
        created_ts="2026-03-09T22:24:25+00:00",
    )
    scenario = ScenarioSpec(
        scenario_id="missing_expected",
        risk_profile="mid",
        risk_model_overrides={"rc_limit": 0.3},
    )
    row = extract_daily_result(
        target_date="2026-03-09",
        bundle_rec=rec,
        scenario=scenario,
        replay_status="PASS",
        drift_summary={"pass": True, "num_diffs": 0, "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}},
        error=None,
    )
    assert row["scenario_metadata_compare_status"] == "scenario_expected_unavailable"
    assert row["scenario_comparable_day"] is False
