#!/usr/bin/env python3
"""T99: run diagnostics scripts in a fixed sequence and stop on first failure."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TESTS = [
    {"name": "T00 compile_all", "path": "scripts/diagnostics/test_compile_all.py", "optional": False},
    {
        "name": "T04 migrate_legacy_outputs_candidate_rule",
        "path": "scripts/diagnostics/test_migrate_legacy_outputs_candidate_rule.py",
        "optional": False,
    },
    {
        "name": "T05 summarize_runs_default_live_only",
        "path": "scripts/diagnostics/test_summarize_runs_default_live_only.py",
        "optional": False,
    },
    {"name": "T01 risk_profile_unit", "path": "scripts/diagnostics/test_risk_profile_unit.py", "optional": False},
    {
        "name": "T02 risk_profile_normalize_default_mid",
        "path": "scripts/diagnostics/test_risk_profile_normalize_default_mid.py",
        "optional": False,
    },
    {"name": "T03 runtime_control_write_helper", "path": "scripts/diagnostics/test_runtime_control_write_helper.py", "optional": False},
    {"name": "T11 runtime_risk_profile_reject", "path": "scripts/diagnostics/test_runtime_risk_profile_reject.py", "optional": False},
    {"name": "T10 runtime_risk_profile_apply", "path": "scripts/diagnostics/test_runtime_risk_profile_apply.py", "optional": False},
    {
        "name": "T13 telemetry_injection_profile_meta",
        "path": "scripts/diagnostics/test_telemetry_injection_profile_meta.py",
        "optional": False,
    },
    {
        "name": "T14 write_live_snapshot_signature_regression",
        "path": "scripts/diagnostics/test_write_live_snapshot_signature_regression.py",
        "optional": False,
    },
    {"name": "T20 ui_pending_display_logic", "path": "scripts/diagnostics/test_ui_pending_display_logic.py", "optional": False},
    {"name": "T21 ui_control_write_equivalence", "path": "scripts/diagnostics/test_ui_control_write_equivalence.py", "optional": False},
    {"name": "T22 ui_filter_minimal", "path": "scripts/diagnostics/test_ui_filter_minimal.py", "optional": False},
    {
        "name": "T24 ui_equity_sanitize_weekend_blackout",
        "path": "scripts/diagnostics/test_ui_equity_sanitize_weekend_blackout.py",
        "optional": False,
    },
    {"name": "T23 ui_use_active_profile_button", "path": "scripts/diagnostics/test_ui_use_active_profile_button.py", "optional": False},
    {"name": "T30 e2e_engine_control_sync", "path": "scripts/diagnostics/test_e2e_engine_control_sync.py", "optional": False},
    {
        "name": "T31 no_forced_rebalance_on_profile_change",
        "path": "scripts/diagnostics/test_no_forced_rebalance_on_profile_change.py",
        "optional": False,
    },
    {
        "name": "T32 quant_dataset_extract_minimal",
        "path": "scripts/diagnostics/test_quant_dataset_extract_minimal.py",
        "optional": False,
    },
    {
        "name": "T33 quant_metrics_minimal",
        "path": "scripts/diagnostics/test_quant_metrics_minimal.py",
        "optional": False,
    },
    {
        "name": "T34 quant_compare_minimal",
        "path": "scripts/diagnostics/test_quant_compare_minimal.py",
        "optional": False,
    },
    {
        "name": "T35 quant_leaderboard_minimal",
        "path": "scripts/diagnostics/test_quant_leaderboard_minimal.py",
        "optional": False,
    },
    {
        "name": "T36 quant_gate_minimal",
        "path": "scripts/diagnostics/test_quant_gate_minimal.py",
        "optional": False,
    },
    {
        "name": "T37 quant_daily_pack_minimal",
        "path": "scripts/diagnostics/test_quant_daily_pack_minimal.py",
        "optional": False,
    },
    {
        "name": "T38 quant_daily_embed_minimal",
        "path": "scripts/diagnostics/test_quant_daily_embed_minimal.py",
        "optional": False,
    },
    {
        "name": "T39 quant_daily_embed_flat_json",
        "path": "scripts/diagnostics/test_quant_daily_embed_flat_json.py",
        "optional": False,
    },
    {
        "name": "T40 update_daily_reports_index_minimal",
        "path": "scripts/diagnostics/test_update_daily_reports_index_minimal.py",
        "optional": False,
    },
    {
        "name": "T41 daily_quant_pipeline_minimal",
        "path": "scripts/diagnostics/test_daily_quant_pipeline_minimal.py",
        "optional": False,
    },
    {
        "name": "T42 quant_replay_minimal",
        "path": "scripts/diagnostics/test_quant_replay_minimal.py",
        "optional": False,
    },
    {
        "name": "T43 quant_replay_window_minimal",
        "path": "scripts/diagnostics/test_quant_replay_window_minimal.py",
        "optional": False,
    },
    {
        "name": "T44 quant_replay_drift_minimal",
        "path": "scripts/diagnostics/test_quant_replay_drift_minimal.py",
        "optional": False,
    },
    {
        "name": "T45 quant_replay_drift_daily_minimal",
        "path": "scripts/diagnostics/test_quant_replay_drift_daily_minimal.py",
        "optional": False,
    },
    {
        "name": "T46 ci_replay_drift_gate_demo",
        "path": "scripts/diagnostics/test_ci_replay_drift_gate_demo.py",
        "optional": False,
    },
    {
        "name": "T47 backtest_price_store_minimal",
        "path": "scripts/diagnostics/test_backtest_price_store_minimal.py",
        "optional": False,
    },
    {
        "name": "T48 backtest_engine_minimal",
        "path": "scripts/diagnostics/test_backtest_engine_minimal.py",
        "optional": False,
    },
    {
        "name": "T49 weights_from_run_minimal",
        "path": "scripts/diagnostics/test_weights_from_run_minimal.py",
        "optional": False,
    },
    {
        "name": "T50 backtest_from_run_minimal",
        "path": "scripts/diagnostics/test_backtest_from_run_minimal.py",
        "optional": False,
    },
    {
        "name": "T51 backtest_attach_daily_minimal",
        "path": "scripts/diagnostics/test_backtest_attach_daily_minimal.py",
        "optional": False,
    },
    {
        "name": "T52 reconcile_live_vs_backtest_minimal",
        "path": "scripts/diagnostics/test_reconcile_live_vs_backtest_minimal.py",
        "optional": False,
    },
    {
        "name": "T53 reconcile_live_vs_backtest_fill_live_minimal",
        "path": "scripts/diagnostics/test_reconcile_live_vs_backtest_fill_live_minimal.py",
        "optional": False,
    },
    {
        "name": "T54 reconcile_auto_evidence_minimal",
        "path": "scripts/diagnostics/test_reconcile_auto_evidence_minimal.py",
        "optional": False,
    },
    {
        "name": "T55 reconcile_auto_infer_baseline_candidate_minimal",
        "path": "scripts/diagnostics/test_reconcile_auto_infer_baseline_candidate_minimal.py",
        "optional": False,
    },
    {
        "name": "T56 index_timeseries_minimal",
        "path": "scripts/diagnostics/test_index_timeseries_minimal.py",
        "optional": False,
    },
    {
        "name": "T57 quant_alerts_minimal",
        "path": "scripts/diagnostics/test_quant_alerts_minimal.py",
        "optional": False,
    },
    {
        "name": "T58 backtest_sweep_minimal",
        "path": "scripts/diagnostics/test_backtest_sweep_minimal.py",
        "optional": False,
    },
    {
        "name": "T59 backtest_sweep_attach_daily_minimal",
        "path": "scripts/diagnostics/test_backtest_sweep_attach_daily_minimal.py",
        "optional": False,
    },
    {
        "name": "T60 quant_alerts_cost_fragile_minimal",
        "path": "scripts/diagnostics/test_quant_alerts_cost_fragile_minimal.py",
        "optional": False,
    },
    {
        "name": "T61 vol_target_scale_stabilizer_minimal",
        "path": "scripts/diagnostics/test_vol_target_scale_stabilizer_minimal.py",
        "optional": False,
    },
    {
        "name": "T62 vol_target_stabilizer_resume_minimal",
        "path": "scripts/diagnostics/test_vol_target_stabilizer_resume_minimal.py",
        "optional": False,
    },
    {
        "name": "T63 min_keep_turnover_minimal",
        "path": "scripts/diagnostics/test_min_keep_turnover_minimal.py",
        "optional": False,
    },
    {
        "name": "T64 exec_blockers_no_trade_attach_minimal",
        "path": "scripts/diagnostics/test_exec_blockers_no_trade_attach_minimal.py",
        "optional": False,
    },
    {"name": "T12 risk_profile_artifacts_fields", "path": "scripts/diagnostics/test_risk_profile_artifacts_fields.py", "optional": False},
]


def _print_summary(rows: list[dict]) -> None:
    print("\n==== TEST SUMMARY ====")
    print(f"{'name':56} {'status':8} {'elapsed_ms':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row['name'][:56]:56} {row['status'][:8]:8} {int(row['elapsed_ms']):10d}")
    print("-" * 80)


def main() -> int:
    results: list[dict] = []
    total_start = time.perf_counter()

    for item in TESTS:
        name = str(item["name"])
        rel_path = str(item["path"])
        optional = bool(item.get("optional", False))
        script_path = ROOT / rel_path

        if not script_path.exists():
            status = "SKIP" if optional else "MISSING"
            row = {"name": name, "status": status, "elapsed_ms": 0}
            results.append(row)
            print(f"[{status}] {name} ({rel_path})")
            if not optional:
                _print_summary(results)
                return 2
            continue

        print(f"\n[RUN] {name}")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if proc.returncode != 0:
            print(f"[FAIL] {name} rc={proc.returncode} elapsed_ms={elapsed_ms}")
            results.append({"name": name, "status": "FAIL", "elapsed_ms": elapsed_ms})
            _print_summary(results)
            return int(proc.returncode or 1)

        print(f"[PASS] {name} elapsed_ms={elapsed_ms}")
        results.append({"name": name, "status": "PASS", "elapsed_ms": elapsed_ms})

    total_ms = int((time.perf_counter() - total_start) * 1000)
    _print_summary(results)
    print(f"[PASS] all tests passed in {total_ms} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
