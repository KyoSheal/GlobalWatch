from __future__ import annotations

from tools.run_walkforward import aggregate_walkforward, generate_walkforward_windows


def test_generate_walkforward_windows_basic():
    dates = [
        "2026-02-10",
        "2026-02-26",
        "2026-02-27",
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
        "2026-03-05",
        "2026-03-06",
        "2026-03-09",
    ]
    windows = generate_walkforward_windows(dates, train_days=3, test_days=2, step_days=2)
    assert len(windows) == 3
    assert windows[0].window_id == "window_000"
    assert windows[0].train_dates == ["2026-02-10", "2026-02-26", "2026-02-27"]
    assert windows[0].test_dates == ["2026-03-02", "2026-03-03"]
    assert windows[2].train_dates == ["2026-03-03", "2026-03-04", "2026-03-05"]
    assert windows[2].test_dates == ["2026-03-06", "2026-03-09"]


def test_aggregate_walkforward_comparability_counts():
    windows = generate_walkforward_windows(
        ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"],
        train_days=2,
        test_days=1,
        step_days=1,
    )
    scenario_window = [
        {
            "window_id": "window_000",
            "scenario_id": "s1",
            "train_start": "2026-03-01",
            "train_end": "2026-03-02",
            "test_start": "2026-03-03",
            "test_end": "2026-03-03",
            "days_total": 1,
            "comparable_days_count": 1,
            "non_comparable_days_count": 0,
            "days_with_trades": 1,
            "fills_total": 2,
            "orders_place_total": 1,
            "estimated_cost_total": 1.5,
            "reason_counts": {"traded": 1},
            "config_metadata_status_counts": {"ok": 1},
        },
        {
            "window_id": "window_000",
            "scenario_id": "s2",
            "train_start": "2026-03-01",
            "train_end": "2026-03-02",
            "test_start": "2026-03-03",
            "test_end": "2026-03-03",
            "days_total": 1,
            "comparable_days_count": 0,
            "non_comparable_days_count": 1,
            "days_with_trades": 0,
            "fills_total": 0,
            "orders_place_total": 0,
            "estimated_cost_total": 0.0,
            "reason_counts": {"MARKET_CLOSED": 1},
            "config_metadata_status_counts": {"legacy_snapshot_missing_metadata": 1},
        },
    ]
    summary = aggregate_walkforward(windows[:1], scenario_window, scenarios_total=2)
    assert summary["windows_total"] == 1
    assert summary["scenarios_total"] == 2
    assert summary["test_days_total"] == 2
    assert summary["comparable_test_days_total"] == 1
    assert summary["non_comparable_test_days_total"] == 1
    assert len(summary["windows"]) == 1
    assert len(summary["scenarios"]) == 2


def test_aggregate_walkforward_prefers_scenario_comparable_counts():
    windows = generate_walkforward_windows(
        ["2026-03-01", "2026-03-02", "2026-03-03"],
        train_days=2,
        test_days=1,
        step_days=1,
    )
    scenario_window = [
        {
            "window_id": "window_000",
            "scenario_id": "s1",
            "train_start": "2026-03-01",
            "train_end": "2026-03-02",
            "test_start": "2026-03-03",
            "test_end": "2026-03-03",
            "days_total": 1,
            "comparable_days_count": 0,
            "non_comparable_days_count": 1,
            "scenario_comparable_days_count": 1,
            "scenario_non_comparable_days_count": 0,
            "days_with_trades": 0,
            "fills_total": 0,
            "orders_place_total": 0,
            "estimated_cost_total": 0.0,
            "reason_counts": {"MARKET_CLOSED": 1},
            "config_metadata_status_counts": {"effective_risk_model_config_fingerprint_changed": 1},
            "scenario_metadata_status_counts": {"ok": 1},
            "comparable_days_with_trades": 0,
            "comparable_blocked_days": 1,
            "comparable_fills_total": 0,
            "comparable_orders_place_total": 0,
            "comparable_estimated_cost_total": 0.0,
            "comparable_reason_counts": {"MARKET_CLOSED": 1},
        }
    ]
    summary = aggregate_walkforward(windows[:1], scenario_window, scenarios_total=1)
    assert summary["comparable_test_days_total"] == 1
    assert summary["non_comparable_test_days_total"] == 0
    assert summary["windows"][0]["scenario_metadata_status_counts"]["ok"] == 1
