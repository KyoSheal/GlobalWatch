from __future__ import annotations

import pytest

from tools.run_walkforward import aggregate_walkforward, generate_walkforward_windows


def test_window_ranking_uses_comparable_only_and_skips_zero_comparable():
    windows = generate_walkforward_windows(
        ["2026-03-01", "2026-03-02", "2026-03-03"],
        train_days=2,
        test_days=1,
        step_days=1,
    )
    scenario_window = [
        {
            "window_id": "window_000",
            "scenario_id": "s_ok",
            "train_start": "2026-03-01",
            "train_end": "2026-03-02",
            "test_start": "2026-03-03",
            "test_end": "2026-03-03",
            "days_total": 1,
            "comparable_days_count": 1,
            "non_comparable_days_count": 0,
            "days_with_trades": 1,
            "fills_total": 5,
            "orders_place_total": 1,
            "estimated_cost_total": 5.0,
            "reason_counts": {"traded": 1},
            "config_metadata_status_counts": {"ok": 1},
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 2,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 5.0,
            "comparable_reason_counts": {"traded": 1},
        },
        {
            "window_id": "window_000",
            "scenario_id": "s_zero",
            "train_start": "2026-03-01",
            "train_end": "2026-03-02",
            "test_start": "2026-03-03",
            "test_end": "2026-03-03",
            "days_total": 1,
            "comparable_days_count": 0,
            "non_comparable_days_count": 1,
            "days_with_trades": 1,
            "fills_total": 9,
            "orders_place_total": 1,
            "estimated_cost_total": 0.0,
            "reason_counts": {"traded": 1},
            "config_metadata_status_counts": {"legacy_snapshot_missing_metadata": 1},
            "comparable_days_with_trades": 0,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 0,
            "comparable_orders_place_total": 0,
            "comparable_estimated_cost_total": 0.0,
            "comparable_reason_counts": {},
        },
    ]
    summary = aggregate_walkforward(windows[:1], scenario_window, scenarios_total=2)
    rankings = summary["window_rankings"]
    assert summary["score_formula"].startswith("10*comparable_days_with_trades")
    r_ok = next(r for r in rankings if r["scenario_id"] == "s_ok")
    r_zero = next(r for r in rankings if r["scenario_id"] == "s_zero")
    assert r_ok["rank"] == 1
    assert r_ok["rank_status"] == "ranked"
    assert pytest.approx(r_ok["score"], rel=1e-8) == 11.5
    assert r_zero["rank"] is None
    assert r_zero["score"] is None
    assert r_zero["rank_status"] == "insufficient_comparable_days"


def test_global_ranking_sorts_by_comparable_score_only():
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
            "days_total": 1,
            "comparable_days_count": 1,
            "non_comparable_days_count": 0,
            "days_with_trades": 1,
            "fills_total": 1,
            "orders_place_total": 1,
            "estimated_cost_total": 0.0,
            "reason_counts": {"traded": 1},
            "config_metadata_status_counts": {"ok": 1},
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 1,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 0.0,
            "comparable_reason_counts": {"traded": 1},
        },
        {
            "window_id": "window_000",
            "scenario_id": "s2",
            "days_total": 1,
            "comparable_days_count": 1,
            "non_comparable_days_count": 0,
            "days_with_trades": 1,
            "fills_total": 1,
            "orders_place_total": 1,
            "estimated_cost_total": 0.0,
            "reason_counts": {"traded": 1},
            "config_metadata_status_counts": {"ok": 1},
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 4,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 0.0,
            "comparable_reason_counts": {"traded": 1},
        },
        {
            "window_id": "window_000",
            "scenario_id": "s3",
            "days_total": 1,
            "comparable_days_count": 0,
            "non_comparable_days_count": 1,
            "days_with_trades": 0,
            "fills_total": 0,
            "orders_place_total": 0,
            "estimated_cost_total": 0.0,
            "reason_counts": {"unknown": 1},
            "config_metadata_status_counts": {"metadata_compare_missing": 1},
            "comparable_days_with_trades": 0,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 0,
            "comparable_orders_place_total": 0,
            "comparable_estimated_cost_total": 0.0,
            "comparable_reason_counts": {},
        },
    ]
    summary = aggregate_walkforward(windows[:1], scenario_window, scenarios_total=3)
    global_rankings = summary["global_rankings"]
    r1 = next(r for r in global_rankings if r["scenario_id"] == "s1")
    r2 = next(r for r in global_rankings if r["scenario_id"] == "s2")
    r3 = next(r for r in global_rankings if r["scenario_id"] == "s3")
    assert r2["rank_global"] == 1
    assert r1["rank_global"] == 2
    assert r3["rank_global"] is None
    assert r3["rank_status"] == "insufficient_comparable_days"


def test_window_winner_selection_filters_by_scenario_aware_comparable_days():
    windows = generate_walkforward_windows(
        ["2026-03-01", "2026-03-02", "2026-03-03"],
        train_days=2,
        test_days=1,
        step_days=1,
    )
    scenario_window = [
        {
            "window_id": "window_000",
            "scenario_id": "eligible_high_score",
            "days_total": 1,
            "scenario_comparable_days_count": 1,
            "scenario_non_comparable_days_count": 0,
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 2,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 0.0,
        },
        {
            "window_id": "window_000",
            "scenario_id": "ineligible_zero_days",
            "days_total": 1,
            "scenario_comparable_days_count": 0,
            "scenario_non_comparable_days_count": 1,
            "comparable_days_with_trades": 0,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 10,
            "comparable_orders_place_total": 0,
            "comparable_estimated_cost_total": 0.0,
        },
    ]
    summary = aggregate_walkforward(
        windows[:1],
        scenario_window,
        scenarios_total=2,
        min_comparable_days_per_window=1,
        min_comparable_days_global=1,
    )
    w0 = summary["windows"][0]
    assert w0["winner_scenario_id"] == "eligible_high_score"
    assert w0["winner_status"] == "ok"
    assert w0["eligible_scenarios_count"] == 1
    rows = w0["window_rankings"]
    winner_row = next(r for r in rows if r["scenario_id"] == "eligible_high_score")
    ineligible_row = next(r for r in rows if r["scenario_id"] == "ineligible_zero_days")
    assert winner_row["eligible_for_window_winner"] is True
    assert winner_row["is_window_winner"] is True
    assert winner_row["window_winner_status"] == "ok"
    assert ineligible_row["eligible_for_window_winner"] is False
    assert ineligible_row["is_window_winner"] is False
    assert ineligible_row["window_winner_status"] == "insufficient_scenario_comparable_days"


def test_global_winner_selection_uses_min_comparable_days_global():
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
            "days_total": 1,
            "scenario_comparable_days_count": 2,
            "scenario_non_comparable_days_count": 0,
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 1,
            "comparable_fills_total": 1,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 0.0,
        },
        {
            "window_id": "window_000",
            "scenario_id": "s2",
            "days_total": 1,
            "scenario_comparable_days_count": 1,
            "scenario_non_comparable_days_count": 0,
            "comparable_days_with_trades": 1,
            "comparable_blocked_days": 0,
            "comparable_fills_total": 5,
            "comparable_orders_place_total": 1,
            "comparable_estimated_cost_total": 0.0,
        },
    ]
    summary = aggregate_walkforward(
        windows[:1],
        scenario_window,
        scenarios_total=2,
        min_comparable_days_per_window=1,
        min_comparable_days_global=2,
    )
    assert summary["global_winner_scenario_id"] == "s1"
    assert summary["global_winner_status"] == "ok"
    assert summary["eligible_global_scenarios_count"] == 1
    g1 = next(r for r in summary["global_rankings"] if r["scenario_id"] == "s1")
    g2 = next(r for r in summary["global_rankings"] if r["scenario_id"] == "s2")
    assert g1["eligible_for_global_winner"] is True
    assert g1["is_global_winner"] is True
    assert g1["global_winner_status"] == "ok"
    assert g2["eligible_for_global_winner"] is False
    assert g2["is_global_winner"] is False
    assert g2["global_winner_status"] == "insufficient_scenario_comparable_days_global"
