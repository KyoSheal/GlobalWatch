from __future__ import annotations

from tools.export_walkforward_report import build_report_summary, render_markdown_report


def test_build_report_summary_and_markdown_minimal():
    wf_summary = {
        "windows_total": 1,
        "scenarios_total": 2,
        "test_days_total": 2,
        "comparable_test_days_total": 2,
        "non_comparable_test_days_total": 0,
        "global_winner_scenario_id": "s1",
        "global_winner_score_total": 12.0,
        "global_winner_status": "ok",
        "eligible_global_scenarios_count": 1,
        "windows": [
            {
                "window_id": "window_000",
                "train_start": "2026-03-01",
                "train_end": "2026-03-02",
                "test_start": "2026-03-03",
                "test_end": "2026-03-03",
                "winner_scenario_id": "s1",
                "winner_score": 12.0,
                "winner_status": "ok",
                "eligible_scenarios_count": 1,
            }
        ],
        "scenarios": [
            {
                "scenario_id": "s1",
                "days_with_trades": 1,
                "fills_total": 3,
                "estimated_cost_total": 1.2,
                "reason_counts": {"traded": 1},
                "scenario_metadata_status_counts": {"ok": 2},
                "non_comparable_test_days_count": 0,
            },
            {
                "scenario_id": "s2",
                "days_with_trades": 0,
                "fills_total": 0,
                "estimated_cost_total": 0.0,
                "reason_counts": {"MARKET_CLOSED": 1},
                "scenario_metadata_status_counts": {"scenario_metadata_missing": 2},
                "non_comparable_test_days_count": 2,
            },
        ],
    }
    wf_rankings = {
        "global_rankings": [
            {
                "scenario_id": "s1",
                "eligible_for_global_winner": True,
                "rank_global": 1,
                "score_total": 12.0,
                "comparable_test_days_total": 2,
                "global_winner_status": "ok",
            },
            {
                "scenario_id": "s2",
                "eligible_for_global_winner": False,
                "rank_global": None,
                "score_total": None,
                "comparable_test_days_total": 0,
                "global_winner_status": "insufficient_scenario_comparable_days_global",
            },
        ]
    }
    scenario_compare = {"scenarios_total": 2, "date_from": "2026-03-01", "date_to": "2026-03-03"}

    summary = build_report_summary(wf_summary, wf_rankings, scenario_compare)
    assert summary["schema_version"] == 1
    assert summary["executive_summary"]["global_winner_scenario_id"] == "s1"
    assert summary["global_winner"]["global_winner_status"] == "ok"
    assert len(summary["window_winners"]) == 1
    assert len(summary["scenario_rankings"]) == 2
    assert summary["comparability_summary"]["eligible_scenarios"] == ["s1"]
    assert summary["comparability_summary"]["ineligible_scenarios"][0]["scenario_id"] == "s2"

    md = render_markdown_report(summary)
    assert "## Executive Summary" in md
    assert "## Global Winner Summary" in md
    assert "## Window Winners" in md
    assert "## Comparability Summary" in md
    assert "`s1`" in md

