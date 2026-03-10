from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _top_items(counter_like: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    c = Counter()
    if isinstance(counter_like, dict):
        for k, v in counter_like.items():
            key = str(k or "").strip()
            if not key:
                continue
            try:
                c[key] += int(v or 0)
            except Exception:
                continue
    return [{"reason": k, "count": int(v)} for k, v in c.most_common(max(1, int(limit or 1)))]


def build_report_summary(
    walkforward_summary: Dict[str, Any],
    walkforward_rankings: Dict[str, Any],
    scenario_compare: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    wf = walkforward_summary if isinstance(walkforward_summary, dict) else {}
    wr = walkforward_rankings if isinstance(walkforward_rankings, dict) else {}
    sc = scenario_compare if isinstance(scenario_compare, dict) else {}
    generated_at = datetime.now(timezone.utc).isoformat()

    windows = wf.get("windows", []) if isinstance(wf.get("windows"), list) else []
    scenarios = wf.get("scenarios", []) if isinstance(wf.get("scenarios"), list) else []
    global_rankings = wr.get("global_rankings", []) if isinstance(wr.get("global_rankings"), list) else []
    if not global_rankings:
        global_rankings = wf.get("global_rankings", []) if isinstance(wf.get("global_rankings"), list) else []

    scenario_by_id: Dict[str, Dict[str, Any]] = {}
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scenario_id", "")).strip()
        if sid:
            scenario_by_id[sid] = row

    global_winner_id = wf.get("global_winner_scenario_id")
    global_winner_status = str(wf.get("global_winner_status", "") or "")
    winner_row = scenario_by_id.get(str(global_winner_id or "").strip(), {})
    winner_rank_row = {}
    for row in global_rankings:
        if isinstance(row, dict) and str(row.get("scenario_id", "")).strip() == str(global_winner_id or "").strip():
            winner_rank_row = row
            break

    # Scenario-aware comparability view.
    scenario_metadata_status_counts = Counter()
    eligible_scenarios = []
    ineligible_scenarios = []
    for row in global_rankings:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scenario_id", "")).strip()
        scenario_row = scenario_by_id.get(sid, {})
        for k, v in (scenario_row.get("scenario_metadata_status_counts", {}) or {}).items():
            key = str(k or "").strip()
            if key:
                try:
                    scenario_metadata_status_counts[key] += int(v or 0)
                except Exception:
                    pass
        if bool(row.get("eligible_for_global_winner", False)):
            eligible_scenarios.append(sid)
        else:
            ineligible_scenarios.append(
                {
                    "scenario_id": sid,
                    "status": str(row.get("global_winner_status", "") or "insufficient_scenario_comparable_days_global"),
                    "scenario_comparable_days_total": int(row.get("comparable_test_days_total", 0) or 0),
                }
            )

    # Reasons aggregate from walk-forward scenarios.
    global_reason_counts = Counter()
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        rc = row.get("reason_counts", {})
        if isinstance(rc, dict):
            for k, v in rc.items():
                key = str(k or "").strip()
                if not key:
                    continue
                try:
                    global_reason_counts[key] += int(v or 0)
                except Exception:
                    continue

    window_winners = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        window_winners.append(
            {
                "window_id": str(w.get("window_id", "")).strip(),
                "train_start": w.get("train_start"),
                "train_end": w.get("train_end"),
                "test_start": w.get("test_start"),
                "test_end": w.get("test_end"),
                "winner_scenario_id": w.get("winner_scenario_id"),
                "winner_score": w.get("winner_score"),
                "winner_status": w.get("winner_status"),
                "eligible_scenarios_count": int(w.get("eligible_scenarios_count", 0) or 0),
            }
        )

    scenario_rankings = []
    for row in global_rankings:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scenario_id", "")).strip()
        srow = scenario_by_id.get(sid, {})
        scenario_rankings.append(
            {
                "scenario_id": sid,
                "eligible_for_global_winner": bool(row.get("eligible_for_global_winner", False)),
                "rank_global": row.get("rank_global"),
                "score_total": row.get("score_total"),
                "scenario_comparable_days_total": int(row.get("comparable_test_days_total", 0) or 0),
                "scenario_non_comparable_days_total": int(
                    srow.get("non_comparable_test_days_count", srow.get("scenario_non_comparable_days_count", 0)) or 0
                ),
                "days_with_trades_total": int(srow.get("days_with_trades", 0) or 0),
                "fills_total": int(srow.get("fills_total", 0) or 0),
                "estimated_cost_total": float(srow.get("estimated_cost_total", 0.0) or 0.0),
                "global_winner_status": row.get("global_winner_status"),
            }
        )

    executive_summary = {
        "report_generated_at": generated_at,
        "windows_total": int(wf.get("windows_total", 0) or 0),
        "scenarios_total": int(wf.get("scenarios_total", 0) or 0),
        "test_days_total": int(wf.get("test_days_total", 0) or 0),
        "scenario_aware_comparable_test_days_total": int(wf.get("comparable_test_days_total", 0) or 0),
        "scenario_aware_non_comparable_test_days_total": int(wf.get("non_comparable_test_days_total", 0) or 0),
        "global_winner_scenario_id": global_winner_id,
        "global_winner_status": global_winner_status,
        "eligible_global_scenarios_count": int(wf.get("eligible_global_scenarios_count", 0) or 0),
    }

    global_winner_summary = {
        "global_winner_scenario_id": global_winner_id,
        "global_winner_score_total": wf.get("global_winner_score_total"),
        "global_winner_status": global_winner_status,
        "scenario_comparable_days_total": int(winner_rank_row.get("comparable_test_days_total", 0) or 0),
        "days_with_trades_total": int(winner_row.get("days_with_trades", 0) or 0),
        "fills_total": int(winner_row.get("fills_total", 0) or 0),
        "estimated_cost_total": float(winner_row.get("estimated_cost_total", 0.0) or 0.0),
        "top_reason_counts": _top_items(winner_row.get("reason_counts", {}), limit=5),
    }

    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "input_sources": {
            "walkforward_summary": "walkforward_summary.json",
            "walkforward_rankings": "walkforward_rankings.json",
            "scenario_compare": "scenario_compare.json" if sc else None,
        },
        "executive_summary": executive_summary,
        "global_winner": global_winner_summary,
        "window_winners": window_winners,
        "scenario_rankings": scenario_rankings,
        "comparability_summary": {
            "scenario_metadata_status_counts": dict(
                sorted(scenario_metadata_status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "eligible_scenarios": eligible_scenarios,
            "ineligible_scenarios": ineligible_scenarios,
        },
        "reason_summary": {
            "global_reason_counts": dict(sorted(global_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "global_reason_counts_top": _top_items(dict(global_reason_counts), limit=8),
        },
        "scenario_compare_bridge": {
            "available": bool(sc),
            "scenarios_total": int(sc.get("scenarios_total", 0) or 0) if sc else 0,
            "date_from": sc.get("date_from") if sc else None,
            "date_to": sc.get("date_to") if sc else None,
        },
    }
    return summary


def render_markdown_report(summary: Dict[str, Any]) -> str:
    exec_s = summary.get("executive_summary", {}) if isinstance(summary.get("executive_summary"), dict) else {}
    gw = summary.get("global_winner", {}) if isinstance(summary.get("global_winner"), dict) else {}
    windows = summary.get("window_winners", []) if isinstance(summary.get("window_winners"), list) else []
    rankings = summary.get("scenario_rankings", []) if isinstance(summary.get("scenario_rankings"), list) else []
    comp = summary.get("comparability_summary", {}) if isinstance(summary.get("comparability_summary"), dict) else {}
    reason = summary.get("reason_summary", {}) if isinstance(summary.get("reason_summary"), dict) else {}

    lines: List[str] = []
    lines.append("# Walk-Forward Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Generated At: `{exec_s.get('report_generated_at')}`")
    lines.append(f"- Windows Total: `{exec_s.get('windows_total')}`")
    lines.append(f"- Scenarios Total: `{exec_s.get('scenarios_total')}`")
    lines.append(f"- Test Days Total: `{exec_s.get('test_days_total')}`")
    lines.append(
        f"- Scenario-aware Comparable Test Days Total: `{exec_s.get('scenario_aware_comparable_test_days_total')}`"
    )
    lines.append(
        f"- Scenario-aware Non-Comparable Test Days Total: `{exec_s.get('scenario_aware_non_comparable_test_days_total')}`"
    )
    lines.append(f"- Global Winner: `{exec_s.get('global_winner_scenario_id')}`")
    lines.append(f"- Global Winner Status: `{exec_s.get('global_winner_status')}`")
    lines.append(f"- Eligible Global Scenarios: `{exec_s.get('eligible_global_scenarios_count')}`")
    lines.append("")

    lines.append("## Global Winner Summary")
    lines.append(f"- Scenario: `{gw.get('global_winner_scenario_id')}`")
    lines.append(f"- Score Total: `{gw.get('global_winner_score_total')}`")
    lines.append(f"- Status: `{gw.get('global_winner_status')}`")
    lines.append(f"- Comparable Days Total: `{gw.get('scenario_comparable_days_total')}`")
    lines.append(f"- Days With Trades Total: `{gw.get('days_with_trades_total')}`")
    lines.append(f"- Fills Total: `{gw.get('fills_total')}`")
    lines.append(f"- Estimated Cost Total: `{gw.get('estimated_cost_total')}`")
    top_reasons = gw.get("top_reason_counts", [])
    if isinstance(top_reasons, list) and top_reasons:
        lines.append("- Top Reasons:")
        for item in top_reasons:
            if not isinstance(item, dict):
                continue
            lines.append(f"  - `{item.get('reason')}`: `{item.get('count')}`")
    lines.append("")

    lines.append("## Window Winners")
    lines.append("| window_id | train | test | winner | winner_score | status | eligible_scenarios_count |")
    lines.append("|---|---|---|---|---:|---|---:|")
    for row in windows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| `{row.get('window_id')}` | `{row.get('train_start')} -> {row.get('train_end')}` | "
            f"`{row.get('test_start')} -> {row.get('test_end')}` | `{row.get('winner_scenario_id')}` | "
            f"`{row.get('winner_score')}` | `{row.get('winner_status')}` | `{row.get('eligible_scenarios_count')}` |"
        )
    lines.append("")

    lines.append("## Scenario Ranking Summary")
    lines.append(
        "| scenario_id | eligible_for_global_winner | rank_global | score_total | "
        "scenario_comparable_days_total | scenario_non_comparable_days_total | days_with_trades_total | fills_total | estimated_cost_total |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rankings:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| `{row.get('scenario_id')}` | `{row.get('eligible_for_global_winner')}` | `{row.get('rank_global')}` | "
            f"`{row.get('score_total')}` | `{row.get('scenario_comparable_days_total')}` | "
            f"`{row.get('scenario_non_comparable_days_total')}` | `{row.get('days_with_trades_total')}` | "
            f"`{row.get('fills_total')}` | `{row.get('estimated_cost_total')}` |"
        )
    lines.append("")

    lines.append("## Comparability Summary")
    lines.append("- Scenario Metadata Status Counts:")
    status_counts = comp.get("scenario_metadata_status_counts", {})
    if isinstance(status_counts, dict) and status_counts:
        for k, v in status_counts.items():
            lines.append(f"  - `{k}`: `{v}`")
    else:
        lines.append("  - `unavailable`")
    lines.append(f"- Eligible Scenarios: `{comp.get('eligible_scenarios')}`")
    lines.append(f"- Ineligible Scenarios: `{comp.get('ineligible_scenarios')}`")
    lines.append("")

    lines.append("## Reason Summary")
    top_global = reason.get("global_reason_counts_top", [])
    if isinstance(top_global, list) and top_global:
        for item in top_global:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('reason')}`: `{item.get('count')}`")
    else:
        lines.append("- `unavailable`")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export minimal walk-forward report (Step 3D).")
    p.add_argument("--walkforward-dir", required=True, help="Directory containing walkforward_summary.json")
    p.add_argument(
        "--scenario-compare",
        default="",
        help="Optional scenario_compare.json path for bridge metadata.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <walkforward-dir>/report)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    wf_dir = os.path.abspath(str(args.walkforward_dir or "").strip())
    if not wf_dir or not os.path.isdir(wf_dir):
        print("[REPORT_EXPORT] invalid --walkforward-dir")
        return 1

    wf_summary_path = os.path.join(wf_dir, "walkforward_summary.json")
    wf_rankings_path = os.path.join(wf_dir, "walkforward_rankings.json")
    wf_summary = _safe_read_json(wf_summary_path)
    wf_rankings = _safe_read_json(wf_rankings_path)
    if not wf_summary:
        print("[REPORT_EXPORT] walkforward_summary.json not found or invalid")
        return 1

    sc_path = str(args.scenario_compare or "").strip()
    scenario_compare = _safe_read_json(sc_path) if sc_path else {}

    out_dir = (
        os.path.abspath(str(args.output_dir).strip())
        if str(args.output_dir or "").strip()
        else os.path.join(wf_dir, "report")
    )
    _ensure_dir(out_dir)

    summary = build_report_summary(wf_summary, wf_rankings, scenario_compare)
    markdown = render_markdown_report(summary)

    md_path = os.path.join(out_dir, "walkforward_report.md")
    json_path = os.path.join(out_dir, "walkforward_report_summary.json")
    _write_text(md_path, markdown)
    _write_json(json_path, summary)

    print(f"[REPORT_EXPORT] markdown={md_path}")
    print(f"[REPORT_EXPORT] summary={json_path}")
    print(
        "[REPORT_EXPORT_SUMMARY] "
        f"global_winner={summary.get('executive_summary', {}).get('global_winner_scenario_id')} "
        f"status={summary.get('executive_summary', {}).get('global_winner_status')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

