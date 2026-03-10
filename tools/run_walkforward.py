from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
for p in (ROOT_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tools.run_replay_batch import (  # type: ignore
        choose_latest_bundle_per_date,
        discover_bundle_records,
        load_scenarios,
        run_scenario_batch,
    )
except Exception:
    from run_replay_batch import (  # type: ignore
        choose_latest_bundle_per_date,
        discover_bundle_records,
        load_scenarios,
        run_scenario_batch,
    )


@dataclass
class WalkForwardWindow:
    window_id: str
    train_dates: List[str]
    test_dates: List[str]

    @property
    def train_start(self) -> str:
        return self.train_dates[0]

    @property
    def train_end(self) -> str:
        return self.train_dates[-1]

    @property
    def test_start(self) -> str:
        return self.test_dates[0]

    @property
    def test_end(self) -> str:
        return self.test_dates[-1]


SCORE_WEIGHTS: Dict[str, float] = {
    "days_with_trades": 10.0,
    "fills_total": 1.0,
    "estimated_cost_total": 0.1,
    "blocked_days": 2.0,
}
SCORE_FORMULA = (
    "10*comparable_days_with_trades + 1*comparable_fills_total "
    "- 0.1*comparable_estimated_cost_total - 2*comparable_blocked_days"
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def generate_walkforward_windows(
    available_dates: List[str],
    *,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> List[WalkForwardWindow]:
    dates = sorted(set(str(d).strip() for d in available_dates if str(d).strip()))
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be > 0")
    step = int(step_days if step_days is not None else test_days)
    if step <= 0:
        raise ValueError("step_days must be > 0")

    windows: List[WalkForwardWindow] = []
    i = 0
    wid = 0
    while i + train_days + test_days <= len(dates):
        train_dates = dates[i : i + train_days]
        test_dates = dates[i + train_days : i + train_days + test_days]
        windows.append(
            WalkForwardWindow(
                window_id=f"window_{wid:03d}",
                train_dates=train_dates,
                test_dates=test_dates,
            )
        )
        wid += 1
        i += step
    return windows


def _filter_dates_in_range(dates: List[str], start_date: str, end_date: str) -> List[str]:
    start = str(start_date).strip()
    end = str(end_date).strip()
    if end < start:
        start, end = end, start
    return [d for d in sorted(set(dates)) if start <= d <= end]


def _merge_counter(dst: Counter, src: Dict[str, Any]) -> None:
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        key = str(k or "").strip()
        if not key:
            continue
        try:
            dst[key] += int(v or 0)
        except Exception:
            continue


def summarize_comparable_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable_rows: List[Dict[str, Any]] = []
    for row in results:
        if "scenario_comparable_day" in row:
            is_comparable = bool(row.get("scenario_comparable_day", False))
        else:
            status = str(row.get("config_metadata_compare_status", "") or "").strip()
            is_comparable = bool(status == "ok")
        if is_comparable:
            comparable_rows.append(row)

    comparable_test_days_count = len(comparable_rows)
    comparable_days_with_trades = 0
    comparable_fills_total = 0
    comparable_orders_place_total = 0
    comparable_estimated_cost_total = 0.0
    comparable_reason_counts = Counter()

    for row in comparable_rows:
        orders_place = int(row.get("orders_place", 0) or 0)
        fills_count = int(row.get("fills_count", 0) or 0)
        if orders_place > 0 or fills_count > 0:
            comparable_days_with_trades += 1
        comparable_fills_total += fills_count
        comparable_orders_place_total += orders_place
        if row.get("estimated_cost") is not None:
            try:
                comparable_estimated_cost_total += float(row.get("estimated_cost", 0.0) or 0.0)
            except Exception:
                pass
        reason = str(row.get("primary_reason", "unknown") or "unknown").strip() or "unknown"
        comparable_reason_counts[reason] += 1

    comparable_blocked_days = max(0, comparable_test_days_count - comparable_days_with_trades)
    return {
        "comparable_test_days_count": int(comparable_test_days_count),
        "comparable_days_with_trades": int(comparable_days_with_trades),
        "comparable_blocked_days": int(comparable_blocked_days),
        "comparable_fills_total": int(comparable_fills_total),
        "comparable_orders_place_total": int(comparable_orders_place_total),
        "comparable_estimated_cost_total": float(comparable_estimated_cost_total),
        "comparable_reason_counts": dict(
            sorted(comparable_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }


def _coerce_comparable_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    comparable_test_days_count = int(
        row.get("scenario_comparable_days_count", row.get("comparable_days_count", 0)) or 0
    )
    comparable_days_with_trades = int(row.get("comparable_days_with_trades", 0) or 0)
    comparable_fills_total = int(row.get("comparable_fills_total", 0) or 0)
    comparable_orders_place_total = int(row.get("comparable_orders_place_total", 0) or 0)
    comparable_estimated_cost_total = float(row.get("comparable_estimated_cost_total", 0.0) or 0.0)
    comparable_blocked_days = int(
        row.get("comparable_blocked_days", max(0, comparable_test_days_count - comparable_days_with_trades)) or 0
    )
    comparable_reason_counts = (
        row.get("comparable_reason_counts", {}) if isinstance(row.get("comparable_reason_counts"), dict) else {}
    )
    return {
        "comparable_test_days_count": int(comparable_test_days_count),
        "comparable_days_with_trades": int(comparable_days_with_trades),
        "comparable_blocked_days": int(max(0, comparable_blocked_days)),
        "comparable_fills_total": int(comparable_fills_total),
        "comparable_orders_place_total": int(comparable_orders_place_total),
        "comparable_estimated_cost_total": float(comparable_estimated_cost_total),
        "comparable_reason_counts": dict(comparable_reason_counts),
    }


def _compute_score(row: Dict[str, Any]) -> Optional[float]:
    comparable_days = int(row.get("comparable_test_days_count", 0) or 0)
    if comparable_days <= 0:
        return None
    comparable_days_with_trades = int(row.get("comparable_days_with_trades", 0) or 0)
    comparable_fills_total = int(row.get("comparable_fills_total", 0) or 0)
    comparable_estimated_cost_total = float(row.get("comparable_estimated_cost_total", 0.0) or 0.0)
    comparable_blocked_days = int(
        row.get("comparable_blocked_days", max(0, comparable_days - comparable_days_with_trades)) or 0
    )
    score = (
        SCORE_WEIGHTS["days_with_trades"] * comparable_days_with_trades
        + SCORE_WEIGHTS["fills_total"] * comparable_fills_total
        - SCORE_WEIGHTS["estimated_cost_total"] * comparable_estimated_cost_total
        - SCORE_WEIGHTS["blocked_days"] * comparable_blocked_days
    )
    return round(float(score), 6)


def _sort_rank_rows(rows: List[Dict[str, Any]], rank_key: str) -> List[Dict[str, Any]]:
    ranked = [r for r in rows if r.get("score") is not None]
    ranked_sorted = sorted(
        ranked,
        key=lambda r: (
            -float(r.get("score", 0.0) or 0.0),
            -int(r.get("comparable_test_days_count", 0) or 0),
            str(r.get("scenario_id", "")),
        ),
    )
    for idx, row in enumerate(ranked_sorted, start=1):
        row[rank_key] = idx
        row["rank_status"] = "ranked"

    non_ranked = [r for r in rows if r.get("score") is None]
    for row in non_ranked:
        row[rank_key] = None
        row["rank_status"] = "insufficient_comparable_days"
    return sorted(
        rows,
        key=lambda r: (
            r.get(rank_key) is None,
            int(r.get(rank_key) or 10**9),
            str(r.get("scenario_id", "")),
        ),
    )


def compute_window_rankings(
    windows_out: List[Dict[str, Any]],
    *,
    min_comparable_days_per_window: int = 1,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    min_days = max(0, int(min_comparable_days_per_window or 0))
    for window in windows_out:
        rows: List[Dict[str, Any]] = []
        for scenario_row in window.get("scenarios", []):
            if not isinstance(scenario_row, dict):
                continue
            comparable_days = int(scenario_row.get("comparable_test_days_count", 0) or 0)
            comparable_days_with_trades = int(scenario_row.get("comparable_days_with_trades", 0) or 0)
            comparable_blocked_days = int(
                scenario_row.get("comparable_blocked_days", max(0, comparable_days - comparable_days_with_trades)) or 0
            )
            row = {
                "window_id": str(window.get("window_id", "")).strip(),
                "scenario_id": str(scenario_row.get("scenario_id", "")).strip(),
                "comparable_test_days_count": comparable_days,
                "comparable_days_with_trades": comparable_days_with_trades,
                "comparable_blocked_days": comparable_blocked_days,
                "comparable_fills_total": int(scenario_row.get("comparable_fills_total", 0) or 0),
                "comparable_orders_place_total": int(scenario_row.get("comparable_orders_place_total", 0) or 0),
                "comparable_estimated_cost_total": float(
                    scenario_row.get("comparable_estimated_cost_total", 0.0) or 0.0
                ),
            }
            row["score"] = _compute_score(row)
            row["eligible_for_window_winner"] = bool(row["comparable_test_days_count"] >= min_days)
            row["window_winner_status"] = (
                "eligible_not_selected" if row["eligible_for_window_winner"] else "insufficient_scenario_comparable_days"
            )
            rows.append(row)
        ranked_rows = _sort_rank_rows(rows, rank_key="rank")
        eligible_rows = [r for r in ranked_rows if bool(r.get("eligible_for_window_winner", False))]
        if eligible_rows:
            winner_row = sorted(
                eligible_rows,
                key=lambda r: (
                    -float(r.get("score", 0.0) or 0.0),
                    -int(r.get("comparable_test_days_count", 0) or 0),
                    str(r.get("scenario_id", "")),
                ),
            )[0]
            winner_id = str(winner_row.get("scenario_id", "")).strip()
            winner_score = winner_row.get("score")
            winner_status = "ok"
            for row in ranked_rows:
                if str(row.get("scenario_id", "")).strip() == winner_id:
                    row["window_winner_status"] = "ok"
                    row["is_window_winner"] = True
                else:
                    row["is_window_winner"] = False
        else:
            winner_id = None
            winner_score = None
            winner_status = "no_eligible_scenario"
            for row in ranked_rows:
                row["is_window_winner"] = False

        window["window_rankings"] = ranked_rows
        window["winner_scenario_id"] = winner_id
        window["winner_score"] = winner_score
        window["winner_status"] = winner_status
        window["eligible_scenarios_count"] = int(len(eligible_rows))
        all_rows.extend(ranked_rows)
    return all_rows


def compute_global_rankings(
    scenarios_out: List[Dict[str, Any]],
    *,
    min_comparable_days_global: int = 2,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    min_days = max(0, int(min_comparable_days_global or 0))
    for scenario_row in scenarios_out:
        comparable_days = int(scenario_row.get("comparable_test_days_count", 0) or 0)
        comparable_days_with_trades = int(scenario_row.get("comparable_days_with_trades", 0) or 0)
        comparable_blocked_days = int(
            scenario_row.get("comparable_blocked_days", max(0, comparable_days - comparable_days_with_trades)) or 0
        )
        row = {
            "scenario_id": str(scenario_row.get("scenario_id", "")).strip(),
            "comparable_test_days_total": comparable_days,
            "comparable_days_with_trades_total": comparable_days_with_trades,
            "comparable_blocked_days_total": comparable_blocked_days,
            "comparable_fills_total": int(scenario_row.get("comparable_fills_total", 0) or 0),
            "comparable_orders_place_total": int(scenario_row.get("comparable_orders_place_total", 0) or 0),
            "comparable_estimated_cost_total": float(scenario_row.get("comparable_estimated_cost_total", 0.0) or 0.0),
        }
        score = _compute_score(
            {
                "comparable_test_days_count": row["comparable_test_days_total"],
                "comparable_days_with_trades": row["comparable_days_with_trades_total"],
                "comparable_blocked_days": row["comparable_blocked_days_total"],
                "comparable_fills_total": row["comparable_fills_total"],
                "comparable_estimated_cost_total": row["comparable_estimated_cost_total"],
            }
        )
        row["score_total"] = score
        if score is None or row["comparable_test_days_total"] <= 0:
            row["score_avg"] = None
        else:
            row["score_avg"] = round(float(score) / max(1, int(row["comparable_test_days_total"])), 6)
        row["eligible_for_global_winner"] = bool(row["comparable_test_days_total"] >= min_days)
        row["global_winner_status"] = (
            "eligible_not_selected" if row["eligible_for_global_winner"] else "insufficient_scenario_comparable_days_global"
        )
        rows.append(row)
    ranked = _sort_rank_rows(
        [
            {
                "scenario_id": r["scenario_id"],
                "score": r["score_total"],
                "comparable_test_days_count": r["comparable_test_days_total"],
            }
            for r in rows
        ],
        rank_key="rank_global",
    )
    rank_map = {str(r.get("scenario_id", "")): r.get("rank_global") for r in ranked}
    status_map = {str(r.get("scenario_id", "")): r.get("rank_status") for r in ranked}
    for row in rows:
        sid = str(row.get("scenario_id", ""))
        row["rank_global"] = rank_map.get(sid)
        row["rank_status"] = status_map.get(sid, "insufficient_comparable_days")
        row["is_global_winner"] = False

    eligible_rows = [r for r in rows if bool(r.get("eligible_for_global_winner", False))]
    if eligible_rows:
        winner_row = sorted(
            eligible_rows,
            key=lambda r: (
                -float(r.get("score_total", 0.0) or 0.0),
                -int(r.get("comparable_test_days_total", 0) or 0),
                str(r.get("scenario_id", "")),
            ),
        )[0]
        winner_id = str(winner_row.get("scenario_id", "")).strip()
        winner_score_total = winner_row.get("score_total")
        global_winner_status = "ok"
        for row in rows:
            if str(row.get("scenario_id", "")).strip() == winner_id:
                row["is_global_winner"] = True
                row["global_winner_status"] = "ok"
    else:
        winner_id = None
        winner_score_total = None
        global_winner_status = "no_eligible_scenario"
    return sorted(
        [
            dict(
                row,
                _global_winner_scenario_id=winner_id,
                _global_winner_score_total=winner_score_total,
                _global_winner_status=global_winner_status,
                _eligible_global_scenarios_count=int(len(eligible_rows)),
            )
            for row in rows
        ],
        key=lambda r: (r.get("rank_global") is None, int(r.get("rank_global") or 10**9), str(r.get("scenario_id", ""))),
    )


def aggregate_walkforward(
    windows: List[WalkForwardWindow],
    scenario_window_summaries: List[Dict[str, Any]],
    scenarios_total: int,
    *,
    min_comparable_days_per_window: int = 1,
    min_comparable_days_global: int = 2,
) -> Dict[str, Any]:
    by_window: Dict[str, Dict[str, Any]] = {}
    by_scenario: Dict[str, Dict[str, Any]] = {}

    for row in scenario_window_summaries:
        window_id = str(row.get("window_id", "")).strip()
        scenario_id = str(row.get("scenario_id", "")).strip()
        if not window_id or not scenario_id:
            continue

        if window_id not in by_window:
            by_window[window_id] = {
                "window_id": window_id,
                "train_start": row.get("train_start"),
                "train_end": row.get("train_end"),
                "test_start": row.get("test_start"),
                "test_end": row.get("test_end"),
                "test_days_total": 0,
                "comparable_test_days_count": 0,
                "non_comparable_test_days_count": 0,
                "days_with_trades": 0,
                "fills_total": 0,
                "orders_place_total": 0,
                "estimated_cost_total": 0.0,
                "reason_counts": Counter(),
                "config_metadata_status_counts": Counter(),
                "scenario_metadata_status_counts": Counter(),
                "comparable_days_with_trades": 0,
                "comparable_blocked_days": 0,
                "comparable_fills_total": 0,
                "comparable_orders_place_total": 0,
                "comparable_estimated_cost_total": 0.0,
                "comparable_reason_counts": Counter(),
                "scenarios": [],
            }
        if scenario_id not in by_scenario:
            by_scenario[scenario_id] = {
                "scenario_id": scenario_id,
                "test_days_total": 0,
                "comparable_test_days_count": 0,
                "non_comparable_test_days_count": 0,
                "days_with_trades": 0,
                "fills_total": 0,
                "orders_place_total": 0,
                "estimated_cost_total": 0.0,
                "reason_counts": Counter(),
                "config_metadata_status_counts": Counter(),
                "scenario_metadata_status_counts": Counter(),
                "comparable_days_with_trades": 0,
                "comparable_blocked_days": 0,
                "comparable_fills_total": 0,
                "comparable_orders_place_total": 0,
                "comparable_estimated_cost_total": 0.0,
                "comparable_reason_counts": Counter(),
            }

        comp_metrics = _coerce_comparable_metrics(row)
        scenario_comp_days = int(row.get("scenario_comparable_days_count", row.get("comparable_days_count", 0)) or 0)
        scenario_non_comp_days = int(
            row.get("scenario_non_comparable_days_count", row.get("non_comparable_days_count", 0)) or 0
        )
        for bucket in (by_window[window_id], by_scenario[scenario_id]):
            bucket["test_days_total"] += int(row.get("days_total", 0) or 0)
            bucket["comparable_test_days_count"] += scenario_comp_days
            bucket["non_comparable_test_days_count"] += scenario_non_comp_days
            bucket["days_with_trades"] += int(row.get("days_with_trades", 0) or 0)
            bucket["fills_total"] += int(row.get("fills_total", 0) or 0)
            bucket["orders_place_total"] += int(row.get("orders_place_total", 0) or 0)
            bucket["estimated_cost_total"] += float(row.get("estimated_cost_total", 0.0) or 0.0)
            _merge_counter(bucket["reason_counts"], row.get("reason_counts", {}))
            _merge_counter(bucket["config_metadata_status_counts"], row.get("config_metadata_status_counts", {}))
            _merge_counter(bucket["scenario_metadata_status_counts"], row.get("scenario_metadata_status_counts", {}))
            bucket["comparable_days_with_trades"] += int(comp_metrics["comparable_days_with_trades"])
            bucket["comparable_blocked_days"] += int(comp_metrics["comparable_blocked_days"])
            bucket["comparable_fills_total"] += int(comp_metrics["comparable_fills_total"])
            bucket["comparable_orders_place_total"] += int(comp_metrics["comparable_orders_place_total"])
            bucket["comparable_estimated_cost_total"] += float(comp_metrics["comparable_estimated_cost_total"])
            _merge_counter(bucket["comparable_reason_counts"], comp_metrics["comparable_reason_counts"])
        by_window[window_id]["scenarios"].append(
            {
                "scenario_id": scenario_id,
                "test_days_total": int(row.get("days_total", 0) or 0),
                "comparable_test_days_count": scenario_comp_days,
                "non_comparable_test_days_count": scenario_non_comp_days,
                "days_with_trades": int(row.get("days_with_trades", 0) or 0),
                "fills_total": int(row.get("fills_total", 0) or 0),
                "orders_place_total": int(row.get("orders_place_total", 0) or 0),
                "estimated_cost_total": float(row.get("estimated_cost_total", 0.0) or 0.0),
                "reason_counts": dict(row.get("reason_counts", {})),
                "config_metadata_status_counts": dict(row.get("config_metadata_status_counts", {})),
                "scenario_metadata_status_counts": dict(row.get("scenario_metadata_status_counts", {})),
                "comparable_days_with_trades": int(comp_metrics["comparable_days_with_trades"]),
                "comparable_blocked_days": int(comp_metrics["comparable_blocked_days"]),
                "comparable_fills_total": int(comp_metrics["comparable_fills_total"]),
                "comparable_orders_place_total": int(comp_metrics["comparable_orders_place_total"]),
                "comparable_estimated_cost_total": float(comp_metrics["comparable_estimated_cost_total"]),
                "comparable_reason_counts": dict(comp_metrics["comparable_reason_counts"]),
            }
        )

    windows_out = []
    for w in windows:
        row = by_window.get(w.window_id, None)
        if row is None:
            windows_out.append(
                {
                    "window_id": w.window_id,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "test_days_total": 0,
                    "comparable_test_days_count": 0,
                    "non_comparable_test_days_count": 0,
                    "days_with_trades": 0,
                    "fills_total": 0,
                    "orders_place_total": 0,
                    "estimated_cost_total": 0.0,
                    "reason_counts": {},
                    "config_metadata_status_counts": {},
                    "scenario_metadata_status_counts": {},
                    "comparable_days_with_trades": 0,
                    "comparable_blocked_days": 0,
                    "comparable_fills_total": 0,
                    "comparable_orders_place_total": 0,
                    "comparable_estimated_cost_total": 0.0,
                    "comparable_reason_counts": {},
                    "scenarios": [],
                }
            )
            continue
        row_out = dict(row)
        row_out["reason_counts"] = dict(sorted(row_out["reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])))
        row_out["config_metadata_status_counts"] = dict(
            sorted(row_out["config_metadata_status_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        row_out["scenario_metadata_status_counts"] = dict(
            sorted(row_out["scenario_metadata_status_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        row_out["comparable_reason_counts"] = dict(
            sorted(row_out["comparable_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        windows_out.append(row_out)

    scenarios_out = []
    for sid in sorted(by_scenario.keys()):
        row = dict(by_scenario[sid])
        row["reason_counts"] = dict(sorted(row["reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])))
        row["config_metadata_status_counts"] = dict(
            sorted(row["config_metadata_status_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        row["scenario_metadata_status_counts"] = dict(
            sorted(row["scenario_metadata_status_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        row["comparable_reason_counts"] = dict(
            sorted(row["comparable_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        scenarios_out.append(row)

    window_rankings = compute_window_rankings(
        windows_out,
        min_comparable_days_per_window=min_comparable_days_per_window,
    )
    global_rankings = compute_global_rankings(
        scenarios_out,
        min_comparable_days_global=min_comparable_days_global,
    )
    global_winner_scenario_id = None
    global_winner_score_total = None
    global_winner_status = "no_eligible_scenario"
    eligible_global_scenarios_count = 0
    if global_rankings:
        meta_sample = global_rankings[0]
        global_winner_scenario_id = meta_sample.get("_global_winner_scenario_id")
        global_winner_score_total = meta_sample.get("_global_winner_score_total")
        global_winner_status = str(meta_sample.get("_global_winner_status", "no_eligible_scenario"))
        eligible_global_scenarios_count = int(meta_sample.get("_eligible_global_scenarios_count", 0) or 0)
        for row in global_rankings:
            row.pop("_global_winner_scenario_id", None)
            row.pop("_global_winner_score_total", None)
            row.pop("_global_winner_status", None)
            row.pop("_eligible_global_scenarios_count", None)
    test_days_total = sum(int(x.get("test_days_total", 0) or 0) for x in scenarios_out)
    comparable_test_days_total = sum(int(x.get("comparable_test_days_count", 0) or 0) for x in scenarios_out)
    non_comparable_test_days_total = sum(int(x.get("non_comparable_test_days_count", 0) or 0) for x in scenarios_out)
    return {
        "schema_version": 1,
        "score_formula": SCORE_FORMULA,
        "score_weights": dict(SCORE_WEIGHTS),
        "winner_selection": {
            "min_comparable_days_per_window": int(max(0, int(min_comparable_days_per_window or 0))),
            "min_comparable_days_global": int(max(0, int(min_comparable_days_global or 0))),
        },
        "windows_total": len(windows),
        "scenarios_total": int(scenarios_total),
        "test_days_total": int(test_days_total),
        "comparable_test_days_total": int(comparable_test_days_total),
        "non_comparable_test_days_total": int(non_comparable_test_days_total),
        "windows": windows_out,
        "scenarios": scenarios_out,
        "window_rankings": window_rankings,
        "global_rankings": global_rankings,
        "global_winner_scenario_id": global_winner_scenario_id,
        "global_winner_score_total": global_winner_score_total,
        "global_winner_status": global_winner_status,
        "eligible_global_scenarios_count": int(eligible_global_scenarios_count),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward skeleton runner (Step 3C).")
    p.add_argument("--search-root", type=str, default="outputs", help="Root directory to scan replay bundles.")
    p.add_argument("--include-test-runs", action="store_true", help="Include bundles under outputs/test.")
    p.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD.")
    p.add_argument("--train-days", type=int, required=True, help="Train window size in days (date points).")
    p.add_argument("--test-days", type=int, required=True, help="Test window size in days (date points).")
    p.add_argument("--step-days", type=int, default=0, help="Window step size; default=test-days.")
    p.add_argument("--scenario-file", type=str, default="", help="Scenario compare JSON file path.")
    p.add_argument("--replay-level", type=str, default=None, choices=["L0", "L1", "l0", "l1"], help="Optional replay level override.")
    p.add_argument("--output-dir", type=str, default="", help="Output directory.")
    p.add_argument("--write-csv", action="store_true", help="Also write CSV outputs.")
    p.add_argument(
        "--min-comparable-days-per-window",
        type=int,
        default=1,
        help="Minimum scenario-aware comparable test days within a window to be eligible for window winner.",
    )
    p.add_argument(
        "--min-comparable-days-global",
        type=int,
        default=2,
        help="Minimum scenario-aware comparable test days across all windows to be eligible for global winner.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = discover_bundle_records(args.search_root, include_test_runs=bool(args.include_test_runs))
    by_date = choose_latest_bundle_per_date(records)
    all_dates = sorted(by_date.keys())
    in_range_dates = _filter_dates_in_range(all_dates, args.start_date, args.end_date)
    if not in_range_dates:
        print("[WALKFORWARD] no replay bundle dates in requested range")
        return 1

    step_days = int(args.step_days or 0) if args.step_days is not None else 0
    windows = generate_walkforward_windows(
        in_range_dates,
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=(step_days if step_days > 0 else None),
    )
    if not windows:
        print("[WALKFORWARD] no windows generated")
        return 1

    scenarios = load_scenarios(args.scenario_file.strip() if isinstance(args.scenario_file, str) else "")
    out_root = (
        args.output_dir.strip()
        if isinstance(args.output_dir, str) and args.output_dir.strip()
        else os.path.join("outputs", "walkforward", datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    _ensure_dir(out_root)
    print(
        "[WALKFORWARD] "
        f"windows={len(windows)} scenarios={len(scenarios)} "
        f"date_points={len(in_range_dates)} output={out_root}"
    )

    scenario_window_summaries: List[Dict[str, Any]] = []
    window_scenario_rows: List[Dict[str, Any]] = []

    for window in windows:
        window_dir = os.path.join(out_root, window.window_id)
        _ensure_dir(window_dir)
        print(
            "[WINDOW] "
            f"id={window.window_id} train={window.train_start}->{window.train_end} "
            f"test={window.test_start}->{window.test_end} points={len(window.test_dates)}"
        )
        for scenario in scenarios:
            scenario_dir = os.path.join(window_dir, scenario.scenario_id)
            _ensure_dir(scenario_dir)
            results, summary = run_scenario_batch(
                scenario=scenario,
                requested_dates=window.test_dates,
                by_date=by_date,
                replay_level=args.replay_level,
                out_dir=scenario_dir,
            )
            comparable_metrics = summarize_comparable_metrics(results)
            _write_jsonl(os.path.join(scenario_dir, "daily_results.jsonl"), results)
            summary_out = dict(summary)
            summary_out["window_id"] = window.window_id
            summary_out["train_start"] = window.train_start
            summary_out["train_end"] = window.train_end
            summary_out["test_start"] = window.test_start
            summary_out["test_end"] = window.test_end
            summary_out.update(comparable_metrics)
            _write_json(os.path.join(scenario_dir, "batch_summary.json"), summary_out)
            scenario_window_summaries.append(summary_out)
            window_scenario_rows.append(
                {
                    "window_id": window.window_id,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "scenario_id": scenario.scenario_id,
                    "test_days_total": int(summary_out.get("days_total", 0) or 0),
                    "comparable_test_days_count": int(
                        summary_out.get("scenario_comparable_days_count", summary_out.get("comparable_days_count", 0))
                        or 0
                    ),
                    "non_comparable_test_days_count": int(
                        summary_out.get(
                            "scenario_non_comparable_days_count",
                            summary_out.get("non_comparable_days_count", 0),
                        )
                        or 0
                    ),
                    "days_with_trades": int(summary_out.get("days_with_trades", 0) or 0),
                    "fills_total": int(summary_out.get("fills_total", 0) or 0),
                    "orders_place_total": int(summary_out.get("orders_place_total", 0) or 0),
                    "estimated_cost_total": float(summary_out.get("estimated_cost_total", 0.0) or 0.0),
                    "reason_counts": dict(summary_out.get("reason_counts", {})),
                    "config_metadata_status_counts": dict(summary_out.get("config_metadata_status_counts", {})),
                    "scenario_metadata_status_counts": dict(summary_out.get("scenario_metadata_status_counts", {})),
                    "comparable_days_with_trades": int(summary_out.get("comparable_days_with_trades", 0) or 0),
                    "comparable_blocked_days": int(summary_out.get("comparable_blocked_days", 0) or 0),
                    "comparable_fills_total": int(summary_out.get("comparable_fills_total", 0) or 0),
                    "comparable_orders_place_total": int(summary_out.get("comparable_orders_place_total", 0) or 0),
                    "comparable_estimated_cost_total": float(
                        summary_out.get("comparable_estimated_cost_total", 0.0) or 0.0
                    ),
                    "comparable_reason_counts": dict(summary_out.get("comparable_reason_counts", {})),
                }
            )

    walkforward_summary = aggregate_walkforward(
        windows,
        scenario_window_summaries,
        scenarios_total=len(scenarios),
        min_comparable_days_per_window=int(max(0, int(args.min_comparable_days_per_window or 0))),
        min_comparable_days_global=int(max(0, int(args.min_comparable_days_global or 0))),
    )
    walkforward_summary["date_range"] = {"start_date": args.start_date, "end_date": args.end_date}
    walkforward_summary["window_config"] = {
        "train_days": int(args.train_days),
        "test_days": int(args.test_days),
        "step_days": int(step_days if step_days > 0 else args.test_days),
        "min_comparable_days_per_window": int(max(0, int(args.min_comparable_days_per_window or 0))),
        "min_comparable_days_global": int(max(0, int(args.min_comparable_days_global or 0))),
    }
    walkforward_summary["date_points"] = in_range_dates
    walkforward_summary["output_dir"] = out_root

    _write_json(os.path.join(out_root, "walkforward_summary.json"), walkforward_summary)
    ranking_payload = {
        "schema_version": 1,
        "score_formula": walkforward_summary.get("score_formula"),
        "score_weights": walkforward_summary.get("score_weights"),
        "winner_selection": walkforward_summary.get("winner_selection", {}),
        "global_winner_scenario_id": walkforward_summary.get("global_winner_scenario_id"),
        "global_winner_score_total": walkforward_summary.get("global_winner_score_total"),
        "global_winner_status": walkforward_summary.get("global_winner_status"),
        "eligible_global_scenarios_count": walkforward_summary.get("eligible_global_scenarios_count"),
        "window_rankings": walkforward_summary.get("window_rankings", []),
        "global_rankings": walkforward_summary.get("global_rankings", []),
    }
    _write_json(os.path.join(out_root, "walkforward_rankings.json"), ranking_payload)
    _write_jsonl(os.path.join(out_root, "window_scenario_results.jsonl"), window_scenario_rows)
    if bool(args.write_csv):
        _write_csv(
            os.path.join(out_root, "window_scenario_results.csv"),
            window_scenario_rows,
            fields=[
                "window_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "scenario_id",
                "test_days_total",
                "comparable_test_days_count",
                "non_comparable_test_days_count",
                "days_with_trades",
                "fills_total",
                "orders_place_total",
                "estimated_cost_total",
                "reason_counts",
                "config_metadata_status_counts",
                "scenario_metadata_status_counts",
                "comparable_days_with_trades",
                "comparable_blocked_days",
                "comparable_fills_total",
                "comparable_orders_place_total",
                "comparable_estimated_cost_total",
                "comparable_reason_counts",
            ],
        )
        _write_csv(
            os.path.join(out_root, "walkforward_window_rankings.csv"),
            ranking_payload["window_rankings"],
            fields=[
                "window_id",
                "scenario_id",
                "comparable_test_days_count",
                "comparable_days_with_trades",
                "comparable_blocked_days",
                "comparable_fills_total",
                "comparable_orders_place_total",
                "comparable_estimated_cost_total",
                "score",
                "rank",
                "rank_status",
                "eligible_for_window_winner",
                "window_winner_status",
                "is_window_winner",
            ],
        )
        _write_csv(
            os.path.join(out_root, "walkforward_global_rankings.csv"),
            ranking_payload["global_rankings"],
            fields=[
                "scenario_id",
                "comparable_test_days_total",
                "comparable_days_with_trades_total",
                "comparable_blocked_days_total",
                "comparable_fills_total",
                "comparable_orders_place_total",
                "comparable_estimated_cost_total",
                "score_total",
                "score_avg",
                "rank_global",
                "rank_status",
                "eligible_for_global_winner",
                "global_winner_status",
                "is_global_winner",
            ],
        )

    print(
        "[WALKFORWARD_SUMMARY] "
        f"windows_total={walkforward_summary.get('windows_total')} "
        f"scenarios_total={walkforward_summary.get('scenarios_total')} "
        f"test_days_total={walkforward_summary.get('test_days_total')} "
        f"comparable_test_days_total={walkforward_summary.get('comparable_test_days_total')} "
        f"non_comparable_test_days_total={walkforward_summary.get('non_comparable_test_days_total')}"
    )
    print(f"[WALKFORWARD_OUTPUT] {os.path.join(out_root, 'walkforward_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
