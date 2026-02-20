#!/usr/bin/env python3
"""A3-3: replay-window drift gate evaluation."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


DEFAULT_RULES: Dict[str, Any] = {
    "max_weights_l1": 0.02,
    "max_abs_weight_delta": 0.01,
    "max_trade_delta_ratio": 0.02,
    "max_fail_cycle_ratio": 0.0,
    "forbid_tags_strict": [
        "NONDETERMINISM_WARNING",
        "macro_not_frozen",
        "PRICE_COVERAGE_LOW",
        "INPUT_DRIFT",
    ],
    "allow_warn_tags_nonstrict": ["REF_MISSING", "SOURCE_MISMATCH"],
}


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_replay_window(replay_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary_path = (replay_dir / "replay_window_summary.csv").resolve()
    manifest_path = (replay_dir / "replay_window_manifest.json").resolve()
    if not summary_path.exists():
        return [], {"error": f"missing_summary:{summary_path}", "summary_path": str(summary_path), "manifest_path": str(manifest_path)}

    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            c = _num_or_none(row.get("cycle"))
            if c is None:
                continue
            rows.append(
                {
                    "cycle": int(c),
                    "time_utc": str(row.get("time_utc") or ""),
                    "price_rows": int(_num_or_none(row.get("price_rows")) or 0),
                    "num_trades": int(_num_or_none(row.get("num_trades")) or 0),
                    "target_hash": str(row.get("target_hash") or ""),
                    "trades_hash": str(row.get("trades_hash") or ""),
                    "gate_fail": str(row.get("gate_fail") or "").lower() in ("1", "true", "yes"),
                    "warnings_count": int(_num_or_none(row.get("warnings_count")) or 0),
                    "ref_status": str(row.get("ref_status") or ""),
                    "attribution_tags": str(row.get("attribution_tags") or ""),
                    "weights_l1": float(_num_or_none(row.get("weights_l1")) or 0.0),
                    "trades_notional_delta": float(_num_or_none(row.get("trades_notional_delta")) or 0.0),
                    "diff_path": str(row.get("diff_path") or ""),
                    "decision_path": str(row.get("decision_path") or ""),
                }
            )

    rows.sort(key=lambda r: int(r.get("cycle") or 0))
    manifest = _read_json(manifest_path) or {}
    return rows, {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def load_cycle_diff(replay_dir: Path, row: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    cycle = int(row.get("cycle") or 0)
    rel = str(row.get("diff_path") or "").strip()
    candidates: List[Path] = []
    if rel:
        p = Path(rel)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((replay_dir / rel).resolve())
    candidates.append((replay_dir / "per_cycle" / str(cycle) / "diff.json").resolve())

    for p in candidates:
        obj = _read_json(p)
        if isinstance(obj, dict):
            return obj, str(p)
    return {}, ""


def _parse_tags(row: Dict[str, Any], diff_obj: Dict[str, Any], warning_tokens: List[str]) -> List[str]:
    tags: List[str] = []
    raw = str(row.get("attribution_tags") or "")
    if raw:
        tags.extend([t for t in raw.split("|") if t])
    diff_tags = diff_obj.get("attribution_tags")
    if isinstance(diff_tags, list):
        tags.extend([str(t) for t in diff_tags if str(t)])
    tags.extend([str(t) for t in warning_tokens if str(t)])
    return sorted(set(tags))


def _load_cycle_equity_from_replay_manifest(replay_dir: Path, cycle: int) -> Optional[float]:
    p = (replay_dir / "per_cycle" / str(cycle) / "replay_manifest.json").resolve()
    obj = _read_json(p)
    if not isinstance(obj, dict):
        return None
    snap = obj.get("snapshot") if isinstance(obj.get("snapshot"), dict) else {}
    return _num_or_none(snap.get("total_equity"))


def compute_cycle_drift(
    *,
    replay_dir: Path,
    row: Dict[str, Any],
    warning_tokens: List[str],
) -> Dict[str, Any]:
    diff_obj, diff_source = load_cycle_diff(replay_dir, row)
    weights_l1 = float(_num_or_none(((diff_obj.get("weights_diff") or {}).get("weights_l1"))) or float(row.get("weights_l1") or 0.0))

    top_deltas = ((diff_obj.get("weights_diff") or {}).get("top_deltas") or []) if isinstance(diff_obj, dict) else []
    max_abs_weight_delta = 0.0
    if isinstance(top_deltas, list):
        for item in top_deltas:
            if not isinstance(item, dict):
                continue
            a = _num_or_none(item.get("abs_delta"))
            if a is None:
                a = abs(float(_num_or_none(item.get("delta")) or 0.0))
            if a is not None:
                max_abs_weight_delta = max(max_abs_weight_delta, float(a))

    trades_notional_delta = float(_num_or_none(((diff_obj.get("trades_diff") or {}).get("notional_delta"))) or float(row.get("trades_notional_delta") or 0.0))
    equity = _load_cycle_equity_from_replay_manifest(replay_dir, int(row.get("cycle") or 0))
    if equity is None or equity <= 0:
        trades_notional_delta_ratio = 0.0
        ratio_warn = "missing_equity_for_ratio"
    else:
        trades_notional_delta_ratio = float(abs(trades_notional_delta) / float(equity))
        ratio_warn = ""

    tags = _parse_tags(row, diff_obj, warning_tokens)

    return {
        "cycle": int(row.get("cycle") or 0),
        "weights_l1": float(weights_l1),
        "max_abs_weight_delta": float(max_abs_weight_delta),
        "trades_notional_delta": float(trades_notional_delta),
        "trades_notional_delta_ratio": float(trades_notional_delta_ratio),
        "tags": tags,
        "warnings": [ratio_warn] if ratio_warn else [],
        "diff_source": diff_source,
    }


def evaluate_drift_rules(
    *,
    cycle_metrics: Dict[str, Any],
    rules: Dict[str, Any],
    strict: bool,
    fail_on_drift: bool,
) -> Dict[str, Any]:
    failed_rules: List[str] = []
    warn_rules: List[str] = []

    if float(cycle_metrics.get("weights_l1", 0.0) or 0.0) > float(rules.get("max_weights_l1", DEFAULT_RULES["max_weights_l1"])):
        failed_rules.append("max_weights_l1")
    if float(cycle_metrics.get("max_abs_weight_delta", 0.0) or 0.0) > float(rules.get("max_abs_weight_delta", DEFAULT_RULES["max_abs_weight_delta"])):
        failed_rules.append("max_abs_weight_delta")
    if float(cycle_metrics.get("trades_notional_delta_ratio", 0.0) or 0.0) > float(rules.get("max_trade_delta_ratio", DEFAULT_RULES["max_trade_delta_ratio"])):
        failed_rules.append("max_trade_delta_ratio")

    tags = set([str(t) for t in (cycle_metrics.get("tags") or [])])
    forbid = set([str(t) for t in (rules.get("forbid_tags") or [])])
    allow_warn = set([str(t) for t in (rules.get("allow_warn_tags_nonstrict") or DEFAULT_RULES["allow_warn_tags_nonstrict"])])

    hit_forbid = sorted(tags & forbid)
    if hit_forbid:
        failed_rules.extend([f"forbid_tag:{t}" for t in hit_forbid])

    if not strict:
        warn_hit = sorted(tags & allow_warn)
        if warn_hit:
            warn_rules.extend([f"warn_tag:{t}" for t in warn_hit])

    status = "PASS"
    if failed_rules:
        status = "FAIL" if fail_on_drift else "WARN"
    elif warn_rules or int(len(cycle_metrics.get("warnings") or [])) > 0:
        status = "WARN"

    return {
        "status": status,
        "failed_rules": sorted(set(failed_rules)),
        "warn_rules": sorted(set(warn_rules)),
        "evidence": {
            "weights_l1": cycle_metrics.get("weights_l1"),
            "max_abs_weight_delta": cycle_metrics.get("max_abs_weight_delta"),
            "trades_notional_delta_ratio": cycle_metrics.get("trades_notional_delta_ratio"),
            "tags": sorted(tags),
        },
    }


def aggregate_window_summary(
    *,
    cycle_results: List[Dict[str, Any]],
    rules: Dict[str, Any],
    strict: bool,
    fail_on_drift: bool,
) -> Dict[str, Any]:
    fails = [r for r in cycle_results if r.get("status") == "FAIL"]
    warns = [r for r in cycle_results if r.get("status") == "WARN"]
    tag_counter: Counter = Counter()
    for r in cycle_results:
        for t in (r.get("tags") or []):
            tag_counter[str(t)] += 1

    fail_ratio = (len(fails) / len(cycle_results)) if cycle_results else 0.0
    status = "PASS"
    if len(fails) > 0 and fail_on_drift:
        status = "FAIL"
    elif len(fails) > 0 or len(warns) > 0:
        status = "WARN"

    max_fail_cycle_ratio = float(rules.get("max_fail_cycle_ratio", DEFAULT_RULES["max_fail_cycle_ratio"]))
    if fail_ratio > max_fail_cycle_ratio and fail_on_drift:
        status = "FAIL"

    worst_weights = sorted(cycle_results, key=lambda r: (-float(r.get("weights_l1", 0.0) or 0.0), int(r.get("cycle") or 0)))[:10]
    worst_trade = sorted(cycle_results, key=lambda r: (-float(r.get("trades_notional_delta", 0.0) or 0.0), int(r.get("cycle") or 0)))[:10]

    return {
        "status": status,
        "cycles": len(cycle_results),
        "fails": len(fails),
        "warns": len(warns),
        "fail_cycle_ratio": fail_ratio,
        "tag_counts": dict(sorted(tag_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "worst_by_weights_l1": [
            {"cycle": int(r.get("cycle") or 0), "weights_l1": float(r.get("weights_l1", 0.0) or 0.0), "tags": list(r.get("tags") or [])}
            for r in worst_weights
        ],
        "worst_by_trade_delta": [
            {
                "cycle": int(r.get("cycle") or 0),
                "trades_notional_delta": float(r.get("trades_notional_delta", 0.0) or 0.0),
                "trades_notional_delta_ratio": float(r.get("trades_notional_delta_ratio", 0.0) or 0.0),
                "tags": list(r.get("tags") or []),
            }
            for r in worst_trade
        ],
    }


def render_drift_report_md(
    *,
    replay_window_dir: Path,
    out_dir: Path,
    strict: bool,
    fail_on_drift: bool,
    rules: Dict[str, Any],
    agg: Dict[str, Any],
    cycle_results: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Replay Drift Gate Report")
    lines.append("")
    lines.append(f"- replay_window_dir: `{replay_window_dir}`")
    lines.append(f"- out_dir: `{out_dir}`")
    lines.append(f"- strict: `{bool(strict)}`")
    lines.append(f"- fail_on_drift: `{bool(fail_on_drift)}`")
    lines.append(f"- generated_utc: `{_now_utc_iso()}`")
    lines.append("")

    lines.append("## Rules")
    lines.append(f"- max_weights_l1: `{rules.get('max_weights_l1')}`")
    lines.append(f"- max_abs_weight_delta: `{rules.get('max_abs_weight_delta')}`")
    lines.append(f"- max_trade_delta_ratio: `{rules.get('max_trade_delta_ratio')}`")
    lines.append(f"- max_fail_cycle_ratio: `{rules.get('max_fail_cycle_ratio')}`")
    lines.append(f"- forbid_tags: `{','.join([str(x) for x in rules.get('forbid_tags', [])])}`")
    lines.append("")

    lines.append("## Window Summary")
    lines.append(f"- status: **{agg.get('status')}**")
    lines.append(f"- cycles: `{agg.get('cycles')}`")
    lines.append(f"- fails: `{agg.get('fails')}`")
    lines.append(f"- warns: `{agg.get('warns')}`")
    lines.append(f"- fail_cycle_ratio: `{float(agg.get('fail_cycle_ratio', 0.0)):.4f}`")
    lines.append("")

    lines.append("## Tag Counts (Top5)")
    tags = list((agg.get("tag_counts") or {}).items())
    if tags:
        for k, v in tags[:5]:
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Worst Cycles (weights_l1)")
    w1 = agg.get("worst_by_weights_l1") or []
    if w1:
        lines.append("| cycle | weights_l1 | tags |")
        lines.append("|---:|---:|---|")
        for item in w1[:10]:
            lines.append(f"| {item.get('cycle')} | {float(item.get('weights_l1', 0.0)):.6f} | {'|'.join(item.get('tags') or [])} |")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Fail Cycles")
    fail_cycles = [r for r in cycle_results if r.get("status") == "FAIL"]
    if fail_cycles:
        lines.append("| cycle | failed_rules | tags |")
        lines.append("|---:|---|---|")
        for r in fail_cycles:
            lines.append(f"| {r.get('cycle')} | {';'.join(r.get('failed_rules') or [])} | {'|'.join(r.get('tags') or [])} |")
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    out_dir: Path,
    result_obj: Dict[str, Any],
    cycle_rows: List[Dict[str, Any]],
    report_md: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = (out_dir / "drift_gate_result.json").resolve()
    summary_path = (out_dir / "drift_gate_summary.csv").resolve()
    report_path = (out_dir / "drift_gate_report.md").resolve()

    _write_json_atomic(result_path, result_obj)
    _write_csv(
        summary_path,
        cycle_rows,
        [
            "cycle",
            "status",
            "weights_l1",
            "max_abs_weight_delta",
            "trades_notional_delta",
            "trades_notional_delta_ratio",
            "tags",
            "failed_rules",
        ],
    )
    _write_text(report_path, report_md)
    return {
        "result": result_path,
        "summary": summary_path,
        "report": report_path,
    }


def run_drift_gate(
    *,
    replay_window_dir: Path,
    out_dir: Path,
    strict: bool,
    fail_on_drift: bool,
    rules: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    started = _now_utc_iso()
    rows, info = load_replay_window(replay_window_dir)
    if not rows:
        result = {
            "status": "FAIL",
            "strict": bool(strict),
            "fail_on_drift": bool(fail_on_drift),
            "rules": rules,
            "window": {"cycles": 0, "fails": 0, "warns": 0},
            "tag_counts": {},
            "error": info.get("error") or "empty_replay_window",
            "generated_utc": _now_utc_iso(),
            "started_utc": started,
        }
        write_outputs(out_dir=out_dir, result_obj=result, cycle_rows=[], report_md="# Replay Drift Gate Report\n\n- input unavailable\n")
        return 2, result

    manifest = info.get("manifest") if isinstance(info.get("manifest"), dict) else {}
    warning_by_cycle: Dict[int, List[str]] = {}
    for w in (manifest.get("warnings") if isinstance(manifest.get("warnings"), list) else []):
        text = str(w or "")
        if text.startswith("cycle_") and ":" in text:
            left, token = text.split(":", 1)
            try:
                c = int(left.replace("cycle_", ""))
            except Exception:
                continue
            warning_by_cycle.setdefault(c, []).append(token)

    cycle_results: List[Dict[str, Any]] = []
    for row in rows:
        cycle = int(row.get("cycle") or 0)
        cmetrics = compute_cycle_drift(
            replay_dir=replay_window_dir,
            row=row,
            warning_tokens=warning_by_cycle.get(cycle, []),
        )
        er = evaluate_drift_rules(
            cycle_metrics=cmetrics,
            rules=rules,
            strict=bool(strict),
            fail_on_drift=bool(fail_on_drift),
        )
        cycle_results.append(
            {
                "cycle": cycle,
                "status": er.get("status"),
                "weights_l1": float(cmetrics.get("weights_l1", 0.0) or 0.0),
                "max_abs_weight_delta": float(cmetrics.get("max_abs_weight_delta", 0.0) or 0.0),
                "trades_notional_delta": float(cmetrics.get("trades_notional_delta", 0.0) or 0.0),
                "trades_notional_delta_ratio": float(cmetrics.get("trades_notional_delta_ratio", 0.0) or 0.0),
                "tags": list(cmetrics.get("tags") or []),
                "failed_rules": list(er.get("failed_rules") or []),
                "warn_rules": list(er.get("warn_rules") or []),
                "diff_source": str(cmetrics.get("diff_source") or ""),
            }
        )

    cycle_results.sort(key=lambda r: int(r.get("cycle") or 0))
    agg = aggregate_window_summary(
        cycle_results=cycle_results,
        rules=rules,
        strict=bool(strict),
        fail_on_drift=bool(fail_on_drift),
    )

    result = {
        "status": agg.get("status"),
        "strict": bool(strict),
        "fail_on_drift": bool(fail_on_drift),
        "rules": rules,
        "window": {
            "cycles": agg.get("cycles"),
            "fails": agg.get("fails"),
            "warns": agg.get("warns"),
            "fail_cycle_ratio": agg.get("fail_cycle_ratio"),
            "worst_by_weights_l1": agg.get("worst_by_weights_l1"),
            "worst_by_trade_delta": agg.get("worst_by_trade_delta"),
        },
        "tag_counts": agg.get("tag_counts"),
        "generated_utc": _now_utc_iso(),
        "started_utc": started,
        "source": {
            "replay_window_dir": str(replay_window_dir),
            "summary_path": info.get("summary_path"),
            "manifest_path": info.get("manifest_path"),
        },
    }

    report_md = render_drift_report_md(
        replay_window_dir=replay_window_dir,
        out_dir=out_dir,
        strict=bool(strict),
        fail_on_drift=bool(fail_on_drift),
        rules=rules,
        agg=agg,
        cycle_results=cycle_results,
    )

    summary_rows = []
    for r in cycle_results:
        summary_rows.append(
            {
                "cycle": int(r.get("cycle") or 0),
                "status": str(r.get("status") or ""),
                "weights_l1": f"{float(r.get('weights_l1', 0.0) or 0.0):.10f}",
                "max_abs_weight_delta": f"{float(r.get('max_abs_weight_delta', 0.0) or 0.0):.10f}",
                "trades_notional_delta": f"{float(r.get('trades_notional_delta', 0.0) or 0.0):.10f}",
                "trades_notional_delta_ratio": f"{float(r.get('trades_notional_delta_ratio', 0.0) or 0.0):.10f}",
                "tags": "|".join(r.get("tags") or []),
                "failed_rules": "|".join(r.get("failed_rules") or []),
            }
        )

    write_outputs(out_dir=out_dir, result_obj=result, cycle_rows=summary_rows, report_md=report_md)

    status = str(result.get("status") or "PASS").upper()
    if status == "PASS":
        return 0, result
    if status == "WARN":
        return 1, result
    return 3, result
