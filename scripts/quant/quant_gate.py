#!/usr/bin/env python3
"""A1-5 Quant regression gate helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_compare import (
    compare_two_runs,
    load_metrics_and_daily,
    render_compare_markdown,
    write_delta_daily_csv,
)
from quant_io_utils import to_iso_utc

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


DEFAULT_RULES: Dict[str, Any] = {
    "sharpe_drop_max": 0.30,
    "max_dd_worsen_max": 0.05,
    "calmar_drop_max": 0.25,
    "total_return_drop_max": 0.03,
    "turnover_ratio_increase_max": 0.25,
    "trades_total_spike_max": 2.5,
    "gating_ratio_limit": 0.70,
}


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_or_generate_metrics(
    dataset_dir: Path,
    *,
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    verbose: bool,
) -> Dict[str, Any]:
    metrics_path = dataset_dir / "metrics" / "metrics.json"
    daily_path = dataset_dir / "metrics" / "daily_returns.csv"
    warnings: List[str] = []
    generated = False
    rc = 0

    if (not metrics_path.exists() or not daily_path.exists()) and auto_metrics:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a2_compute_metrics.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--out-dir",
            str((dataset_dir / "metrics").resolve()),
            "--report-tz",
            str(report_tz),
            "--annualization",
            str(int(annualization)),
            "--rf",
            str(float(rf)),
            "--min-points",
            str(int(min_points)),
        ]
        if verbose:
            cmd.append("--verbose")
        proc = subprocess.run(cmd, cwd=str(ROOT))
        rc = int(proc.returncode)
        generated = proc.returncode == 0
        if proc.returncode != 0:
            warnings.append(f"auto_metrics_failed_rc_{proc.returncode}")

    metrics, daily, quality = load_metrics_and_daily(
        dataset_dir,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
    )
    if not metrics_path.exists():
        warnings.append("missing_metrics_json")
    if not daily_path.exists():
        warnings.append("missing_daily_returns_csv")

    return {
        "dataset_dir": str(dataset_dir),
        "metrics": metrics,
        "daily": daily,
        "quality": quality,
        "warnings": warnings,
        "generated": generated,
        "auto_metrics_rc": rc,
    }


def compute_gating_ratio(dataset_dir: Path) -> Dict[str, Any]:
    cycles_path = dataset_dir / "cycles.csv"
    counts: Dict[str, int] = {}
    total = 0
    gated = 0
    if not cycles_path.exists():
        return {
            "ratio": None,
            "total_cycles": 0,
            "gated_cycles": 0,
            "top_reasons": [],
        }

    import csv

    try:
        with cycles_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                total += 1
                reason = (
                    str(row.get("skip_reason", "") or "").strip()
                    or str(row.get("cov_gate_reason", "") or "").strip()
                    or str(row.get("decision_path", "") or "").strip()
                )
                if reason:
                    gated += 1
                    k = reason.lower()
                    counts[k] = counts.get(k, 0) + 1
    except Exception:
        return {
            "ratio": None,
            "total_cycles": 0,
            "gated_cycles": 0,
            "top_reasons": [],
        }

    top = sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))[:3]
    ratio = (gated / total) if total > 0 else None
    return {
        "ratio": ratio,
        "total_cycles": total,
        "gated_cycles": gated,
        "top_reasons": [{"reason": r, "count": c} for r, c in top],
    }


def _max_dd_abs(v: Any) -> Optional[float]:
    d = _num_or_none(v)
    if d is None:
        return None
    return abs(min(0.0, float(d)))


def evaluate_gate(
    baseline_bundle: Dict[str, Any],
    candidate_bundle: Dict[str, Any],
    *,
    rules: Dict[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    b = baseline_bundle.get("metrics", {}) or {}
    c = candidate_bundle.get("metrics", {}) or {}
    pb = b.get("performance", {}) or {}
    pc = c.get("performance", {}) or {}
    rb = b.get("risk", {}) or {}
    rc = c.get("risk", {}) or {}
    tb = b.get("trading", {}) or {}
    tc = c.get("trading", {}) or {}
    dqb = b.get("data_quality", {}) or {}
    dqc = c.get("data_quality", {}) or {}

    fail_rules: List[Dict[str, Any]] = []
    warnings: List[str] = []

    sharpe_b = _num_or_none(rb.get("sharpe"))
    sharpe_c = _num_or_none(rc.get("sharpe"))
    if sharpe_b is not None and sharpe_c is not None:
        if sharpe_c < sharpe_b - float(rules.get("sharpe_drop_max", DEFAULT_RULES["sharpe_drop_max"])):
            fail_rules.append({"rule": "sharpe_drop_max", "baseline": sharpe_b, "candidate": sharpe_c})

    dd_b = _max_dd_abs(rb.get("max_drawdown"))
    dd_c = _max_dd_abs(rc.get("max_drawdown"))
    if dd_b is not None and dd_c is not None:
        if dd_c > dd_b + float(rules.get("max_dd_worsen_max", DEFAULT_RULES["max_dd_worsen_max"])):
            fail_rules.append({"rule": "max_dd_worsen_max", "baseline_abs_dd": dd_b, "candidate_abs_dd": dd_c})

    calmar_b = _num_or_none(rb.get("calmar"))
    calmar_c = _num_or_none(rc.get("calmar"))
    if calmar_b is not None and calmar_c is not None:
        if calmar_c < calmar_b - float(rules.get("calmar_drop_max", DEFAULT_RULES["calmar_drop_max"])):
            fail_rules.append({"rule": "calmar_drop_max", "baseline": calmar_b, "candidate": calmar_c})

    ret_b = _num_or_none(pb.get("total_return"))
    ret_c = _num_or_none(pc.get("total_return"))
    if ret_b is not None and ret_c is not None:
        if ret_c < ret_b - float(rules.get("total_return_drop_max", DEFAULT_RULES["total_return_drop_max"])):
            fail_rules.append({"rule": "total_return_drop_max", "baseline": ret_b, "candidate": ret_c})

    trn_b = _num_or_none(tb.get("turnover_ratio"))
    trn_c = _num_or_none(tc.get("turnover_ratio"))
    if trn_b is not None and trn_c is not None:
        if trn_c > trn_b + float(rules.get("turnover_ratio_increase_max", DEFAULT_RULES["turnover_ratio_increase_max"])):
            fail_rules.append({"rule": "turnover_ratio_increase_max", "baseline": trn_b, "candidate": trn_c})

    ntr_b = _num_or_none(tb.get("trades_total"))
    ntr_c = _num_or_none(tc.get("trades_total"))
    if ntr_b is not None and ntr_c is not None and ntr_b > 0:
        mult = float(rules.get("trades_total_spike_max", DEFAULT_RULES["trades_total_spike_max"]))
        if ntr_c > ntr_b * mult:
            fail_rules.append({"rule": "trades_total_spike_max", "baseline": ntr_b, "candidate": ntr_c, "multiplier": mult})

    gb = baseline_bundle.get("gating_ratio", {}) or {}
    gc = candidate_bundle.get("gating_ratio", {}) or {}
    ratio_b = _num_or_none(gb.get("ratio"))
    ratio_c = _num_or_none(gc.get("ratio"))
    gate_limit = float(rules.get("gating_ratio_limit", DEFAULT_RULES["gating_ratio_limit"]))
    if ratio_b is not None and ratio_c is not None:
        if ratio_c > gate_limit and ratio_b <= gate_limit:
            fail_rules.append({"rule": "gating_ratio_limit", "baseline_ratio": ratio_b, "candidate_ratio": ratio_c, "limit": gate_limit})

    if bool(dqc.get("insufficient_points", False)):
        warnings.append("candidate_insufficient_points")
    if int(len(dqc.get("missing_files", []) or [])) > 0:
        warnings.append("candidate_missing_files")
    pw_c = dqc.get("parse_warnings", {}) or {}
    parse_count = 0
    if isinstance(pw_c, dict):
        for v in pw_c.values():
            try:
                parse_count += int(v)
            except Exception:
                continue
    if parse_count > 0:
        warnings.append("candidate_parse_warnings")

    if strict and warnings:
        for w in warnings:
            fail_rules.append({"rule": f"strict_{w}"})

    status = "PASS" if len(fail_rules) == 0 else "FAIL"
    return {
        "status": status,
        "fail_rules": fail_rules,
        "warnings": warnings,
        "delta": {
            "sharpe": None if sharpe_b is None or sharpe_c is None else sharpe_c - sharpe_b,
            "total_return": None if ret_b is None or ret_c is None else ret_c - ret_b,
            "max_drawdown_abs": None if dd_b is None or dd_c is None else dd_c - dd_b,
            "calmar": None if calmar_b is None or calmar_c is None else calmar_c - calmar_b,
            "turnover_ratio": None if trn_b is None or trn_c is None else trn_c - trn_b,
            "trades_total": None if ntr_b is None or ntr_c is None else ntr_c - ntr_b,
            "gating_ratio": None if ratio_b is None or ratio_c is None else ratio_c - ratio_b,
        },
    }


def render_gate_report_md(
    *,
    baseline_bundle: Dict[str, Any],
    candidate_bundle: Dict[str, Any],
    gate_eval: Dict[str, Any],
) -> str:
    b = baseline_bundle.get("metrics", {}) or {}
    c = candidate_bundle.get("metrics", {}) or {}
    pb = b.get("performance", {}) or {}
    pc = c.get("performance", {}) or {}
    rb = b.get("risk", {}) or {}
    rc = c.get("risk", {}) or {}
    tb = b.get("trading", {}) or {}
    tc = c.get("trading", {}) or {}
    gb = baseline_bundle.get("gating_ratio", {}) or {}
    gc = candidate_bundle.get("gating_ratio", {}) or {}

    lines: List[str] = []
    lines.append("# Quant Gate Report")
    lines.append("")
    lines.append(f"- status: **{gate_eval.get('status', 'FAIL')}**")
    lines.append("")
    lines.append("| metric | baseline | candidate |")
    lines.append("|---|---:|---:|")
    for k, bv, cv in [
        ("total_return", pb.get("total_return"), pc.get("total_return")),
        ("sharpe", rb.get("sharpe"), rc.get("sharpe")),
        ("max_drawdown", rb.get("max_drawdown"), rc.get("max_drawdown")),
        ("calmar", rb.get("calmar"), rc.get("calmar")),
        ("turnover_ratio", tb.get("turnover_ratio"), tc.get("turnover_ratio")),
        ("trades_total", tb.get("trades_total"), tc.get("trades_total")),
        ("gating_ratio", gb.get("ratio"), gc.get("ratio")),
    ]:
        lines.append(f"| {k} | {bv} | {cv} |")
    lines.append("")
    lines.append("## Fail Rules")
    if gate_eval.get("fail_rules"):
        for fr in gate_eval["fail_rules"]:
            lines.append(f"- {fr.get('rule')}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    if gate_eval.get("warnings"):
        for w in gate_eval["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Gating Top3")
    for who, gx in [("baseline", gb), ("candidate", gc)]:
        lines.append(f"- {who}:")
        top = gx.get("top_reasons", []) or []
        if top:
            for item in top:
                lines.append(f"  - {item.get('reason','')}: {item.get('count',0)}")
        else:
            lines.append("  - none")
    lines.append("")
    return "\n".join(lines)


def write_gate_outputs(
    *,
    out_dir: Path,
    gate_result: Dict[str, Any],
    gate_report_md: str,
    gate_compare_md: str,
    delta_daily_rows: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "gate_result.json", gate_result)
    _write_text(out_dir / "gate_report.md", gate_report_md)
    _write_text(out_dir / "gate_compare.md", gate_compare_md)
    write_delta_daily_csv(out_dir / "gate_delta_daily_returns.csv", delta_daily_rows)


def run_gate(
    *,
    baseline_dir: Path,
    candidate_dirs: List[Path],
    out_dir: Optional[Path],
    auto_metrics: bool,
    report_tz: str,
    annualization: int,
    rf: float,
    min_points: int,
    rules: Dict[str, Any],
    strict: bool,
    verbose: bool,
) -> Tuple[int, Dict[str, Any]]:
    if not baseline_dir.exists():
        return 2, {"error": f"baseline not found: {baseline_dir}"}
    if not candidate_dirs:
        return 2, {"error": "no candidate provided"}

    baseline_bundle = load_or_generate_metrics(
        baseline_dir,
        auto_metrics=auto_metrics,
        report_tz=report_tz,
        annualization=annualization,
        rf=rf,
        min_points=min_points,
        verbose=verbose,
    )
    baseline_bundle["gating_ratio"] = compute_gating_ratio(baseline_dir)
    if baseline_bundle.get("auto_metrics_rc", 0) != 0:
        return 2, {"error": "baseline auto-metrics failed", "baseline": baseline_bundle}

    summary_items: List[Dict[str, Any]] = []
    failed_any = False
    for idx, candidate_dir in enumerate(candidate_dirs, start=1):
        cb = load_or_generate_metrics(
            candidate_dir,
            auto_metrics=auto_metrics,
            report_tz=report_tz,
            annualization=annualization,
            rf=rf,
            min_points=min_points,
            verbose=verbose,
        )
        cb["gating_ratio"] = compute_gating_ratio(candidate_dir)
        if cb.get("auto_metrics_rc", 0) != 0:
            return 2, {"error": f"candidate auto-metrics failed: {candidate_dir}", "candidate": cb}

        compare_obj, daily_delta_rows = compare_two_runs(
            dataset_a=baseline_dir,
            dataset_b=candidate_dir,
            metrics_a=baseline_bundle.get("metrics", {}) or {},
            metrics_b=cb.get("metrics", {}) or {},
            daily_a=baseline_bundle.get("daily", []) or [],
            daily_b=cb.get("daily", []) or [],
            quality_a=baseline_bundle.get("quality", {}) or {},
            quality_b=cb.get("quality", {}) or {},
            report_tz=report_tz,
            annualization=annualization,
            rf=rf,
            fail_rules=[],
        )

        gate_eval = evaluate_gate(
            baseline_bundle,
            cb,
            rules=rules,
            strict=bool(strict),
        )
        status = str(gate_eval.get("status", "FAIL"))
        failed_any = failed_any or (status != "PASS")

        candidate_name = str(candidate_dir.name or f"candidate_{idx}")
        if out_dir is None:
            target_out = candidate_dir / "gate"
        else:
            if len(candidate_dirs) == 1:
                target_out = out_dir
            else:
                target_out = out_dir / f"{idx:02d}_{candidate_name}"

        gate_result = {
            "schema_version": 1,
            "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
            "baseline_dir": str(baseline_dir),
            "candidate_dir": str(candidate_dir),
            "status": status,
            "rules": rules,
            "strict": bool(strict),
            "gate_eval": gate_eval,
            "baseline": {
                "metrics": baseline_bundle.get("metrics", {}),
                "gating_ratio": baseline_bundle.get("gating_ratio", {}),
            },
            "candidate": {
                "metrics": cb.get("metrics", {}),
                "gating_ratio": cb.get("gating_ratio", {}),
            },
            "compare": compare_obj,
        }
        gate_report_md = render_gate_report_md(
            baseline_bundle=baseline_bundle,
            candidate_bundle=cb,
            gate_eval=gate_eval,
        )
        gate_compare_md = render_compare_markdown(compare_obj)
        write_gate_outputs(
            out_dir=target_out,
            gate_result=gate_result,
            gate_report_md=gate_report_md,
            gate_compare_md=gate_compare_md,
            delta_daily_rows=daily_delta_rows,
        )

        summary_items.append(
            {
                "candidate_dir": str(candidate_dir),
                "status": status,
                "out_dir": str(target_out),
                "fail_rules": [x.get("rule") for x in gate_eval.get("fail_rules", []) if isinstance(x, dict)],
                "warnings": list(gate_eval.get("warnings", [])),
            }
        )

    summary = {
        "schema_version": 1,
        "generated_at_utc": to_iso_utc(datetime.now(timezone.utc)),
        "baseline_dir": str(baseline_dir),
        "candidates": summary_items,
    }
    if out_dir is not None and len(candidate_dirs) > 1:
        _write_json(out_dir / "gate_summary.json", summary)
        lines = ["# Gate Summary", ""]
        for item in summary_items:
            lines.append(
                f"- {item['candidate_dir']}: {item['status']} "
                f"(fails={item.get('fail_rules', [])}, warnings={item.get('warnings', [])})"
            )
        lines.append("")
        _write_text(out_dir / "gate_summary.md", "\n".join(lines))

    return (3 if failed_any else 0), summary

