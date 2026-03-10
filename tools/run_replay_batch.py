from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_trading import replay_bundle_once  # noqa: E402
from paper_trading import _apply_replay_risk_overrides  # noqa: E402


ET_ZONE = ZoneInfo("America/New_York")
RISK_MODEL_OVERRIDE_KEYS = {
    "rc_limit",
    "portfolio_cov_rc_hysteresis_band",
    "portfolio_cov_rc_abort_buffer_enabled",
    "portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts",
    "portfolio_cov_rc_abort_buffer_relax_delta",
    "portfolio_cov_rc_abort_buffer_active_cycles",
}


@dataclass
class BundleRecord:
    date_et: str
    bundle_dir: str
    run_dir: str
    manifest_path: str
    run_id: str
    created_ts: str


@dataclass
class ScenarioSpec:
    scenario_id: str
    risk_profile: str | None
    risk_model_overrides: Dict[str, Any]


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_iso_to_utc(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slugify_id(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return s.strip("_") or "scenario"


def _normalize_risk_model_overrides(overrides: Any) -> Dict[str, Any]:
    src = overrides if isinstance(overrides, dict) else {}
    out: Dict[str, Any] = {}
    for raw_k, raw_v in src.items():
        k = str(raw_k or "").strip()
        if k not in RISK_MODEL_OVERRIDE_KEYS:
            continue
        out[k] = raw_v
    return out


def _derive_date_from_bundle(bundle_dir: str, manifest: Dict[str, Any]) -> Optional[str]:
    expected = _safe_read_json(os.path.join(bundle_dir, "expected", "snapshot_key_fields.json"))
    ts = expected.get("timestamp")
    dt = _parse_iso_to_utc(ts)
    if dt is None:
        dt = _parse_iso_to_utc(manifest.get("created_ts"))
    if dt is None:
        return None
    return dt.astimezone(ET_ZONE).date().isoformat()


def discover_bundle_records(search_root: str, include_test_runs: bool = False) -> List[BundleRecord]:
    pattern = os.path.join(search_root, "**", "replay_bundle", "manifest.json")
    manifests = glob.glob(pattern, recursive=True)
    out: List[BundleRecord] = []
    for manifest_path in manifests:
        norm = os.path.normpath(manifest_path).lower()
        if not include_test_runs and f"{os.sep}test{os.sep}" in norm:
            continue
        bundle_dir = os.path.dirname(manifest_path)
        run_dir = os.path.dirname(bundle_dir)
        manifest = _safe_read_json(manifest_path)
        date_et = _derive_date_from_bundle(bundle_dir, manifest)
        if not date_et:
            continue
        out.append(
            BundleRecord(
                date_et=date_et,
                bundle_dir=bundle_dir,
                run_dir=run_dir,
                manifest_path=manifest_path,
                run_id=str(manifest.get("run_id", "") or ""),
                created_ts=str(manifest.get("created_ts", "") or ""),
            )
        )
    return out


def _bundle_sort_key(rec: BundleRecord) -> tuple:
    dt = _parse_iso_to_utc(rec.created_ts)
    epoch = dt.timestamp() if dt is not None else 0.0
    return (epoch, rec.bundle_dir)


def choose_latest_bundle_per_date(records: Iterable[BundleRecord]) -> Dict[str, BundleRecord]:
    chosen: Dict[str, BundleRecord] = {}
    for rec in records:
        prev = chosen.get(rec.date_et)
        if prev is None or _bundle_sort_key(rec) > _bundle_sort_key(prev):
            chosen[rec.date_et] = rec
    return chosen


def _parse_date_token(token: str) -> date:
    return date.fromisoformat(str(token).strip())


def resolve_requested_dates(args: argparse.Namespace, available_dates: List[str]) -> List[str]:
    tokens: List[str] = []
    if isinstance(args.date, str) and args.date.strip():
        tokens.append(args.date.strip())
    if isinstance(args.dates, str) and args.dates.strip():
        tokens.extend([x.strip() for x in args.dates.split(",") if x.strip()])
    if args.start_date and args.end_date:
        start = _parse_date_token(args.start_date)
        end = _parse_date_token(args.end_date)
        if end < start:
            start, end = end, start
        cursor = start
        while cursor <= end:
            tokens.append(cursor.isoformat())
            cursor += timedelta(days=1)

    if not tokens:
        return sorted(set(available_dates))

    valid: List[str] = []
    seen = set()
    for t in tokens:
        d = _parse_date_token(t).isoformat()
        if d not in seen:
            seen.add(d)
            valid.append(d)
    return valid


def _top_reason_from_skip_reasons(skip_reasons: Any) -> str:
    if not isinstance(skip_reasons, dict) or not skip_reasons:
        return ""
    best_key = ""
    best_val = -1.0
    for k, v in skip_reasons.items():
        key = str(k or "").strip()
        val = _to_float(v, 0.0)
        if key and val > best_val:
            best_key = key
            best_val = val
    return best_key


def _top_reason_from_blockers(blockers: Any) -> str:
    if not isinstance(blockers, list) or not blockers:
        return ""
    best_reason = ""
    best_count = -1.0
    for row in blockers:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "")).strip()
        count = _to_float(row.get("count"), 0.0)
        if reason and count > best_count:
            best_reason = reason
            best_count = count
    return best_reason


def choose_primary_reason(snapshot: Dict[str, Any]) -> str:
    rebalance_skipped_reason = str(snapshot.get("rebalance_skipped_reason", "")).strip()
    if rebalance_skipped_reason:
        return rebalance_skipped_reason

    risk_gate = snapshot.get("risk_gate_decision", {}) if isinstance(snapshot.get("risk_gate_decision"), dict) else {}
    risk_reason = str(risk_gate.get("reason", "")).strip()
    if risk_reason:
        return risk_reason

    no_trade = snapshot.get("no_trade_summary", {}) if isinstance(snapshot.get("no_trade_summary"), dict) else {}
    gate_reason = str(no_trade.get("gate_reason", "")).strip()
    if gate_reason:
        return gate_reason

    execution = snapshot.get("execution_summary", {}) if isinstance(snapshot.get("execution_summary"), dict) else {}
    skip_reason = _top_reason_from_skip_reasons(execution.get("skip_reasons"))
    if skip_reason:
        return skip_reason

    blocker_reason = _top_reason_from_blockers(no_trade.get("top_blockers"))
    if blocker_reason:
        return blocker_reason

    orders_place = _to_int(execution.get("orders_place"), 0)
    fills_count = _to_int(snapshot.get("fills_count_today"), orders_place)
    if orders_place > 0 or fills_count > 0:
        return "traded"
    if _to_int(execution.get("orders_skip"), 0) > 0:
        return "no_orders"
    return "unknown"


def _resolve_effective_param(
    snapshot: Dict[str, Any],
    scenario: ScenarioSpec,
    *,
    cfg: Dict[str, Any],
    cfg_key: str,
    snapshot_key: str,
    scenario_key: str,
    fallback_default: Any = None,
) -> tuple[Any, str]:
    if cfg_key in cfg and cfg.get(cfg_key) is not None:
        return cfg.get(cfg_key), "replay_snapshot_risk_model_config"
    if snapshot_key in snapshot and snapshot.get(snapshot_key) is not None:
        return snapshot.get(snapshot_key), "replay_snapshot"
    if scenario_key in scenario.risk_model_overrides:
        return scenario.risk_model_overrides.get(scenario_key), "scenario_override_fallback"
    return fallback_default, "unavailable"


def _build_scenario_expected_snapshot(bundle_rec: BundleRecord | None, scenario: ScenarioSpec) -> Dict[str, Any]:
    if bundle_rec is None:
        return {}
    expected_path = os.path.join(bundle_rec.bundle_dir, "expected", "snapshot_key_fields.json")
    expected_fields = _safe_read_json(expected_path)
    if not isinstance(expected_fields, dict) or not expected_fields:
        return {}
    try:
        scenario_expected = _apply_replay_risk_overrides(
            copy.deepcopy(expected_fields),
            scenario_id=scenario.scenario_id,
            risk_profile_override=scenario.risk_profile,
            risk_model_overrides=scenario.risk_model_overrides,
        )
    except Exception:
        return {}
    return scenario_expected if isinstance(scenario_expected, dict) else {}


def extract_daily_result(
    target_date: str,
    bundle_rec: BundleRecord | None,
    scenario: ScenarioSpec,
    replay_status: str,
    drift_summary: Dict[str, Any] | None,
    error: str | None = None,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    drift_report: Dict[str, Any] = {}
    drift_report_exists = False
    if bundle_rec is not None:
        replay_snapshot_path = os.path.join(bundle_rec.bundle_dir, "outputs", "replay_snapshot.json")
        snapshot = _safe_read_json(replay_snapshot_path)
        drift_report_path = os.path.join(bundle_rec.bundle_dir, "outputs", "drift_report.json")
        if os.path.exists(drift_report_path):
            drift_report_exists = True
            drift_report = _safe_read_json(drift_report_path)

    execution = snapshot.get("execution_summary", {}) if isinstance(snapshot.get("execution_summary"), dict) else {}
    risk_gate = snapshot.get("risk_gate_decision", {}) if isinstance(snapshot.get("risk_gate_decision"), dict) else {}
    cost_summary = snapshot.get("cost_summary", {}) if isinstance(snapshot.get("cost_summary"), dict) else {}
    cost_totals = cost_summary.get("totals", {}) if isinstance(cost_summary.get("totals"), dict) else {}
    risk_model_cfg = (
        snapshot.get("effective_risk_model_config", {})
        if isinstance(snapshot.get("effective_risk_model_config"), dict)
        else {}
    )
    risk_model_schema_version = snapshot.get("effective_risk_model_config_schema_version")
    risk_model_fingerprint = snapshot.get("effective_risk_model_config_fingerprint")
    if risk_model_fingerprint is not None:
        risk_model_fingerprint = str(risk_model_fingerprint).strip() or None
    if risk_model_schema_version is not None and risk_model_fingerprint:
        risk_model_metadata_source = "replay_snapshot_metadata"
    elif risk_model_cfg:
        risk_model_metadata_source = "legacy_snapshot_missing_metadata"
    else:
        risk_model_metadata_source = "unavailable"

    orders_place = _to_int(execution.get("orders_place"), 0)
    orders_skip = _to_int(execution.get("orders_skip"), 0)
    fills_count = _to_int(snapshot.get("fills_count_today"), orders_place)
    turnover = snapshot.get("turnover_notional_post")
    if turnover is None:
        turnover = snapshot.get("turnover_notional_pre")
    turnover_val = _to_float(turnover, 0.0) if turnover is not None else None

    total_cost = cost_totals.get("total")
    if total_cost is None and isinstance(snapshot.get("estimated_cost"), (int, float, str)):
        total_cost = snapshot.get("estimated_cost")
    total_cost_val = _to_float(total_cost, 0.0) if total_cost is not None else None

    final_gate_decision = str(risk_gate.get("final_gate_decision", "")).strip().upper()
    if not final_gate_decision:
        if str(snapshot.get("rebalance_skipped_reason", "")).strip().startswith("risk_gate:"):
            final_gate_decision = "ABORT"
        elif orders_place > 0:
            final_gate_decision = "ALLOW"
        elif orders_skip > 0:
            final_gate_decision = "SKIP"
        else:
            final_gate_decision = ""

    primary_reason = choose_primary_reason(snapshot) if snapshot else ("no_bundle" if bundle_rec is None else "unknown")
    drift_obj = drift_summary if isinstance(drift_summary, dict) else {}
    if not drift_obj and isinstance(drift_report.get("summary"), dict):
        drift_obj = drift_report.get("summary", {})
    sev = drift_obj.get("severity_counts", {}) if isinstance(drift_obj.get("severity_counts"), dict) else {}

    drift_meta = drift_report.get("config_metadata_compare", {}) if isinstance(drift_report.get("config_metadata_compare"), dict) else {}
    if bundle_rec is None:
        config_metadata_status = "drift_report_missing"
    elif not drift_report_exists or not isinstance(drift_report, dict) or not drift_report:
        config_metadata_status = "drift_report_missing"
    elif not drift_meta:
        config_metadata_status = "metadata_compare_missing"
    else:
        config_metadata_status = str(drift_meta.get("status", "")).strip() or "metadata_compare_missing"

    scenario_expected_snapshot = _build_scenario_expected_snapshot(bundle_rec, scenario)
    scenario_schema_expected = (
        scenario_expected_snapshot.get("effective_risk_model_config_schema_version")
        if isinstance(scenario_expected_snapshot, dict)
        else None
    )
    scenario_fp_expected = (
        scenario_expected_snapshot.get("effective_risk_model_config_fingerprint")
        if isinstance(scenario_expected_snapshot, dict)
        else None
    )
    scenario_schema_actual = risk_model_schema_version
    scenario_fp_actual = risk_model_fingerprint
    if scenario_fp_expected is not None:
        scenario_fp_expected = str(scenario_fp_expected).strip() or None
    if scenario_fp_actual is not None:
        scenario_fp_actual = str(scenario_fp_actual).strip() or None

    scenario_schema_match: Optional[bool]
    scenario_fingerprint_match: Optional[bool]
    if scenario_schema_expected is None or scenario_schema_actual is None:
        scenario_schema_match = None
    else:
        scenario_schema_match = bool(scenario_schema_expected == scenario_schema_actual)
    if scenario_fp_expected is None or scenario_fp_actual is None:
        scenario_fingerprint_match = None
    else:
        scenario_fingerprint_match = bool(str(scenario_fp_expected) == str(scenario_fp_actual))

    if bundle_rec is None:
        scenario_metadata_status = "scenario_expected_unavailable"
    elif scenario_schema_actual is None or not scenario_fp_actual:
        scenario_metadata_status = "scenario_metadata_missing"
    elif scenario_schema_expected is None or not scenario_fp_expected:
        scenario_metadata_status = "scenario_expected_unavailable"
    elif scenario_schema_match is not True:
        scenario_metadata_status = "scenario_schema_version_mismatch"
    elif scenario_fingerprint_match is not True:
        scenario_metadata_status = "scenario_effective_risk_model_fingerprint_mismatch"
    else:
        scenario_metadata_status = "ok"
    scenario_comparable_day = bool(scenario_metadata_status == "ok")

    effective_rc_limit, effective_rc_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="rc_limit",
        snapshot_key="effective_rc_limit",
        scenario_key="rc_limit",
        fallback_default=None,
    )
    if effective_rc_limit is None:
        threshold = risk_gate.get("threshold")
        if threshold is not None:
            effective_rc_limit = threshold
            effective_rc_source = "risk_gate_threshold"
    effective_hyst, effective_hyst_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="portfolio_cov_rc_hysteresis_band",
        snapshot_key="effective_hysteresis_band",
        scenario_key="portfolio_cov_rc_hysteresis_band",
        fallback_default=None,
    )
    if effective_hyst is None:
        hyst = risk_gate.get("hysteresis_band")
        if hyst is not None:
            effective_hyst = hyst
            effective_hyst_source = "risk_gate_hysteresis_band"
    effective_abort_buffer_enabled, effective_abort_buffer_enabled_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="portfolio_cov_rc_abort_buffer_enabled",
        snapshot_key="effective_abort_buffer_enabled",
        scenario_key="portfolio_cov_rc_abort_buffer_enabled",
        fallback_default=False,
    )
    effective_abort_buffer_trigger, effective_abort_buffer_trigger_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts",
        snapshot_key="effective_abort_buffer_trigger_consecutive_aborts",
        scenario_key="portfolio_cov_rc_abort_buffer_trigger_consecutive_aborts",
        fallback_default=None,
    )
    effective_abort_buffer_delta, effective_abort_buffer_delta_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="portfolio_cov_rc_abort_buffer_relax_delta",
        snapshot_key="effective_abort_buffer_relax_delta",
        scenario_key="portfolio_cov_rc_abort_buffer_relax_delta",
        fallback_default=None,
    )
    effective_abort_buffer_cycles, effective_abort_buffer_cycles_source = _resolve_effective_param(
        snapshot,
        scenario,
        cfg=risk_model_cfg,
        cfg_key="portfolio_cov_rc_abort_buffer_active_cycles",
        snapshot_key="effective_abort_buffer_active_cycles",
        scenario_key="portfolio_cov_rc_abort_buffer_active_cycles",
        fallback_default=None,
    )
    effective_param_sources = {
        effective_rc_source,
        effective_hyst_source,
        effective_abort_buffer_enabled_source,
        effective_abort_buffer_trigger_source,
        effective_abort_buffer_delta_source,
        effective_abort_buffer_cycles_source,
    }
    effective_param_sources = {s for s in effective_param_sources if s and s != "unavailable"}
    if not effective_param_sources:
        effective_param_source = "unavailable"
    elif len(effective_param_sources) == 1:
        effective_param_source = list(effective_param_sources)[0]
    else:
        effective_param_source = "mixed:" + ",".join(sorted(effective_param_sources))

    return {
        "scenario_id": scenario.scenario_id,
        "date": target_date,
        "replay_status": replay_status,
        "risk_profile_requested": scenario.risk_profile,
        "risk_model_overrides": dict(scenario.risk_model_overrides),
        "effective_risk_profile": str(snapshot.get("effective_risk_profile", "")).strip().lower() or None,
        "effective_risk_model_source": str(snapshot.get("effective_risk_model_source", "")).strip() or None,
        "effective_risk_model_config_schema_version": risk_model_schema_version,
        "effective_risk_model_config_fingerprint": risk_model_fingerprint,
        "effective_risk_model_metadata_source": risk_model_metadata_source,
        "active_risk_profile": str(snapshot.get("active_risk_profile", "")).strip().lower() or None,
        "requested_risk_profile": str(snapshot.get("requested_risk_profile", "")).strip().lower() or None,
        "risk_profile_source": str(snapshot.get("risk_profile_source", "")).strip() or None,
        "final_gate_decision": final_gate_decision or None,
        "primary_reason": primary_reason,
        "rebalance_skipped_reason": str(snapshot.get("rebalance_skipped_reason", "")).strip() or None,
        "orders_place": int(orders_place),
        "orders_skip": int(orders_skip),
        "fills_count": int(fills_count),
        "turnover": turnover_val,
        "estimated_cost": total_cost_val,
        "effective_rc_limit": _to_float(effective_rc_limit, 0.0) if effective_rc_limit is not None else None,
        "effective_hysteresis_band": _to_float(effective_hyst, 0.0) if effective_hyst is not None else None,
        "effective_abort_buffer_enabled": bool(effective_abort_buffer_enabled),
        "effective_abort_buffer_trigger_consecutive_aborts": (
            _to_int(effective_abort_buffer_trigger, 0) if effective_abort_buffer_trigger is not None else None
        ),
        "effective_abort_buffer_relax_delta": (
            _to_float(effective_abort_buffer_delta, 0.0) if effective_abort_buffer_delta is not None else None
        ),
        "effective_abort_buffer_active_cycles": (
            _to_int(effective_abort_buffer_cycles, 0) if effective_abort_buffer_cycles is not None else None
        ),
        "effective_rc_limit_source": effective_rc_source,
        "effective_hysteresis_band_source": effective_hyst_source,
        "effective_param_source": effective_param_source,
        "config_metadata_compare_status": config_metadata_status,
        "config_metadata_schema_version_expected": drift_meta.get("effective_risk_model_config_schema_version_expected"),
        "config_metadata_schema_version_actual": drift_meta.get("effective_risk_model_config_schema_version_actual"),
        "config_metadata_fingerprint_expected": drift_meta.get("effective_risk_model_config_fingerprint_expected"),
        "config_metadata_fingerprint_actual": drift_meta.get("effective_risk_model_config_fingerprint_actual"),
        "config_metadata_schema_version_match": drift_meta.get("schema_version_match"),
        "config_metadata_fingerprint_match": drift_meta.get("fingerprint_match"),
        "scenario_metadata_compare_status": scenario_metadata_status,
        "scenario_metadata_schema_version_expected": scenario_schema_expected,
        "scenario_metadata_schema_version_actual": scenario_schema_actual,
        "scenario_metadata_fingerprint_expected": scenario_fp_expected,
        "scenario_metadata_fingerprint_actual": scenario_fp_actual,
        "scenario_metadata_schema_version_match": scenario_schema_match,
        "scenario_metadata_fingerprint_match": scenario_fingerprint_match,
        "scenario_comparable_day": scenario_comparable_day,
        "snapshot_timestamp": str(snapshot.get("timestamp", "")).strip() or None,
        "drift_pass": bool(drift_obj.get("pass", False)),
        "drift_num_diffs": _to_int(drift_obj.get("num_diffs"), 0),
        "drift_critical": _to_int(sev.get("CRITICAL"), 0),
        "drift_major": _to_int(sev.get("MAJOR"), 0),
        "drift_minor": _to_int(sev.get("MINOR"), 0),
        "source_run_dir": bundle_rec.run_dir if bundle_rec is not None else None,
        "bundle_dir": bundle_rec.bundle_dir if bundle_rec is not None else None,
        "error": error,
    }


def aggregate_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counts = Counter()
    metadata_status_counts = Counter()
    scenario_metadata_status_counts = Counter()
    days_with_trades = 0
    scenario_comparable_days_count = 0
    fills_total = 0
    orders_place_total = 0
    orders_skip_total = 0
    estimated_cost_total = 0.0
    turnover_total = 0.0

    for row in results:
        reason_counts[str(row.get("primary_reason", "unknown") or "unknown")] += 1
        metadata_status = str(row.get("config_metadata_compare_status", "") or "metadata_compare_missing").strip()
        metadata_status_counts[metadata_status] += 1
        scenario_metadata_status = str(
            row.get("scenario_metadata_compare_status", "") or "scenario_metadata_missing"
        ).strip()
        scenario_metadata_status_counts[scenario_metadata_status] += 1
        if bool(row.get("scenario_comparable_day", False)):
            scenario_comparable_days_count += 1
        orders_place = _to_int(row.get("orders_place"), 0)
        fills = _to_int(row.get("fills_count"), 0)
        orders_skip = _to_int(row.get("orders_skip"), 0)
        if orders_place > 0 or fills > 0:
            days_with_trades += 1
        fills_total += fills
        orders_place_total += orders_place
        orders_skip_total += orders_skip
        if row.get("estimated_cost") is not None:
            estimated_cost_total += _to_float(row.get("estimated_cost"), 0.0)
        if row.get("turnover") is not None:
            turnover_total += _to_float(row.get("turnover"), 0.0)

    days_total = len(results)
    days_pass = sum(1 for x in results if str(x.get("replay_status", "")).upper() == "PASS")
    days_fail = days_total - days_pass
    comparable_days_count = int(metadata_status_counts.get("ok", 0))
    non_comparable_days_count = int(max(0, days_total - comparable_days_count))
    scenario_non_comparable_days_count = int(max(0, days_total - scenario_comparable_days_count))
    return {
        "schema_version": 1,
        "days_total": int(days_total),
        "days_pass": int(days_pass),
        "days_fail": int(days_fail),
        "days_with_trades": int(days_with_trades),
        "days_without_trades": int(days_total - days_with_trades),
        "fills_total": int(fills_total),
        "orders_place_total": int(orders_place_total),
        "orders_skip_total": int(orders_skip_total),
        "turnover_total": float(turnover_total),
        "estimated_cost_total": float(estimated_cost_total),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "config_metadata_status_counts": dict(sorted(metadata_status_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "comparable_days_count": comparable_days_count,
        "non_comparable_days_count": non_comparable_days_count,
        "scenario_metadata_status_counts": dict(
            sorted(scenario_metadata_status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "scenario_comparable_days_count": int(scenario_comparable_days_count),
        "scenario_non_comparable_days_count": int(scenario_non_comparable_days_count),
    }


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


def _default_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="default",
        risk_profile=None,
        risk_model_overrides={},
    )


def load_scenarios(scenario_file: str | None) -> List[ScenarioSpec]:
    if not scenario_file:
        return [_default_scenario()]
    payload = _safe_read_json(scenario_file)
    rows = payload.get("scenarios", []) if isinstance(payload.get("scenarios"), list) else []
    if not rows:
        raise ValueError(f"No scenarios found in {scenario_file}")

    out: List[ScenarioSpec] = []
    seen = set()
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        sid = _slugify_id(str(row.get("scenario_id", "") or f"scenario_{idx}"))
        if sid in seen:
            suffix = 2
            base = sid
            while f"{base}_{suffix}" in seen:
                suffix += 1
            sid = f"{base}_{suffix}"
        seen.add(sid)

        risk_profile = str(row.get("risk_profile", "") or "").strip().lower() or None
        overrides = _normalize_risk_model_overrides(row.get("risk_model_overrides", {}))
        out.append(
            ScenarioSpec(
                scenario_id=sid,
                risk_profile=risk_profile,
                risk_model_overrides=overrides,
            )
        )
    if not out:
        raise ValueError(f"No valid scenarios parsed from {scenario_file}")
    return out


def run_scenario_batch(
    *,
    scenario: ScenarioSpec,
    requested_dates: List[str],
    by_date: Dict[str, BundleRecord],
    replay_level: str | None,
    out_dir: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for d in requested_dates:
        rec = by_date.get(d)
        if rec is None:
            row = extract_daily_result(
                target_date=d,
                bundle_rec=None,
                scenario=scenario,
                replay_status="FAIL",
                drift_summary=None,
                error="no_bundle_for_date",
            )
            results.append(row)
            print(f"[BATCH_DAY] scenario={scenario.scenario_id} date={d} status=FAIL reason=no_bundle_for_date")
            continue

        error = None
        replay_status = "FAIL"
        drift_summary: Dict[str, Any] | None = None
        try:
            drift = replay_bundle_once(
                rec.bundle_dir,
                replay_level=replay_level,
                scenario_id=scenario.scenario_id,
                risk_profile_override=scenario.risk_profile,
                risk_model_overrides=scenario.risk_model_overrides,
            )
            drift_obj = drift if isinstance(drift, dict) else {}
            drift_summary = drift_obj.get("summary", {}) if isinstance(drift_obj.get("summary"), dict) else {}
            replay_status = "PASS"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            replay_status = "FAIL"

        row = extract_daily_result(
            target_date=d,
            bundle_rec=rec,
            scenario=scenario,
            replay_status=replay_status,
            drift_summary=drift_summary,
            error=error,
        )
        results.append(row)
        print(
            "[BATCH_DAY] "
            f"scenario={scenario.scenario_id} date={d} status={replay_status} "
            f"reason={row.get('primary_reason')} orders_place={row.get('orders_place')} fills={row.get('fills_count')}"
        )

    summary = aggregate_batch_results(results)
    summary["scenario_id"] = scenario.scenario_id
    summary["risk_profile_requested"] = scenario.risk_profile
    summary["risk_model_overrides"] = dict(scenario.risk_model_overrides)
    summary["date_from"] = min(requested_dates) if requested_dates else None
    summary["date_to"] = max(requested_dates) if requested_dates else None
    summary["requested_dates"] = list(requested_dates)
    summary["resolved_dates_with_bundle"] = sorted([d for d in requested_dates if d in by_date])
    summary["output_dir"] = out_dir
    return results, summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch replay runner (Step 3A + Step 3B scenario compare).")
    p.add_argument("--search-root", type=str, default="outputs", help="Root directory to scan replay bundles.")
    p.add_argument("--include-test-runs", action="store_true", help="Include bundles under outputs/test.")
    p.add_argument("--date", type=str, default="", help="Single trading date YYYY-MM-DD.")
    p.add_argument("--dates", type=str, default="", help="Comma separated trading dates.")
    p.add_argument("--start-date", type=str, default="", help="Range start YYYY-MM-DD.")
    p.add_argument("--end-date", type=str, default="", help="Range end YYYY-MM-DD.")
    p.add_argument("--output-dir", type=str, default="", help="Output directory for batch artifacts.")
    p.add_argument("--replay-level", type=str, default=None, choices=["L0", "L1", "l0", "l1"], help="Optional replay level override.")
    p.add_argument("--scenario-file", type=str, default="", help="Scenario compare JSON file path.")
    p.add_argument("--write-csv", action="store_true", help="Also write CSV outputs.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    records = discover_bundle_records(args.search_root, include_test_runs=bool(args.include_test_runs))
    by_date = choose_latest_bundle_per_date(records)
    available_dates = sorted(by_date.keys())
    if not available_dates:
        print("[BATCH_REPLAY] no replay_bundle manifests discovered")
        return 1

    requested_dates = resolve_requested_dates(args, available_dates)
    if not requested_dates:
        print("[BATCH_REPLAY] no target dates resolved")
        return 1

    scenarios = load_scenarios(args.scenario_file.strip() if isinstance(args.scenario_file, str) else "")
    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = (
        args.output_dir.strip()
        if isinstance(args.output_dir, str) and args.output_dir.strip()
        else os.path.join("outputs", "scenario_compare", now_tag)
    )
    _ensure_dir(out_root)
    print(
        "[BATCH_REPLAY] "
        f"dates={len(requested_dates)} available={len(available_dates)} "
        f"scenarios={len(scenarios)} output={out_root}"
    )

    compare_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        scenario_dir = os.path.join(out_root, scenario.scenario_id)
        _ensure_dir(scenario_dir)
        print(
            "[SCENARIO] "
            f"id={scenario.scenario_id} risk_profile={scenario.risk_profile} "
            f"overrides={json.dumps(scenario.risk_model_overrides, ensure_ascii=False)}"
        )
        results, summary = run_scenario_batch(
            scenario=scenario,
            requested_dates=requested_dates,
            by_date=by_date,
            replay_level=args.replay_level,
            out_dir=scenario_dir,
        )
        _write_jsonl(os.path.join(scenario_dir, "daily_results.jsonl"), results)
        _write_json(os.path.join(scenario_dir, "batch_summary.json"), summary)
        if bool(args.write_csv):
            _write_csv(
                os.path.join(scenario_dir, "daily_results.csv"),
                results,
                fields=[
                    "scenario_id",
                    "date",
                    "replay_status",
                    "risk_profile_requested",
                    "effective_risk_profile",
                    "effective_risk_model_config_schema_version",
                    "effective_risk_model_config_fingerprint",
                    "effective_risk_model_metadata_source",
                    "active_risk_profile",
                    "final_gate_decision",
                    "primary_reason",
                    "orders_place",
                    "orders_skip",
                    "fills_count",
                    "turnover",
                    "estimated_cost",
                    "effective_rc_limit",
                    "effective_hysteresis_band",
                    "effective_abort_buffer_enabled",
                    "effective_abort_buffer_trigger_consecutive_aborts",
                    "effective_abort_buffer_relax_delta",
                    "effective_abort_buffer_active_cycles",
                    "effective_param_source",
                    "config_metadata_compare_status",
                    "config_metadata_schema_version_expected",
                    "config_metadata_schema_version_actual",
                    "config_metadata_fingerprint_expected",
                    "config_metadata_fingerprint_actual",
                    "config_metadata_schema_version_match",
                    "config_metadata_fingerprint_match",
                    "scenario_metadata_compare_status",
                    "scenario_metadata_schema_version_expected",
                    "scenario_metadata_schema_version_actual",
                    "scenario_metadata_fingerprint_expected",
                    "scenario_metadata_fingerprint_actual",
                    "scenario_metadata_schema_version_match",
                    "scenario_metadata_fingerprint_match",
                    "scenario_comparable_day",
                    "error",
                ],
            )
        compare_rows.append(summary)
        print(
            "[SCENARIO_SUMMARY] "
            f"id={scenario.scenario_id} days_total={summary.get('days_total')} "
            f"days_with_trades={summary.get('days_with_trades')} "
            f"reason_counts={json.dumps(summary.get('reason_counts', {}), ensure_ascii=False)}"
        )

    compare_payload = {
        "schema_version": 1,
        "scenarios_total": len(compare_rows),
        "date_from": min(requested_dates),
        "date_to": max(requested_dates),
        "requested_dates": requested_dates,
        "scenarios": compare_rows,
        "output_dir": out_root,
    }
    compare_json_path = os.path.join(out_root, "scenario_compare.json")
    _write_json(compare_json_path, compare_payload)
    if bool(args.write_csv):
        _write_csv(
            os.path.join(out_root, "scenario_compare.csv"),
            compare_rows,
            fields=[
                "scenario_id",
                "days_total",
                "days_pass",
                "days_fail",
                "days_with_trades",
                "days_without_trades",
                "fills_total",
                "orders_place_total",
                "orders_skip_total",
                "turnover_total",
                "estimated_cost_total",
                "comparable_days_count",
                "non_comparable_days_count",
                "scenario_comparable_days_count",
                "scenario_non_comparable_days_count",
                "config_metadata_status_counts",
                "scenario_metadata_status_counts",
                "reason_counts",
            ],
        )

    print(
        "[SCENARIO_COMPARE] "
        f"scenarios_total={len(compare_rows)} "
        f"date_from={compare_payload['date_from']} date_to={compare_payload['date_to']}"
    )
    print(f"[SCENARIO_COMPARE_OUTPUT] {compare_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
