"""Daily report generation and aggregation utilities for paper trading."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import argparse
from collections import Counter
from datetime import date as date_cls
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from atomic_io import safe_read_json as io_safe_read_json
from cov_coverage import default_cov_coverage

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_TZ = "America/Vancouver"
MARKET_TZ = "America/New_York"
DEFAULT_MAIN_REPORT_DIR = os.path.join("outputs", "Daily Report")
INDEX_FILENAME = "daily_reports_index.json"
REPORT_SCHEMA_VERSION = 1


def get_daily_report_dir(base_out_dir: str = "outputs") -> Path:
    """Return the canonical Daily Report directory under the given base output root."""
    root = Path(str(base_out_dir or "outputs")).resolve()
    return root / "Daily Report"


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _coerce_zone(tz_name: Optional[str]):
    if ZoneInfo is None:
        return datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(str(tz_name or DEFAULT_TZ))
    except Exception:
        return ZoneInfo(DEFAULT_TZ)


def _parse_datetime(value: Any, tz_name: Optional[str] = None) -> Optional[datetime]:
    tzinfo = _coerce_zone(tz_name)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value))
        except Exception:
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except Exception:
                    continue
        if dt is None:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    try:
        return dt.astimezone(tzinfo)
    except Exception:
        return dt


def _parse_date(value: Any, tz_name: Optional[str] = None) -> Optional[date_cls]:
    if isinstance(value, date_cls) and not isinstance(value, datetime):
        return value
    dt = _parse_datetime(value, tz_name)
    if dt is not None:
        return dt.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except Exception:
            return None
    return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path) or "."
    _ensure_dir(folder)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_daily_report_", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        obj = io_safe_read_json(path, retries=2, sleep_ms=15)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _ensure_report_meta_fields(report: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
    """Ensure report metadata fields exist; return (patched_report, changed)."""
    if not isinstance(report, dict):
        return {}, False
    snapshot_obj: Dict[str, Any] = snapshot if isinstance(snapshot, dict) else {}
    patched = dict(report)
    changed = False
    if "report_schema_version" not in patched:
        patched["report_schema_version"] = int(REPORT_SCHEMA_VERSION)
        changed = True
    if not str(patched.get("generated_at") or "").strip():
        patched["generated_at"] = datetime.now(timezone.utc).isoformat()
        changed = True
    if not str(patched.get("risk_profile") or "").strip():
        patched["risk_profile"] = str(
            snapshot_obj.get("active_risk_profile")
            or snapshot_obj.get("requested_risk_profile")
            or "mid"
        ).strip().lower() or "mid"
        changed = True
    if not str(patched.get("active_risk_profile") or "").strip():
        patched["active_risk_profile"] = str(
            snapshot_obj.get("active_risk_profile")
            or patched.get("risk_profile")
            or "mid"
        ).strip().lower() or "mid"
        changed = True
    if patched.get("risk_profile_template_version", None) in (None, ""):
        patched["risk_profile_template_version"] = snapshot_obj.get("risk_profile_template_version")
        changed = True
    if patched.get("risk_profile_overrides_hash", None) in (None, ""):
        patched["risk_profile_overrides_hash"] = str(snapshot_obj.get("risk_profile_overrides_hash") or "")
        changed = True
    if not str(patched.get("risk_profile_source") or "").strip():
        patched["risk_profile_source"] = str(snapshot_obj.get("risk_profile_source") or "unknown")
        changed = True
    if not str(patched.get("last_risk_profile_change_ts") or "").strip():
        patched["last_risk_profile_change_ts"] = str(snapshot_obj.get("last_risk_profile_change_ts") or "")
        changed = True
    if not str(patched.get("last_risk_profile_change_old") or "").strip():
        patched["last_risk_profile_change_old"] = str(snapshot_obj.get("last_risk_profile_change_old") or "")
        changed = True
    if not str(patched.get("last_risk_profile_change_new") or "").strip():
        patched["last_risk_profile_change_new"] = str(snapshot_obj.get("last_risk_profile_change_new") or "")
        changed = True
    if not str(patched.get("last_risk_profile_change_source") or "").strip():
        patched["last_risk_profile_change_source"] = str(snapshot_obj.get("last_risk_profile_change_source") or "")
        changed = True
    cov_coverage = patched.get("cov_coverage")
    if not isinstance(cov_coverage, dict):
        snapshot_cov = snapshot_obj.get("cov_coverage")
        if isinstance(snapshot_cov, dict):
            patched["cov_coverage"] = dict(snapshot_cov)
        else:
            patched["cov_coverage"] = default_cov_coverage()
        changed = True
    else:
        if "schema_version" not in cov_coverage:
            cov_coverage["schema_version"] = 1
            patched["cov_coverage"] = cov_coverage

    returns_cov_diag = patched.get("returns_coverage_diag")
    if not isinstance(returns_cov_diag, dict):
        snapshot_returns_cov_diag = snapshot_obj.get("returns_coverage_diag")
        if isinstance(snapshot_returns_cov_diag, dict):
            patched["returns_coverage_diag"] = dict(snapshot_returns_cov_diag)
        else:
            patched["returns_coverage_diag"] = {"schema_version": 1, "items": []}
            changed = True
    else:
        if "schema_version" not in returns_cov_diag:
            returns_cov_diag["schema_version"] = 1
            patched["returns_coverage_diag"] = returns_cov_diag

    if "ticker_proxy_used" not in patched:
        patched["ticker_proxy_used"] = bool(snapshot_obj.get("ticker_proxy_used", False))
        changed = True
    if not isinstance(patched.get("ticker_proxy_map_used"), list):
        snapshot_proxy_rows = snapshot_obj.get("ticker_proxy_map_used")
        if isinstance(snapshot_proxy_rows, list):
            patched["ticker_proxy_map_used"] = list(snapshot_proxy_rows)
        else:
            patched["ticker_proxy_map_used"] = []
        changed = True
    if not str(patched.get("asset_policy_mode") or "").strip():
        patched["asset_policy_mode"] = str(snapshot_obj.get("asset_policy_mode") or "FORCE_PROXY")
        changed = True
    if not isinstance(patched.get("asset_policy_decisions"), list):
        snapshot_asset_policy_rows = snapshot_obj.get("asset_policy_decisions")
        if isinstance(snapshot_asset_policy_rows, list):
            patched["asset_policy_decisions"] = list(snapshot_asset_policy_rows)
        else:
            patched["asset_policy_decisions"] = []
        changed = True
    if not isinstance(patched.get("asset_policy_summary"), dict):
        snapshot_asset_policy_summary = snapshot_obj.get("asset_policy_summary")
        if isinstance(snapshot_asset_policy_summary, dict):
            patched["asset_policy_summary"] = dict(snapshot_asset_policy_summary)
        else:
            patched["asset_policy_summary"] = {
                "counts": {"ALLOW_ORIGINAL": 0, "USE_PROXY": 0, "DISABLE": 0},
                "top_reasons": [],
            }
        changed = True
    if not isinstance(patched.get("cost_summary"), dict):
        patched["cost_summary"] = _build_cost_summary(
            snapshot_obj,
            patched.get("trades", {}) if isinstance(patched.get("trades"), dict) else {},
        )
        changed = True
    if not isinstance(patched.get("performance_summary"), dict):
        patched["performance_summary"] = _build_performance_summary(
            patched.get("equity", {}) if isinstance(patched.get("equity"), dict) else {},
            patched.get("cost_summary", {}) if isinstance(patched.get("cost_summary"), dict) else {},
            patched.get("trades", {}) if isinstance(patched.get("trades"), dict) else {},
        )
        changed = True

    computed_no_trade_summary = build_no_trade_summary(
        trades=patched.get("trades", {}) if isinstance(patched.get("trades"), dict) else {},
        snapshot=snapshot_obj,
        risk_gate_decision=(
            patched.get("risk_gate_decision")
            if isinstance(patched.get("risk_gate_decision"), dict)
            else snapshot_obj.get("risk_gate_decision")
        ),
        cov_coverage=(
            patched.get("cov_coverage")
            if isinstance(patched.get("cov_coverage"), dict)
            else snapshot_obj.get("cov_coverage")
        ),
        returns_coverage_diag=(
            patched.get("returns_coverage_diag")
            if isinstance(patched.get("returns_coverage_diag"), dict)
            else snapshot_obj.get("returns_coverage_diag")
        ),
        asset_policy_mode=str(patched.get("asset_policy_mode") or snapshot_obj.get("asset_policy_mode") or "FORCE_PROXY"),
        execution_proxy_used=bool(snapshot_obj.get("execution_proxy_used", snapshot_obj.get("ticker_proxy_used", False))),
        proxy_scope=_norm_text(snapshot_obj.get("proxy_scope") or snapshot_obj.get("ticker_proxy_scope")),
    )
    existing_no_trade = patched.get("no_trade_summary")
    if not isinstance(existing_no_trade, dict):
        patched["no_trade_summary"] = computed_no_trade_summary
        changed = True
    else:
        merged_no_trade = dict(computed_no_trade_summary)
        merged_no_trade.update(existing_no_trade)
        if merged_no_trade != existing_no_trade:
            patched["no_trade_summary"] = merged_no_trade
            changed = True
    computed_health = build_risk_model_health(
        report=patched,
        snapshot=snapshot_obj,
        daily_fields={
            "no_trade_summary": patched.get("no_trade_summary"),
            "cost_summary": patched.get("cost_summary"),
            "execution_summary": patched.get("execution_summary"),
            "risk_gate_decision": patched.get("risk_gate_decision"),
            "cov_coverage": patched.get("cov_coverage"),
            "returns_coverage_diag": patched.get("returns_coverage_diag"),
            "asset_policy_mode": patched.get("asset_policy_mode"),
            "execution_proxy_used": patched.get("ticker_proxy_used"),
        },
    )
    existing_health = patched.get("risk_model_health")
    if not isinstance(existing_health, dict):
        patched["risk_model_health"] = computed_health
        changed = True
    else:
        merged_health = dict(computed_health)
        for key, value in existing_health.items():
            if isinstance(value, dict) and isinstance(merged_health.get(key), dict):
                merged_nested = dict(merged_health.get(key, {}))
                merged_nested.update(value)
                merged_health[key] = merged_nested
            else:
                merged_health[key] = value
        if merged_health != existing_health:
            patched["risk_model_health"] = merged_health
            changed = True
    return patched, changed


def _normalize_report_dirs(report_dirs: Optional[List[str]]) -> List[str]:
    raw = report_dirs or [str(get_daily_report_dir("outputs"))]
    if isinstance(raw, str):  # type: ignore[unreachable]
        raw = [raw]  # type: ignore[assignment]
    dedup: List[str] = []
    seen = set()
    for item in raw:
        if not item:
            continue
        abspath = os.path.abspath(str(item))
        key = abspath.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(abspath)
    if not dedup:
        dedup = [str(get_daily_report_dir("outputs"))]
    return dedup


def _resolve_run_reports_dir(snapshot_path: str, snapshot: Optional[Dict[str, Any]] = None) -> str:
    """Resolve canonical Daily Report dir under the snapshot's base output root."""
    snap_obj = snapshot if isinstance(snapshot, dict) else {}
    candidate_base_dir = _norm_text(snap_obj.get("base_out_dir"))
    if candidate_base_dir:
        return str(get_daily_report_dir(candidate_base_dir))
    snap_path = _norm_text(snapshot_path)
    if snap_path:
        snap_dir = os.path.dirname(os.path.abspath(snap_path))
        if snap_dir:
            return str(get_daily_report_dir(snap_dir))
    return str(get_daily_report_dir("outputs"))


def _find_existing_report(date_str: str, report_dirs: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    for report_dir in _normalize_report_dirs(report_dirs):
        report_path = os.path.join(report_dir, f"{date_str}.json")
        payload = _safe_read_json(report_path)
        if payload and str(payload.get("date", "")) == date_str:
            return payload, report_path
    return None, None


def _is_existing_report_usable(report: Dict[str, Any]) -> bool:
    if not isinstance(report, dict):
        return False
    trades = report.get("trades", {})
    if not isinstance(trades, dict):
        return False
    if "data_quality" not in trades:
        return False
    meta = trades.get("meta", {})
    if not isinstance(meta, dict):
        return False
    if "excluded_counts" not in meta:
        return False
    try:
        trade_count = int(_as_float(trades.get("trade_count"), 0.0))
    except Exception:
        trade_count = 0
    if trade_count == 0:
        excluded = meta.get("excluded_counts", {})
        if isinstance(excluded, dict):
            if int(_as_float(excluded.get("account_filtered"), 0.0)) > 0:
                return False
            if int(_as_float(excluded.get("session_filtered"), 0.0)) > 0:
                return False
    return True


def _extract_stale_signal(snapshot: Dict[str, Any]) -> Tuple[float, int]:
    if not isinstance(snapshot, dict):
        return 0.0, 0
    stale_ratio = _as_float(snapshot.get("stale_ratio"), -1.0)
    observe_count = int(_as_float(snapshot.get("observe_count"), 0))
    if stale_ratio >= 0:
        return stale_ratio, observe_count

    stale_info = snapshot.get("stale_info", {})
    if isinstance(stale_info, dict):
        stale_ratio = _as_float(stale_info.get("stale_ratio"), -1.0)
        observe_count = int(_as_float(stale_info.get("observe_count"), 0))
        if stale_ratio >= 0:
            return stale_ratio, observe_count

    statuses = snapshot.get("price_statuses", {})
    if isinstance(statuses, dict) and statuses:
        total = 0
        stale = 0
        for status in statuses.values():
            total += 1
            if str(status).upper() == "STALE":
                stale += 1
        if total > 0:
            return float(stale / total), total
    return 0.0, observe_count


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


def _previous_trading_day(day: date_cls) -> date_cls:
    out = day - timedelta(days=1)
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def _next_trading_day(day: date_cls) -> date_cls:
    out = day + timedelta(days=1)
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def get_market_session_state(
    now_dt: Any,
    tz_market: str = MARKET_TZ,
    open_time_et: time = time(9, 30),
    close_time_et: time = time(16, 0),
    open_grace_min: int = 15,
    close_grace_min: int = 10,
) -> Dict[str, Any]:
    """Return market session state in ET and completed-trading-day pointers."""
    market_tz = _coerce_zone(tz_market)
    now_parsed = _parse_datetime(now_dt, tz_market) or datetime.now(market_tz)
    now_et = now_parsed.astimezone(market_tz)
    today = now_et.date()

    open_dt = datetime.combine(today, open_time_et, tzinfo=market_tz)
    close_dt = datetime.combine(today, close_time_et, tzinfo=market_tz)
    open_grace_dt = open_dt + timedelta(minutes=max(0, int(open_grace_min)))
    close_grace_dt = close_dt + timedelta(minutes=max(0, int(close_grace_min)))

    if today.weekday() >= 5:
        state = "WEEKEND"
        trading_date = _next_trading_day(today)
        last_completed = _previous_trading_day(trading_date)
    elif now_et < open_dt:
        state = "PRE_OPEN"
        trading_date = today
        last_completed = _previous_trading_day(today)
    elif now_et < close_dt:
        state = "OPEN"
        trading_date = today
        last_completed = _previous_trading_day(today)
    else:
        state = "POST_CLOSE"
        trading_date = today
        if now_et >= close_grace_dt:
            last_completed = today
        else:
            last_completed = _previous_trading_day(today)

    return {
        "state": state,
        "now_et": now_et.isoformat(),
        "trading_date_et": trading_date.isoformat(),
        "last_completed_trading_date_et": last_completed.isoformat(),
        "open_time_et": open_dt.isoformat(),
        "close_time_et": close_dt.isoformat(),
        "open_grace_min": int(max(0, int(open_grace_min))),
        "close_grace_min": int(max(0, int(close_grace_min))),
        "open_grace_passed": bool(now_et >= open_grace_dt) if state == "OPEN" else False,
        "close_grace_passed": bool(now_et >= close_grace_dt) if state == "POST_CLOSE" else False,
        "is_trading_day": bool(today.weekday() < 5),
    }


def is_market_closed(
    now: Any,
    tz: str,
    snapshot: Dict[str, Any],
    stale_tracker: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Return (closed, reason) by time rule or stale streak rule."""
    local_tz = _coerce_zone(tz)
    now_dt = _parse_datetime(now, tz) or datetime.now(local_tz)
    session = get_market_session_state(now_dt, tz_market=MARKET_TZ)

    tracker = stale_tracker if isinstance(stale_tracker, dict) else {}
    ratio_threshold = float(max(0.0, min(1.0, _as_float(tracker.get("ratio_threshold"), 0.8))))
    streak_threshold = int(max(1, _as_float(tracker.get("threshold"), 3)))
    stale_ratio, observe_count = _extract_stale_signal(snapshot)
    stale_allowed = bool(session.get("state") == "OPEN" and session.get("open_grace_passed"))
    stale_hit = bool(stale_allowed and stale_ratio >= ratio_threshold and observe_count >= 0)
    streak_now = int(_as_float(tracker.get("streak"), 0))
    if stale_hit:
        streak_now += 1
    elif stale_allowed:
        streak_now = 0
    else:
        # PRE_OPEN / POST_CLOSE / WEEKEND should never accumulate stale streak.
        streak_now = 0
    tracker["streak"] = streak_now
    tracker["threshold"] = streak_threshold
    tracker["ratio_threshold"] = ratio_threshold
    tracker["last_ratio"] = stale_ratio
    tracker["last_observe_count"] = observe_count
    tracker["updated_at"] = now_dt.isoformat()
    tracker["session_state"] = session.get("state")
    tracker["stale_allowed"] = stale_allowed

    if bool(session.get("state") == "POST_CLOSE" and session.get("close_grace_passed")):
        return True, {
            "method": "time",
            "details": {
                "session": session,
                "stale_ratio": stale_ratio,
                "streak": streak_now,
                "threshold": streak_threshold,
                "ratio_threshold": ratio_threshold,
                "observe_count": observe_count,
            },
        }

    if stale_hit and streak_now >= streak_threshold:
        return True, {
            "method": "stale_streak",
            "details": {
                "session": session,
                "stale_ratio": stale_ratio,
                "streak": streak_now,
                "threshold": streak_threshold,
                "ratio_threshold": ratio_threshold,
                "observe_count": observe_count,
            },
        }

    return False, {
        "method": "not_closed",
        "details": {
            "session": session,
            "stale_ratio": stale_ratio,
            "streak": streak_now,
            "threshold": streak_threshold,
            "ratio_threshold": ratio_threshold,
            "stale_allowed": stale_allowed,
            "observe_count": observe_count,
        },
    }


def _build_trades_for_date(
    trades_csv_path: str,
    report_date: date_cls,
    tz: str,
    allowed_envs: Optional[List[str]] = None,
    account_id: Optional[str] = None,
    session_id: Optional[str] = None,
    strict_session: bool = True,
    max_cycle_hint: Optional[int] = None,
    cycle_outlier_buffer: int = 5,
    legacy_ticker_blacklist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    buy_notional = 0.0
    sell_notional = 0.0
    trade_count = 0
    by_ticker: Dict[str, Dict[str, Any]] = {}
    raw_rows: List[Dict[str, Any]] = []
    allowed_env_set = {str(x).strip().lower() for x in (allowed_envs or ["live"]) if str(x).strip()}
    wanted_account = _norm_text(account_id)
    wanted_session = _norm_text(session_id)
    excluded_counts: Dict[str, int] = {
        "file_missing": 0,
        "invalid_row": 0,
        "date_mismatch": 0,
        "missing_or_invalid_side": 0,
        "missing_ticker": 0,
        "env_filtered": 0,
        "account_filtered": 0,
        "session_filtered": 0,
        "legacy_missing_account": 0,
        "legacy_missing_session": 0,
        "cycle_outlier": 0,
        "legacy_ticker_blacklist": 0,
    }
    ticker_black_set = {str(x).upper().strip() for x in (legacy_ticker_blacklist or ["AAA", "TEST", "SMOKE"]) if str(x).strip()}
    if not os.path.exists(trades_csv_path):
        excluded_counts["file_missing"] = 1
        return {
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "net_flow": 0.0,
            "trade_count": 0,
            "by_ticker": {},
            "raw": [],
            "meta": {
                "allowed_envs": sorted(list(allowed_env_set)),
                "account_id": wanted_account or None,
                "session_id": wanted_session or None,
                "strict_session": bool(strict_session),
                "max_cycle_hint": max_cycle_hint,
                "cycle_outlier_buffer": int(cycle_outlier_buffer),
                "excluded_counts": excluded_counts,
                "included_count": 0,
            },
            "data_quality": "ok",
            "issues": [],
        }

    with open(trades_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                excluded_counts["invalid_row"] += 1
                continue
            ts_text = row.get("timestamp") or row.get("time") or row.get("datetime")
            ts = _parse_datetime(ts_text, tz)
            if ts is None or ts.date() != report_date:
                excluded_counts["date_mismatch"] += 1
                continue

            ticker = str(row.get("ticker", "")).upper().strip()
            if not ticker:
                excluded_counts["missing_ticker"] += 1
                continue
            side = str(row.get("side", row.get("direction", ""))).upper().strip()
            if side not in ("BUY", "SELL"):
                excluded_counts["missing_or_invalid_side"] += 1
                continue

            row_env = str(row.get("env", "") or "").strip().lower()
            if row_env:
                if allowed_env_set and row_env not in allowed_env_set:
                    excluded_counts["env_filtered"] += 1
                    continue

            row_account = _norm_text(row.get("account_id"))
            row_session = _norm_text(row.get("session_id"))
            legacy_unscoped = False
            if wanted_account:
                if row_account:
                    if row_account != wanted_account:
                        excluded_counts["account_filtered"] += 1
                        continue
                else:
                    excluded_counts["legacy_missing_account"] += 1
                    legacy_unscoped = True

            if wanted_session and strict_session:
                if row_session:
                    if row_session != wanted_session:
                        excluded_counts["session_filtered"] += 1
                        continue
                else:
                    excluded_counts["legacy_missing_session"] += 1
                    legacy_unscoped = True

            if legacy_unscoped:
                cycle_value = row.get("cycle")
                cycle_num = None
                try:
                    cycle_num = int(float(str(cycle_value).strip()))
                except Exception:
                    cycle_num = None
                if (
                    max_cycle_hint is not None
                    and cycle_num is not None
                    and cycle_num > int(max_cycle_hint) + int(max(0, cycle_outlier_buffer))
                ):
                    excluded_counts["cycle_outlier"] += 1
                    continue
                if ticker in ticker_black_set:
                    excluded_counts["legacy_ticker_blacklist"] += 1
                    continue

            qty = int(_as_float(row.get("quantity", row.get("qty")), 0.0))
            price = _as_float(row.get("price"), 0.0)
            notional = _as_float(row.get("notional"), qty * price)
            source = str(row.get("reason", row.get("source", "rebalance")) or "rebalance")

            if side == "BUY":
                buy_notional += notional
            else:
                sell_notional += notional
            trade_count += 1

            info = by_ticker.setdefault(ticker, {"buy": 0.0, "sell": 0.0, "count": 0})
            if side == "BUY":
                info["buy"] = _as_float(info.get("buy"), 0.0) + notional
            else:
                info["sell"] = _as_float(info.get("sell"), 0.0) + notional
            info["count"] = int(_as_float(info.get("count"), 0.0)) + 1

            raw_rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "notional": notional,
                    "source": source,
                    "turnover_scale": _as_float(row.get("turnover_scale"), 0.0),
                    "turnover_limit": _as_float(row.get("turnover_limit"), 0.0),
                    "turnover_notional_pre": _as_float(row.get("turnover_notional_pre"), 0.0),
                    "turnover_notional_post": _as_float(row.get("turnover_notional_post"), 0.0),
                    "priority": row.get("priority"),
                    "force_reason": row.get("force_reason"),
                    "planner_score": _as_float(row.get("planner_score"), 0.0),
                    "account_id": row_account or None,
                    "session_id": row_session or None,
                    "env": row_env or None,
                }
            )

    for ticker in list(by_ticker.keys()):
        by_ticker[ticker]["buy"] = float(by_ticker[ticker]["buy"])
        by_ticker[ticker]["sell"] = float(by_ticker[ticker]["sell"])
        by_ticker[ticker]["count"] = int(by_ticker[ticker]["count"])

    return {
        "buy_notional": float(buy_notional),
        "sell_notional": float(sell_notional),
        "net_flow": float(buy_notional - sell_notional),
        "trade_count": int(trade_count),
        "by_ticker": by_ticker,
        "raw": raw_rows,
        "meta": {
            "allowed_envs": sorted(list(allowed_env_set)),
            "account_id": wanted_account or None,
            "session_id": wanted_session or None,
            "strict_session": bool(strict_session),
            "max_cycle_hint": max_cycle_hint,
            "cycle_outlier_buffer": int(cycle_outlier_buffer),
            "excluded_counts": excluded_counts,
            "included_count": int(trade_count),
        },
        "data_quality": "ok",
        "issues": [],
    }


def _build_positions_end(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    total_equity = _as_float(snapshot.get("total_equity"), 0.0)
    cash = _as_float(snapshot.get("cash"), 0.0)
    positions_value = _as_float(snapshot.get("positions_value"), max(0.0, total_equity - cash))
    details = snapshot.get("positions_detail", {})
    holdings: Dict[str, Dict[str, Any]] = {}
    if isinstance(details, dict):
        for ticker, item in details.items():
            if not isinstance(item, dict):
                continue
            t = str(ticker).upper().strip()
            if not t:
                continue
            qty = int(_as_float(item.get("quantity", item.get("qty")), 0.0))
            px = _as_float(item.get("price"), 0.0)
            val = _as_float(item.get("value"), qty * px)
            w = (val / total_equity) if total_equity > 0 else 0.0
            if val <= 0:
                continue
            holdings[t] = {
                "qty": qty,
                "price": px,
                "value": val,
                "weight": w,
            }
    return {
        "cash": float(cash),
        "positions_value": float(positions_value),
        "holdings": holdings,
    }


def _build_equity_block(
    report_date: date_cls,
    snapshot: Dict[str, Any],
    historical_reports: List[Dict[str, Any]],
    tz: str,
) -> Dict[str, Any]:
    end_equity = _as_float(snapshot.get("total_equity"), None)  # type: ignore[arg-type]
    equity_history = snapshot.get("equity_history", [])
    day_points: List[Tuple[datetime, float]] = []
    if isinstance(equity_history, list):
        for point in equity_history:
            if not isinstance(point, dict):
                continue
            ts = _parse_datetime(point.get("time"), tz)
            eq = _as_float(point.get("equity"), None)  # type: ignore[arg-type]
            if ts is None or eq is None:
                continue
            if ts.date() == report_date:
                day_points.append((ts, float(eq)))
            if end_equity is None:
                end_equity = float(eq)
    day_points.sort(key=lambda x: x[0])
    start_equity = day_points[0][1] if day_points else None

    if start_equity is None:
        prev_day = report_date - timedelta(days=1)
        prev_map = {str(r.get("date")): r for r in historical_reports if isinstance(r, dict)}
        prev_report = prev_map.get(prev_day.isoformat())
        if isinstance(prev_report, dict):
            prev_end = _as_float((prev_report.get("equity") or {}).get("end_equity"), None)  # type: ignore[arg-type]
            if prev_end is not None:
                start_equity = prev_end

    note = None
    pnl = None
    pnl_pct = None
    if start_equity is not None and end_equity is not None and start_equity > 0:
        pnl = float(end_equity - start_equity)
        pnl_pct = float((pnl / start_equity) * 100.0)
    else:
        note = "equity_history不足，且无可用前日end_equity，PnL无法计算"

    return {
        "start_equity": float(start_equity) if start_equity is not None else None,
        "end_equity": float(end_equity) if end_equity is not None else None,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "note": note,
    }


def _apply_trade_data_quality_checks(trades: Dict[str, Any], equity: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(trades or {})
    issues = list(out.get("issues", [])) if isinstance(out.get("issues"), list) else []
    start_equity = _as_float((equity or {}).get("start_equity"), None)  # type: ignore[arg-type]
    sell_notional = _as_float(out.get("sell_notional"), 0.0)
    net_flow = _as_float(out.get("net_flow"), 0.0)

    data_quality = "ok"
    if start_equity is not None and start_equity > 0:
        tolerance = 0.01 * start_equity
        max_reasonable = start_equity + sell_notional + tolerance
        if net_flow > max_reasonable:
            data_quality = "inconsistent"
            issues.append(
                "net_flow exceeds reasonable bound: "
                f"net_flow={net_flow:.2f} > start_equity+sell_notional+tolerance={max_reasonable:.2f}"
            )
    out["data_quality"] = data_quality
    out["issues"] = issues
    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    if start_equity is not None and start_equity > 0:
        meta["consistency_tolerance"] = float(0.01 * start_equity)
    out["meta"] = meta
    return out


def _normalize_reason_counts(raw: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for reason, count in raw.items():
        reason_key = str(reason or "").strip().upper()
        if not reason_key:
            continue
        out[reason_key] = out.get(reason_key, 0) + int(_as_float(count, 0.0))
    return out


def _extract_gate_reason(snapshot: Dict[str, Any], gate_decision: Dict[str, Any]) -> str:
    if isinstance(gate_decision, dict):
        gd_reason = str(gate_decision.get("reason", "")).strip()
        if gd_reason:
            return gd_reason
    skip_reason = str((snapshot or {}).get("rebalance_skipped_reason", "")).strip()
    if skip_reason.startswith("risk_gate:"):
        return skip_reason.split(":", 1)[1]
    if skip_reason == "risk_gate_stub":
        return skip_reason
    rg_reason = str((snapshot or {}).get("risk_gate_reason", "")).strip()
    if rg_reason:
        return rg_reason
    return ""


def _infer_proxy_scope(asset_policy_mode: str, execution_proxy_used: bool, snapshot: Dict[str, Any], proxy_scope: Optional[str] = None) -> str:
    scope_raw = _norm_text(proxy_scope or snapshot.get("proxy_scope") or snapshot.get("ticker_proxy_scope")).lower()
    if scope_raw in {"risk_only", "risk_and_execution", "off"}:
        return scope_raw
    mode = str(asset_policy_mode or "").strip().upper()
    if mode == "FORCE_PROXY":
        return "risk_and_execution" if bool(execution_proxy_used) else "risk_only"
    return "off"


def _build_cost_summary(snapshot: Dict[str, Any], trades: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_obj = snapshot if isinstance(snapshot, dict) else {}
    trades_obj = trades if isinstance(trades, dict) else {}
    raw_cost = snapshot_obj.get("cost_summary", {}) if isinstance(snapshot_obj.get("cost_summary"), dict) else {}
    cost_model = snapshot_obj.get("cost_model", {}) if isinstance(snapshot_obj.get("cost_model"), dict) else {}
    totals_raw = raw_cost.get("totals", {}) if isinstance(raw_cost.get("totals"), dict) else {}

    total = _as_float(totals_raw.get("total"), _as_float(raw_cost.get("total"), 0.0))
    fee = _as_float(totals_raw.get("fee"), _as_float(raw_cost.get("fee"), 0.0))
    slippage = _as_float(totals_raw.get("slippage"), _as_float(raw_cost.get("slippage"), 0.0))
    impact = _as_float(totals_raw.get("impact"), _as_float(raw_cost.get("impact"), 0.0))

    traded_notional = _as_float(
        raw_cost.get("traded_notional"),
        abs(_as_float(trades_obj.get("buy_notional"), 0.0)) + abs(_as_float(trades_obj.get("sell_notional"), 0.0)),
    )
    trades_count = int(
        _as_float(
            raw_cost.get("trades_count"),
            _as_float(raw_cost.get("num_trades"), _as_float(trades_obj.get("trade_count"), 0.0)),
        )
    )
    cost_bps = _as_float(raw_cost.get("cost_bps"), None)  # type: ignore[arg-type]
    if cost_bps is None:
        cost_bps = float(total / traded_notional * 10000.0) if traded_notional > 1e-12 else 0.0

    apply_to = raw_cost.get("apply_to", ["BUY", "SELL"])
    if not isinstance(apply_to, list):
        apply_to = ["BUY", "SELL"]

    return {
        "schema_version": 1,
        "enabled": bool(raw_cost.get("enabled", cost_model.get("enabled", False))),
        "currency": str(raw_cost.get("currency", "USD") or "USD"),
        "slippage_bps": _as_float(raw_cost.get("slippage_bps"), 0.0),
        "fee_per_trade": _as_float(raw_cost.get("fee_per_trade"), 0.0),
        "fee_bps": _as_float(raw_cost.get("fee_bps"), 0.0),
        "min_fee": _as_float(raw_cost.get("min_fee"), 0.0),
        "apply_to": [str(x).upper().strip() for x in apply_to if str(x).strip()],
        "note": str(raw_cost.get("note", "simple bps slippage + fee") or "simple bps slippage + fee"),
        "totals": {
            "total": float(total),
            "fee": float(fee),
            "slippage": float(slippage),
            "impact": float(impact),
        },
        "trades_count": int(max(0, trades_count)),
        "traded_notional": float(max(0.0, traded_notional)),
        "cost_bps": float(cost_bps),
    }


def _build_performance_summary(equity: Dict[str, Any], cost_summary: Dict[str, Any], trades: Dict[str, Any]) -> Dict[str, Any]:
    equity_obj = equity if isinstance(equity, dict) else {}
    cost_obj = cost_summary if isinstance(cost_summary, dict) else {}
    trades_obj = trades if isinstance(trades, dict) else {}
    totals = cost_obj.get("totals", {}) if isinstance(cost_obj.get("totals"), dict) else {}

    total_cost = _as_float(totals.get("total"), _as_float(cost_obj.get("total"), 0.0))
    traded_notional = _as_float(
        cost_obj.get("traded_notional"),
        abs(_as_float(trades_obj.get("buy_notional"), 0.0)) + abs(_as_float(trades_obj.get("sell_notional"), 0.0)),
    )
    pnl = equity_obj.get("pnl")
    pnl_val = _as_float(pnl, None)  # type: ignore[arg-type]
    if pnl_val is None:
        gross_pnl = None
        net_pnl = None
    else:
        net_pnl = float(pnl_val)
        gross_pnl = float(net_pnl + total_cost)
    end_equity = _as_float(equity_obj.get("end_equity"), 0.0)
    cost_to_equity_pct = float(total_cost / end_equity * 100.0) if end_equity > 1e-12 else None

    return {
        "schema_version": 1,
        "gross_pnl_estimate": gross_pnl,
        "net_pnl_estimate": net_pnl,
        "traded_notional": float(max(0.0, traded_notional)),
        "cost_total": float(total_cost),
        "cost_bps": float(cost_obj.get("cost_bps", float(total_cost / traded_notional * 10000.0) if traded_notional > 1e-12 else 0.0)),
        "cost_to_equity_pct": cost_to_equity_pct,
    }


def _extract_price_health_counts(snapshot_obj: Dict[str, Any], returns_missing_count: Optional[int] = None) -> Dict[str, Any]:
    snap = snapshot_obj if isinstance(snapshot_obj, dict) else {}
    missing_count = None
    stale_count = None
    live_count = None

    price_debug = snap.get("price_debug", {}) if isinstance(snap.get("price_debug"), dict) else {}
    if isinstance(price_debug, dict) and price_debug:
        m = 0
        s = 0
        l = 0
        for _, row in price_debug.items():
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "")).upper().strip()
            if status == "MISSING":
                m += 1
            elif status == "STALE":
                s += 1
            elif status in ("LIVE", "RECENT"):
                l += 1
        missing_count = int(m)
        stale_count = int(s)
        live_count = int(l)
    else:
        execution_summary = snap.get("execution_summary", {}) if isinstance(snap.get("execution_summary"), dict) else {}
        skip_reasons = execution_summary.get("skip_reasons", {}) if isinstance(execution_summary.get("skip_reasons"), dict) else {}
        if isinstance(skip_reasons, dict) and skip_reasons:
            missing_count = int(_as_float(skip_reasons.get("PRICE_MISSING"), 0.0))
            stale_count = int(_as_float(skip_reasons.get("PRICE_STALE"), 0.0))

    if missing_count is None and returns_missing_count is not None:
        missing_count = int(max(0, int(returns_missing_count)))

    return {
        "missing_count": missing_count,
        "stale_count": stale_count,
        "live_count": live_count,
    }


def build_risk_model_health(
    report: Optional[Dict[str, Any]],
    snapshot: Optional[Dict[str, Any]] = None,
    daily_fields: Optional[Dict[str, Any]] = None,
    telemetry_optional: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_obj = report if isinstance(report, dict) else {}
    snapshot_obj = snapshot if isinstance(snapshot, dict) else {}
    fields = daily_fields if isinstance(daily_fields, dict) else {}
    _ = telemetry_optional  # reserved for future expansion

    date_str = str(report_obj.get("date", "")).strip()
    no_trade = fields.get("no_trade_summary")
    if not isinstance(no_trade, dict):
        no_trade = report_obj.get("no_trade_summary", {}) if isinstance(report_obj.get("no_trade_summary"), dict) else {}
    risk_gate_decision = fields.get("risk_gate_decision")
    if not isinstance(risk_gate_decision, dict):
        risk_gate_decision = report_obj.get("risk_gate_decision", {}) if isinstance(report_obj.get("risk_gate_decision"), dict) else {}
    cov_coverage = fields.get("cov_coverage")
    if not isinstance(cov_coverage, dict):
        cov_coverage = report_obj.get("cov_coverage", {}) if isinstance(report_obj.get("cov_coverage"), dict) else {}
    returns_diag = fields.get("returns_coverage_diag")
    if not isinstance(returns_diag, dict):
        returns_diag = report_obj.get("returns_coverage_diag", {}) if isinstance(report_obj.get("returns_coverage_diag"), dict) else {}
    execution_summary = fields.get("execution_summary")
    if not isinstance(execution_summary, dict):
        execution_summary = (
            report_obj.get("execution_summary", {})
            if isinstance(report_obj.get("execution_summary"), dict)
            else snapshot_obj.get("execution_summary", {})
            if isinstance(snapshot_obj.get("execution_summary"), dict)
            else {}
        )
    cost_summary = fields.get("cost_summary")
    if not isinstance(cost_summary, dict):
        cost_summary = report_obj.get("cost_summary", {}) if isinstance(report_obj.get("cost_summary"), dict) else {}

    gate_reason = str(risk_gate_decision.get("reason", "")).strip()
    if not gate_reason:
        gate_reason = str(no_trade.get("gate_reason", "")).strip()
    if not gate_reason:
        skip_reason = str(snapshot_obj.get("rebalance_skipped_reason", report_obj.get("rebalance_skipped_reason", ""))).strip()
        if skip_reason.startswith("risk_gate:"):
            gate_reason = skip_reason.split(":", 1)[1]
        elif skip_reason == "risk_gate_stub":
            gate_reason = skip_reason

    blockers = no_trade.get("top_blockers", []) if isinstance(no_trade.get("top_blockers"), list) else []
    blocker_reasons = {str(x.get("reason", "")).upper().strip() for x in blockers if isinstance(x, dict)}
    gate_triggered = bool(gate_reason) or ("RISK_GATE" in blocker_reasons)
    if not gate_reason and gate_triggered:
        gate_reason = "RISK_GATE"

    returns_items = returns_diag.get("items", []) if isinstance(returns_diag.get("items"), list) else []
    returns_missing_top: List[Dict[str, Any]] = []
    for row in returns_items[:10]:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        reason = str(row.get("reason_code", row.get("reason", "UNKNOWN"))).upper().strip() or "UNKNOWN"
        returns_missing_top.append({"ticker": ticker, "reason": reason})
    returns_missing_count = int(len(returns_items))

    cov_missing_count = cov_coverage.get("missing_count", None) if isinstance(cov_coverage, dict) else None
    if cov_missing_count is None and isinstance(cov_coverage.get("missing_tickers"), list):
        cov_missing_count = int(len(cov_coverage.get("missing_tickers", [])))

    price_counts = _extract_price_health_counts(snapshot_obj if snapshot_obj else report_obj, returns_missing_count=returns_missing_count)

    skip_reasons = execution_summary.get("skip_reasons", {}) if isinstance(execution_summary.get("skip_reasons"), dict) else {}
    skip_rows = []
    for reason, count in sorted(skip_reasons.items(), key=lambda kv: (-int(_as_float(kv[1], 0.0)), str(kv[0]))):
        skip_rows.append({"reason": str(reason), "count": int(_as_float(count, 0.0))})
        if len(skip_rows) >= 8:
            break

    mode = str(
        fields.get("asset_policy_mode")
        or report_obj.get("asset_policy_mode")
        or snapshot_obj.get("asset_policy_mode")
        or "FORCE_PROXY"
    ).strip().upper() or "FORCE_PROXY"
    execution_proxy_used = bool(
        fields.get("execution_proxy_used")
        if fields.get("execution_proxy_used") is not None
        else report_obj.get("execution_proxy_used", report_obj.get("ticker_proxy_used", snapshot_obj.get("execution_proxy_used", snapshot_obj.get("ticker_proxy_used", False))))
    )

    totals = cost_summary.get("totals", {}) if isinstance(cost_summary.get("totals"), dict) else {}
    cost_total = _as_float(totals.get("total"), _as_float(cost_summary.get("cost_total"), _as_float(cost_summary.get("total"), 0.0)))
    cost_bps = _as_float(cost_summary.get("cost_bps"), None)  # type: ignore[arg-type]
    if cost_bps is None:
        traded_notional = _as_float(cost_summary.get("traded_notional"), 0.0)
        cost_bps = float(cost_total / traded_notional * 10000.0) if traded_notional > 1e-12 else 0.0
    trades_count = int(
        _as_float(
            cost_summary.get("trades_count"),
            _as_float(cost_summary.get("num_trades"), _as_float(execution_summary.get("orders_place"), 0.0)),
        )
    )

    return {
        "schema_version": 1,
        "date": date_str,
        "risk_gate": {
            "triggered": bool(gate_triggered),
            "reason": gate_reason,
            "metric_name": str(risk_gate_decision.get("metric_name", "")).strip(),
            "metric_value": risk_gate_decision.get("metric_value"),
            "threshold": risk_gate_decision.get("threshold"),
            "stage": str(risk_gate_decision.get("stage", "unknown") or "unknown"),
        },
        "coverage": {
            "cov_known_weight": _as_float(cov_coverage.get("known_weight"), None),  # type: ignore[arg-type]
            "cov_missing_weight_total": _as_float(cov_coverage.get("missing_weight_total"), None),  # type: ignore[arg-type]
            "cov_missing_count": int(_as_float(cov_missing_count, 0.0)) if cov_missing_count is not None else None,
            "returns_missing_count": int(returns_missing_count),
            "returns_missing_top": returns_missing_top,
        },
        "prices": price_counts,
        "execution": {
            "orders_place": int(_as_float(execution_summary.get("orders_place"), 0.0)),
            "orders_skip": int(_as_float(execution_summary.get("orders_skip"), 0.0)),
            "top_skip_reasons": skip_rows,
        },
        "policy": {
            "asset_policy_mode": mode,
            "execution_proxy_used": bool(execution_proxy_used),
        },
        "cost": {
            "enabled": bool(cost_summary.get("enabled", False)),
            "cost_total": float(cost_total),
            "cost_bps": float(cost_bps) if cost_bps is not None else None,
            "trades_count": int(max(0, trades_count)),
        },
    }


def build_no_trade_summary(
    trades: Optional[Dict[str, Any]],
    snapshot: Optional[Dict[str, Any]],
    risk_gate_decision: Optional[Dict[str, Any]] = None,
    cov_coverage: Optional[Dict[str, Any]] = None,
    returns_coverage_diag: Optional[Dict[str, Any]] = None,
    asset_policy_mode: Optional[str] = None,
    execution_proxy_used: Optional[bool] = None,
    proxy_scope: Optional[str] = None,
) -> Dict[str, Any]:
    trades_obj = trades if isinstance(trades, dict) else {}
    snapshot_obj = snapshot if isinstance(snapshot, dict) else {}
    gate_obj = risk_gate_decision if isinstance(risk_gate_decision, dict) else {}
    cov_obj = cov_coverage if isinstance(cov_coverage, dict) else {}
    returns_obj = returns_coverage_diag if isinstance(returns_coverage_diag, dict) else {}
    execution_summary = (
        dict(snapshot_obj.get("execution_summary"))
        if isinstance(snapshot_obj.get("execution_summary"), dict)
        else {}
    )
    reason_counts = _normalize_reason_counts(execution_summary.get("skip_reasons", {}))
    orders_place = int(_as_float(execution_summary.get("orders_place"), 0.0))
    orders_skip = int(_as_float(execution_summary.get("orders_skip"), 0.0))
    trade_count = int(_as_float(trades_obj.get("trade_count"), 0.0))
    if orders_place <= 0 and trade_count > 0:
        orders_place = trade_count
    has_trade = bool(orders_place > 0 or trade_count > 0)

    blockers = []
    for reason_key, count in sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        blockers.append({"reason": str(reason_key), "count": int(count)})
    blockers = blockers[:8]

    gate_reason = _extract_gate_reason(snapshot_obj, gate_obj)
    if not blockers and gate_reason and not has_trade:
        blockers = [{"reason": "RISK_GATE", "count": 1}]

    metric_name = str(gate_obj.get("metric_name", "")).strip()
    metric_value = gate_obj.get("metric_value", None)
    threshold_value = gate_obj.get("threshold", None)
    metric_stage = str(gate_obj.get("stage", "")).strip()

    returns_missing_top: List[Dict[str, Any]] = []
    items = returns_obj.get("items", []) if isinstance(returns_obj.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        reason_code = str(item.get("reason_code", "UNKNOWN")).upper().strip() or "UNKNOWN"
        returns_missing_top.append({"ticker": ticker, "reason": reason_code})
        if len(returns_missing_top) >= 5:
            break

    mode = str(asset_policy_mode or snapshot_obj.get("asset_policy_mode") or "FORCE_PROXY").strip().upper() or "FORCE_PROXY"
    proxy_used = bool(
        execution_proxy_used
        if execution_proxy_used is not None
        else snapshot_obj.get("execution_proxy_used", snapshot_obj.get("ticker_proxy_used", False))
    )
    inferred_scope = _infer_proxy_scope(mode, proxy_used, snapshot_obj, proxy_scope=proxy_scope)

    cov_known_weight = _as_float(cov_obj.get("known_weight"), None)  # type: ignore[arg-type]
    cov_missing_weight_total = _as_float(cov_obj.get("missing_weight_total"), None)  # type: ignore[arg-type]

    return {
        "schema_version": 1,
        "has_trade": bool(has_trade),
        "orders_place": int(max(0, orders_place)),
        "orders_skip": int(max(0, orders_skip)),
        "top_blockers": blockers,
        "gate_reason": gate_reason or "",
        "gate_metric": {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold": threshold_value,
            "stage": metric_stage,
        },
        "data_issues": {
            "returns_missing_top": returns_missing_top,
            "cov_known_weight": cov_known_weight,
            "cov_missing_weight_total": cov_missing_weight_total,
        },
        "policy": {
            "asset_policy_mode": mode,
            "execution_proxy_used": bool(proxy_used),
            "proxy_scope": inferred_scope,
        },
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def risk_score(
    ticker: str,
    holding: Dict[str, Any],
    snapshot: Dict[str, Any],
    report_context: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Deterministic risk score (0-10) with explainable evidence."""
    weight = _as_float(holding.get("weight"), 0.0)
    value = _as_float(holding.get("value"), 0.0)

    conc_score = _clamp01((weight - 0.10) / 0.15) * 10.0
    size_signal = value * max(weight, 1e-9)
    max_size_signal = max(1e-9, _as_float(report_context.get("max_size_signal"), 0.0))
    size_score = _clamp01(size_signal / max_size_signal) * 10.0

    vol = _as_float(snapshot.get("portfolio_vol_cov_annualized"), 0.0)
    herf = _as_float(snapshot.get("herfindahl_index"), 0.0)
    drawdown = _as_float(snapshot.get("drawdown"), 0.0)
    rc_frac = _as_float(snapshot.get("max_rc_fraction_cov"), 0.0)

    vol_norm = _clamp01((vol - 0.18) / 0.20)
    herf_norm = _clamp01((herf - 0.22) / 0.18)
    dd_norm = _clamp01(drawdown / 0.08)
    rc_norm = _clamp01((rc_frac - 0.30) / 0.25)
    portfolio_risk_level = max(vol_norm, herf_norm, dd_norm, rc_norm)
    proxy_score = portfolio_risk_level * 10.0

    score = 0.50 * conc_score + 0.30 * size_score + 0.20 * proxy_score
    evidence = {
        "weight": weight,
        "value": value,
        "size_signal_value_x_weight": size_signal,
        "factor_concentration_0_10": conc_score,
        "factor_size_0_10": size_score,
        "factor_portfolio_proxy_0_10": proxy_score,
        "portfolio_proxy": {
            "portfolio_vol_cov_annualized": vol,
            "herfindahl_index": herf,
            "drawdown": drawdown,
            "max_rc_fraction_cov": rc_frac,
            "level_0_1": portfolio_risk_level,
        },
    }
    return float(round(score, 4)), evidence


def _build_risk_section(positions_end: Dict[str, Any], snapshot: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    holdings = positions_end.get("holdings", {})
    if not isinstance(holdings, dict):
        holdings = {}

    max_value = 0.0
    max_size_signal = 0.0
    for v in holdings.values():
        if isinstance(v, dict):
            value = _as_float(v.get("value"), 0.0)
            weight = _as_float(v.get("weight"), 0.0)
            max_value = max(max_value, value)
            max_size_signal = max(max_size_signal, value * max(weight, 1e-9))
    context = {"max_value": max_value, "max_size_signal": max_size_signal}

    scored: List[Dict[str, Any]] = []
    for ticker, h in holdings.items():
        if not isinstance(h, dict):
            continue
        score, evidence = risk_score(str(ticker), h, snapshot, context)
        scored.append({"ticker": str(ticker).upper(), "score": score, "evidence": evidence})
    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return {
        "rules": [
            "High concentration: weight >= 10% starts penalty, >25% near max score",
            "High volatility proxy: value * weight ranked within portfolio",
            "Portfolio risk proxy: higher vol/herfindahl/drawdown/max_rc raises top holdings risk",
        ],
        "risky_tickers": scored[: max(1, int(top_n))],
    }


def _collect_holding_series(history_reports: List[Dict[str, Any]], ticker: str) -> List[Tuple[str, float]]:
    series: List[Tuple[str, float]] = []
    for report in history_reports:
        if not isinstance(report, dict):
            continue
        date_str = str(report.get("date", ""))
        holdings = (((report.get("positions_end") or {}).get("holdings")) or {})
        if not isinstance(holdings, dict):
            continue
        item = holdings.get(ticker)
        if isinstance(item, dict):
            weight = _as_float(item.get("weight"), 0.0)
            if weight > 0:
                series.append((date_str, float(weight)))
    return series


def _build_conviction_section(
    report_date: date_cls,
    positions_end: Dict[str, Any],
    trades: Dict[str, Any],
    history_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    holdings = positions_end.get("holdings", {})
    if not isinstance(holdings, dict):
        holdings = {}
    trade_raw = trades.get("raw", [])
    if not isinstance(trade_raw, list):
        trade_raw = []

    history_before_today = []
    for r in history_reports:
        rd = _parse_date(r.get("date"))
        if isinstance(rd, date_cls) and rd < report_date:
            history_before_today.append(r)

    long_term: List[Dict[str, Any]] = []
    for ticker, h in sorted(holdings.items()):
        if not isinstance(h, dict):
            continue
        weights = [w for _, w in _collect_holding_series(history_before_today, ticker)]
        if len(weights) < 3:
            continue
        avg_w = sum(weights) / len(weights)
        variance = sum((x - avg_w) ** 2 for x in weights) / max(1, len(weights))
        std_w = math.sqrt(max(0.0, variance))
        today_w = _as_float(h.get("weight"), 0.0)
        if avg_w >= 0.04 and std_w <= 0.03 and today_w >= 0.03:
            long_term.append(
                {
                    "ticker": ticker,
                    "why": (
                        f"近{len(weights)}个报告日平均持仓权重{avg_w:.2%}，波动{std_w:.2%}，"
                        f"当前权重{today_w:.2%}，满足长期核心仓规则"
                    ),
                }
            )

    last_report = None
    if history_before_today:
        history_before_today_sorted = sorted(
            history_before_today,
            key=lambda x: str(x.get("date", "")),
        )
        last_report = history_before_today_sorted[-1]
    prev_holdings = (((last_report or {}).get("positions_end") or {}).get("holdings")) or {}
    if not isinstance(prev_holdings, dict):
        prev_holdings = {}

    buy_stats: Dict[str, Dict[str, float]] = {}
    for row in trade_raw:
        if not isinstance(row, dict):
            continue
        if str(row.get("side", "")).upper() != "BUY":
            continue
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        b = buy_stats.setdefault(ticker, {"notional": 0.0, "count": 0.0})
        b["notional"] += _as_float(row.get("notional"), 0.0)
        b["count"] += 1.0

    long_term_tickers = {str(x.get("ticker", "")).upper() for x in long_term}
    short_term: List[Dict[str, Any]] = []
    for ticker, stat in sorted(buy_stats.items(), key=lambda x: x[1]["notional"], reverse=True):
        if ticker in long_term_tickers:
            continue
        today_item = holdings.get(ticker, {})
        prev_item = prev_holdings.get(ticker, {})
        today_qty = int(_as_float((today_item or {}).get("qty"), 0.0))
        prev_qty = int(_as_float((prev_item or {}).get("qty"), 0.0))
        today_w = _as_float((today_item or {}).get("weight"), 0.0)
        prev_w = _as_float((prev_item or {}).get("weight"), 0.0)
        delta_w = today_w - prev_w

        reason = None
        if prev_qty <= 0 and today_qty > 0:
            reason = f"今日新开仓，买入金额${stat['notional']:.2f}，当前权重{today_w:.2%}"
        elif today_qty > prev_qty and (stat["notional"] >= 500.0 or (prev_qty > 0 and today_qty >= int(prev_qty * 1.3))):
            reason = (
                f"今日显著加仓，买入金额${stat['notional']:.2f}，仓位{prev_qty} -> {today_qty}，"
                f"权重变化{delta_w:+.2%}"
            )
        elif abs(delta_w) >= 0.03:
            reason = f"持仓权重单日变化{delta_w:+.2%}，短期仓位调整特征明显"

        if reason:
            short_term.append({"ticker": ticker, "why": reason})

    if len(short_term) > 5:
        short_term = short_term[:5]
    if len(long_term) > 5:
        long_term = long_term[:5]

    return {
        "long_term": long_term,
        "short_term": short_term,
        "notes": "基于持仓稳定性、连续出现次数、当日新开仓/加仓和权重跳变的确定性规则（当前以持仓权重替代target_weight）",
    }


def _dedupe_reports(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[str, Dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        date_str = str(report.get("date", "")).strip()
        if not date_str:
            continue
        dedup[date_str] = report
    return [dedup[k] for k in sorted(dedup.keys())]


def _build_index_entry(report: Dict[str, Any], report_path: str) -> Dict[str, Any]:
    equity = report.get("equity", {}) if isinstance(report.get("equity"), dict) else {}
    trades = report.get("trades", {}) if isinstance(report.get("trades"), dict) else {}
    risky = report.get("risk", {}) if isinstance(report.get("risk"), dict) else {}
    conviction = report.get("conviction", {}) if isinstance(report.get("conviction"), dict) else {}
    no_trade = report.get("no_trade_summary", {}) if isinstance(report.get("no_trade_summary"), dict) else {}
    risky_list = risky.get("risky_tickers", []) if isinstance(risky.get("risky_tickers"), list) else []

    return {
        "date": str(report.get("date", "")),
        "path": report_path,
        "generated_at_local": report.get("generated_at_local"),
        "summary": {
            "pnl": equity.get("pnl"),
            "pnl_pct": equity.get("pnl_pct"),
            "trade_count": trades.get("trade_count"),
            "buy_notional": trades.get("buy_notional"),
            "sell_notional": trades.get("sell_notional"),
            "net_flow": trades.get("net_flow"),
            "data_quality": trades.get("data_quality", "ok"),
        },
        "risk_top": [
            {"ticker": x.get("ticker"), "score": x.get("score")}
            for x in risky_list[:3]
            if isinstance(x, dict)
        ],
        "conviction_counts": {
            "long_term": len(conviction.get("long_term", [])) if isinstance(conviction.get("long_term"), list) else 0,
            "short_term": len(conviction.get("short_term", [])) if isinstance(conviction.get("short_term"), list) else 0,
        },
        "no_trade_summary": {
            "has_trade": bool(no_trade.get("has_trade", False)),
            "orders_place": int(_as_float(no_trade.get("orders_place"), 0.0)),
            "orders_skip": int(_as_float(no_trade.get("orders_skip"), 0.0)),
            "top_blockers": [
                {"reason": str(x.get("reason", "")), "count": int(_as_float(x.get("count"), 0.0))}
                for x in (no_trade.get("top_blockers", []) if isinstance(no_trade.get("top_blockers"), list) else [])[:3]
                if isinstance(x, dict)
            ],
        },
    }


def _write_index(report_dir: str) -> None:
    reports = load_reports(report_dir)
    entries = []
    for report in reports:
        date_str = str(report.get("date", "")).strip()
        if not date_str:
            continue
        report_path = os.path.join(os.path.abspath(report_dir), f"{date_str}.json")
        entries.append(_build_index_entry(report, report_path))
    entries.sort(key=lambda x: str(x.get("date", "")), reverse=True)
    payload = {
        "updated_at": datetime.now(_coerce_zone(DEFAULT_TZ)).isoformat(),
        "report_dir": os.path.abspath(report_dir),
        "reports": entries,
    }
    _atomic_write_json(os.path.join(report_dir, INDEX_FILENAME), payload)


def generate_daily_report(
    date: Any,
    snapshot_path: str,
    trades_csv_path: str,
    report_dirs: List[str],
    tz: str,
) -> Dict[str, Any]:
    """Generate a single-day report dict (idempotent by existing file)."""
    report_date = _parse_date(date, tz) or datetime.now(_coerce_zone(tz)).date()
    date_str = report_date.isoformat()
    snapshot = _safe_read_json(snapshot_path) or {}
    run_reports_dir = _resolve_run_reports_dir(snapshot_path, snapshot)
    normalized_dirs = _normalize_report_dirs([run_reports_dir] + list(report_dirs or []))

    existing, existing_path = _find_existing_report(date_str, normalized_dirs)
    if existing is not None:
        if _is_existing_report_usable(existing):
            existing_patched, existing_changed = _ensure_report_meta_fields(existing, snapshot=snapshot)
            if existing_changed and existing_path:
                try:
                    _atomic_write_json(existing_path, existing_patched)
                except Exception:
                    pass
            existing_patched["_already_exists"] = True
            existing_patched["_existing_path"] = existing_path
            return existing_patched

    snapshot_account_id = _norm_text(snapshot.get("account_id")) or "paper_main"
    snapshot_session_id = _norm_text(snapshot.get("session_id"))
    snapshot_run_id = _norm_text(snapshot.get("run_id")) or snapshot_session_id
    snapshot_schema_version = snapshot.get("schema_version")
    snapshot_cycle_id = snapshot.get("cycle_id")
    if snapshot_cycle_id in (None, ""):
        snapshot_cycle_id = snapshot.get("cycle")
    snapshot_env = _norm_text(snapshot.get("env")).lower()
    snapshot_cycle = int(_as_float(snapshot.get("cycle"), 0.0))
    allowed_envs = ["live"]
    if snapshot_env and snapshot_env in ("live",):
        allowed_envs = [snapshot_env]

    history: List[Dict[str, Any]] = []
    for report_dir in normalized_dirs:
        history.extend(load_reports(report_dir, lookback_days=400))
    history = _dedupe_reports(history)

    trades = _build_trades_for_date(
        trades_csv_path=trades_csv_path,
        report_date=report_date,
        tz=tz,
        allowed_envs=allowed_envs,
        account_id=snapshot_account_id or None,
        session_id=snapshot_session_id or None,
        strict_session=True,
        max_cycle_hint=snapshot_cycle if snapshot_cycle > 0 else None,
        cycle_outlier_buffer=5,
        legacy_ticker_blacklist=["AAA", "TEST", "SMOKE"],
    )
    positions_end = _build_positions_end(snapshot)
    equity = _build_equity_block(report_date, snapshot, history, tz)
    trades = _apply_trade_data_quality_checks(trades, equity)
    risk = _build_risk_section(positions_end, snapshot, top_n=5)
    conviction = _build_conviction_section(report_date, positions_end, trades, history)
    risk_gate_decision = (
        dict(snapshot.get("risk_gate_decision"))
        if isinstance(snapshot.get("risk_gate_decision"), dict)
        else {}
    )
    execution_summary = (
        dict(snapshot.get("execution_summary"))
        if isinstance(snapshot.get("execution_summary"), dict)
        else {}
    )
    no_trade_summary = build_no_trade_summary(
        trades=trades,
        snapshot=snapshot,
        risk_gate_decision=risk_gate_decision,
        cov_coverage=snapshot.get("cov_coverage") if isinstance(snapshot.get("cov_coverage"), dict) else {},
        returns_coverage_diag=snapshot.get("returns_coverage_diag") if isinstance(snapshot.get("returns_coverage_diag"), dict) else {},
        asset_policy_mode=str(snapshot.get("asset_policy_mode") or "FORCE_PROXY"),
        execution_proxy_used=bool(snapshot.get("execution_proxy_used", snapshot.get("ticker_proxy_used", False))),
        proxy_scope=_norm_text(snapshot.get("proxy_scope") or snapshot.get("ticker_proxy_scope")),
    )
    cost_summary = _build_cost_summary(snapshot, trades)
    performance_summary = _build_performance_summary(equity, cost_summary, trades)
    risk_model_health = build_risk_model_health(
        report={
            "date": date_str,
            "execution_summary": execution_summary,
            "risk_gate_decision": risk_gate_decision,
            "cov_coverage": snapshot.get("cov_coverage") if isinstance(snapshot.get("cov_coverage"), dict) else {},
            "returns_coverage_diag": snapshot.get("returns_coverage_diag") if isinstance(snapshot.get("returns_coverage_diag"), dict) else {"schema_version": 1, "items": []},
            "asset_policy_mode": str(snapshot.get("asset_policy_mode") or "FORCE_PROXY"),
            "execution_proxy_used": bool(snapshot.get("execution_proxy_used", snapshot.get("ticker_proxy_used", False))),
            "cost_summary": cost_summary,
            "no_trade_summary": no_trade_summary,
        },
        snapshot=snapshot,
        daily_fields={
            "no_trade_summary": no_trade_summary,
            "cost_summary": cost_summary,
            "execution_summary": execution_summary,
            "risk_gate_decision": risk_gate_decision,
            "cov_coverage": snapshot.get("cov_coverage") if isinstance(snapshot.get("cov_coverage"), dict) else {},
            "returns_coverage_diag": snapshot.get("returns_coverage_diag") if isinstance(snapshot.get("returns_coverage_diag"), dict) else {"schema_version": 1, "items": []},
            "asset_policy_mode": str(snapshot.get("asset_policy_mode") or "FORCE_PROXY"),
            "execution_proxy_used": bool(snapshot.get("execution_proxy_used", snapshot.get("ticker_proxy_used", False))),
        },
    )

    generated_at = datetime.now(_coerce_zone(tz)).isoformat()
    price_fetch_stats = snapshot.get("price_fetch_stats", {})
    if not isinstance(price_fetch_stats, dict):
        price_fetch_stats = {}
    pf_batch_calls = int(_as_float(price_fetch_stats.get("batch_calls"), 0.0))
    pf_hit = int(_as_float(price_fetch_stats.get("cache_hits"), 0.0))
    pf_miss = int(_as_float(price_fetch_stats.get("cache_misses"), 0.0))
    pf_ms = int(_as_float(price_fetch_stats.get("elapsed_ms"), 0.0))
    price_fetch_summary = (
        f"PRICE_FETCH: batch_calls={pf_batch_calls}, hit={pf_hit}, miss={pf_miss}, ms={pf_ms}"
        if price_fetch_stats
        else None
    )
    report = {
        "date": date_str,
        "generated_at_local": generated_at,
        "run_id": snapshot_run_id or None,
        "active_risk_profile": str(snapshot.get("active_risk_profile") or snapshot.get("requested_risk_profile") or "mid").strip().lower() or "mid",
        "risk_profile_source": str(snapshot.get("risk_profile_source") or "unknown"),
        "last_risk_profile_change_ts": str(snapshot.get("last_risk_profile_change_ts") or ""),
        "last_risk_profile_change_old": str(snapshot.get("last_risk_profile_change_old") or ""),
        "last_risk_profile_change_new": str(snapshot.get("last_risk_profile_change_new") or ""),
        "last_risk_profile_change_source": str(snapshot.get("last_risk_profile_change_source") or ""),
        "schema_version": snapshot_schema_version,
        "cycle_id": snapshot_cycle_id,
        "risk_profile": str(snapshot.get("active_risk_profile") or snapshot.get("requested_risk_profile") or "mid").strip().lower() or "mid",
        "risk_profile_template_version": snapshot.get("risk_profile_template_version"),
        "risk_profile_overrides_hash": str(snapshot.get("risk_profile_overrides_hash") or ""),
        "market_close": {
            "closed": True,
            "reason": {
                "method": "external_trigger",
                "details": {},
            },
        },
        "equity": {
            "start_equity": equity.get("start_equity"),
            "end_equity": equity.get("end_equity"),
            "pnl": equity.get("pnl"),
            "pnl_pct": equity.get("pnl_pct"),
            "note": equity.get("note"),
        },
        "trades": trades,
        "positions_end": positions_end,
        "execution_summary": execution_summary,
        "cost_summary": cost_summary,
        "performance_summary": performance_summary,
        "risk_model_health": risk_model_health,
        "risk_gate_decision": risk_gate_decision,
        "cov_coverage": (
            dict(snapshot.get("cov_coverage"))
            if isinstance(snapshot.get("cov_coverage"), dict)
            else default_cov_coverage()
        ),
        "returns_coverage_diag": (
            dict(snapshot.get("returns_coverage_diag"))
            if isinstance(snapshot.get("returns_coverage_diag"), dict)
            else {"schema_version": 1, "items": []}
        ),
        "ticker_proxy_used": bool(snapshot.get("ticker_proxy_used", False)),
        "ticker_proxy_map_used": (
            list(snapshot.get("ticker_proxy_map_used"))
            if isinstance(snapshot.get("ticker_proxy_map_used"), list)
            else []
        ),
        "asset_policy_mode": str(snapshot.get("asset_policy_mode") or "FORCE_PROXY"),
        "asset_policy_decisions": (
            list(snapshot.get("asset_policy_decisions"))
            if isinstance(snapshot.get("asset_policy_decisions"), list)
            else []
        ),
        "asset_policy_summary": (
            dict(snapshot.get("asset_policy_summary"))
            if isinstance(snapshot.get("asset_policy_summary"), dict)
            else {"counts": {"ALLOW_ORIGINAL": 0, "USE_PROXY": 0, "DISABLE": 0}, "top_reasons": []}
        ),
        "no_trade_summary": no_trade_summary,
        "risk": risk,
        "conviction": conviction,
        "meta": {
            "snapshot_path": snapshot_path,
            "trades_csv_path": trades_csv_path,
            "reports_dir": run_reports_dir,
            "timezone": tz,
            "account_id": snapshot_account_id or None,
            "session_id": snapshot_session_id or None,
            "env": snapshot_env or "live",
            "price_fetch_stats": price_fetch_stats,
            "price_fetch_summary": price_fetch_summary,
        },
    }
    report, _ = _ensure_report_meta_fields(report, snapshot=snapshot)
    return report


def write_daily_report(report_dict: Dict[str, Any], report_dirs: List[str]) -> List[str]:
    """Write report into all target directories atomically and refresh index."""
    if not isinstance(report_dict, dict):
        return []
    report_dict, _ = _ensure_report_meta_fields(report_dict)
    date_str = str(report_dict.get("date", "")).strip()
    if not date_str:
        return []

    meta = report_dict.get("meta", {}) if isinstance(report_dict.get("meta"), dict) else {}
    meta_reports_dir = _norm_text(meta.get("reports_dir"))
    if meta_reports_dir:
        raw_dirs = [meta_reports_dir]
    else:
        raw_dirs = report_dirs or [str(get_daily_report_dir("outputs"))]
    if isinstance(raw_dirs, str):  # type: ignore[unreachable]
        raw_dirs = [raw_dirs]  # type: ignore[assignment]

    wrote_paths: List[str] = []
    for report_dir_item in raw_dirs:
        if not report_dir_item:
            continue
        report_dir = os.path.abspath(str(report_dir_item))
        _ensure_dir(report_dir)
        report_path = os.path.join(report_dir, f"{date_str}.json")
        _atomic_write_json(report_path, report_dict)
        wrote_paths.append(report_path)
        try:
            _write_index(report_dir)
        except Exception:
            # Report file is more important than index refresh.
            pass
    return wrote_paths


def load_reports(report_dir: str, lookback_days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load report JSON files from one directory."""
    reports: List[Dict[str, Any]] = []
    if not report_dir or not os.path.isdir(report_dir):
        return reports

    for name in os.listdir(report_dir):
        if not name.lower().endswith(".json"):
            continue
        if name == INDEX_FILENAME:
            continue
        full_path = os.path.join(report_dir, name)
        payload = _safe_read_json(full_path)
        if not payload:
            continue
        date_str = str(payload.get("date", "")).strip()
        if not date_str:
            continue
        payload = dict(payload)
        payload["_path"] = full_path
        reports.append(payload)

    reports.sort(key=lambda x: str(x.get("date", "")))
    if lookback_days is None or lookback_days <= 0 or not reports:
        return reports

    latest_date = _parse_date(reports[-1].get("date"))
    if latest_date is None:
        return reports
    cutoff = latest_date - timedelta(days=int(lookback_days) - 1)
    out: List[Dict[str, Any]] = []
    for report in reports:
        rd = _parse_date(report.get("date"))
        if rd is not None and rd >= cutoff:
            out.append(report)
    return out


def aggregate_reports(reports: List[Dict[str, Any]], window_days: int) -> Dict[str, Any]:
    """Aggregate daily reports for a given trailing window."""
    if window_days <= 0:
        window_days = 1

    valid = [r for r in reports if isinstance(r, dict) and str(r.get("date", "")).strip()]
    valid.sort(key=lambda x: str(x.get("date", "")))
    available = len(valid)
    if available < window_days:
        return {
            "status": "insufficient",
            "window_days": int(window_days),
            "required_reports": int(window_days),
            "available_reports": int(available),
            "message": "时间不足",
        }

    selected = valid[-window_days:]
    buy_notional = 0.0
    sell_notional = 0.0
    net_flow = 0.0
    trade_count = 0

    pnl_sum = 0.0
    pnl_available = 0
    growth = 1.0
    pnl_pct_available = 0
    cost_total_sum = 0.0
    cost_fee_sum = 0.0
    cost_slippage_sum = 0.0
    cost_traded_notional_sum = 0.0

    risk_stat: Dict[str, Dict[str, Any]] = {}
    long_stat: Dict[str, Dict[str, Any]] = {}
    short_stat: Dict[str, Dict[str, Any]] = {}
    quality_issues: List[str] = []
    aggregate_quality = "ok"
    nt_orders_place = 0
    nt_orders_skip = 0
    nt_has_trade = False
    blocker_counter: Counter = Counter()
    gate_reason_counter: Counter = Counter()
    returns_missing_counter: Counter = Counter()
    last_gate_metric: Dict[str, Any] = {"metric_name": "", "metric_value": None, "threshold": None, "stage": ""}
    last_cov_known_weight = None
    last_cov_missing_weight_total = None
    latest_policy_mode = "FORCE_PROXY"
    latest_proxy_scope = "off"
    any_execution_proxy_used = False
    health_triggered_any = False
    health_trigger_count = 0
    health_reason_counter: Counter = Counter()
    health_metric_values: List[float] = []
    health_orders_place = 0
    health_orders_skip = 0
    health_missing_count = 0
    health_stale_count = 0
    health_live_count = 0
    health_cost_total = 0.0
    health_cost_bps_vals: List[float] = []
    health_daily_rows: List[Dict[str, Any]] = []

    for report in selected:
        trades = report.get("trades", {}) if isinstance(report.get("trades"), dict) else {}
        trade_quality = str(trades.get("data_quality", "ok") or "ok").strip().lower()
        if trade_quality == "inconsistent":
            aggregate_quality = "inconsistent"
            issues = trades.get("issues", [])
            if isinstance(issues, list):
                for issue in issues:
                    quality_issues.append(f"{report.get('date')}: {issue}")
        buy_notional += _as_float(trades.get("buy_notional"), 0.0)
        sell_notional += _as_float(trades.get("sell_notional"), 0.0)
        net_flow += _as_float(trades.get("net_flow"), 0.0)
        trade_count += int(_as_float(trades.get("trade_count"), 0.0))

        equity = report.get("equity", {}) if isinstance(report.get("equity"), dict) else {}
        pnl = equity.get("pnl")
        if pnl is not None:
            pnl_sum += _as_float(pnl, 0.0)
            pnl_available += 1
        pnl_pct = equity.get("pnl_pct")
        if pnl_pct is not None:
            growth *= (1.0 + _as_float(pnl_pct, 0.0) / 100.0)
            pnl_pct_available += 1

        cost_summary = report.get("cost_summary", {}) if isinstance(report.get("cost_summary"), dict) else {}
        totals = cost_summary.get("totals", {}) if isinstance(cost_summary.get("totals"), dict) else {}
        cost_total_sum += _as_float(totals.get("total"), _as_float(cost_summary.get("total"), 0.0))
        cost_fee_sum += _as_float(totals.get("fee"), _as_float(cost_summary.get("fee"), 0.0))
        cost_slippage_sum += _as_float(totals.get("slippage"), _as_float(cost_summary.get("slippage"), 0.0))
        cost_traded_notional_sum += _as_float(
            cost_summary.get("traded_notional"),
            abs(_as_float(trades.get("buy_notional"), 0.0)) + abs(_as_float(trades.get("sell_notional"), 0.0)),
        )

        risk = report.get("risk", {}) if isinstance(report.get("risk"), dict) else {}
        for item in risk.get("risky_tickers", []) if isinstance(risk.get("risky_tickers"), list) else []:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            score = _as_float(item.get("score"), 0.0)
            info = risk_stat.setdefault(
                ticker,
                {"ticker": ticker, "count": 0, "score_sum": 0.0, "last_score": 0.0, "last_evidence": {}, "last_date": ""},
            )
            info["count"] += 1
            info["score_sum"] += score
            info["last_score"] = score
            info["last_evidence"] = item.get("evidence", {})
            info["last_date"] = str(report.get("date", ""))

        conviction = report.get("conviction", {}) if isinstance(report.get("conviction"), dict) else {}
        for kind, stat in (("long_term", long_stat), ("short_term", short_stat)):
            items = conviction.get(kind, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker", "")).upper().strip()
                if not ticker:
                    continue
                why = str(item.get("why", "")).strip()
                info = stat.setdefault(ticker, {"ticker": ticker, "count": 0, "last_why": "", "last_date": ""})
                info["count"] += 1
                info["last_why"] = why or info["last_why"]
                info["last_date"] = str(report.get("date", ""))

        no_trade = report.get("no_trade_summary")
        if not isinstance(no_trade, dict):
            no_trade = build_no_trade_summary(
                trades=trades,
                snapshot=report,
                risk_gate_decision=report.get("risk_gate_decision") if isinstance(report.get("risk_gate_decision"), dict) else {},
                cov_coverage=report.get("cov_coverage") if isinstance(report.get("cov_coverage"), dict) else {},
                returns_coverage_diag=report.get("returns_coverage_diag") if isinstance(report.get("returns_coverage_diag"), dict) else {},
                asset_policy_mode=str(report.get("asset_policy_mode") or "FORCE_PROXY"),
                execution_proxy_used=bool(report.get("execution_proxy_used", report.get("ticker_proxy_used", False))),
                proxy_scope=_norm_text(report.get("proxy_scope") or report.get("ticker_proxy_scope")),
            )

        nt_orders_place += int(_as_float(no_trade.get("orders_place"), 0.0))
        nt_orders_skip += int(_as_float(no_trade.get("orders_skip"), 0.0))
        nt_has_trade = bool(nt_has_trade or bool(no_trade.get("has_trade", False)))

        blockers = no_trade.get("top_blockers", []) if isinstance(no_trade.get("top_blockers"), list) else []
        for row in blockers:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason", "")).strip().upper()
            if not reason:
                continue
            blocker_counter[reason] += int(_as_float(row.get("count"), 0.0))

        gate_reason = str(no_trade.get("gate_reason", "")).strip()
        if gate_reason:
            gate_reason_counter[gate_reason] += 1

        gate_metric = no_trade.get("gate_metric", {}) if isinstance(no_trade.get("gate_metric"), dict) else {}
        metric_name = str(gate_metric.get("metric_name", "")).strip()
        if metric_name:
            last_gate_metric = {
                "metric_name": metric_name,
                "metric_value": gate_metric.get("metric_value"),
                "threshold": gate_metric.get("threshold"),
                "stage": str(gate_metric.get("stage", "")).strip(),
            }

        data_issues = no_trade.get("data_issues", {}) if isinstance(no_trade.get("data_issues"), dict) else {}
        returns_missing_top = data_issues.get("returns_missing_top", []) if isinstance(data_issues.get("returns_missing_top"), list) else []
        for row in returns_missing_top:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper().strip()
            reason = str(row.get("reason", "")).upper().strip() or "UNKNOWN"
            if not ticker:
                continue
            returns_missing_counter[(ticker, reason)] += int(max(1, _as_float(row.get("count"), 1.0)))

        cov_known = _as_float(data_issues.get("cov_known_weight"), None)  # type: ignore[arg-type]
        cov_missing = _as_float(data_issues.get("cov_missing_weight_total"), None)  # type: ignore[arg-type]
        if cov_known is not None:
            last_cov_known_weight = cov_known
        if cov_missing is not None:
            last_cov_missing_weight_total = cov_missing

        policy = no_trade.get("policy", {}) if isinstance(no_trade.get("policy"), dict) else {}
        mode = str(policy.get("asset_policy_mode", "")).strip().upper()
        if mode:
            latest_policy_mode = mode
        scope = str(policy.get("proxy_scope", "")).strip().lower()
        if scope:
            latest_proxy_scope = scope
        any_execution_proxy_used = bool(any_execution_proxy_used or bool(policy.get("execution_proxy_used", False)))

        report_health = report.get("risk_model_health")
        if not isinstance(report_health, dict):
            report_health = build_risk_model_health(
                report=report,
                snapshot=report,
                daily_fields={
                    "no_trade_summary": no_trade,
                    "execution_summary": report.get("execution_summary") if isinstance(report.get("execution_summary"), dict) else {},
                    "risk_gate_decision": report.get("risk_gate_decision") if isinstance(report.get("risk_gate_decision"), dict) else {},
                    "cov_coverage": report.get("cov_coverage") if isinstance(report.get("cov_coverage"), dict) else {},
                    "returns_coverage_diag": report.get("returns_coverage_diag") if isinstance(report.get("returns_coverage_diag"), dict) else {"schema_version": 1, "items": []},
                    "asset_policy_mode": mode or latest_policy_mode,
                    "execution_proxy_used": bool(policy.get("execution_proxy_used", False)),
                    "cost_summary": report.get("cost_summary") if isinstance(report.get("cost_summary"), dict) else {},
                },
            )

        gate_obj = report_health.get("risk_gate", {}) if isinstance(report_health.get("risk_gate"), dict) else {}
        triggered = bool(gate_obj.get("triggered", False))
        reason = str(gate_obj.get("reason", "")).strip()
        metric_value = _as_float(gate_obj.get("metric_value"), None)  # type: ignore[arg-type]
        if triggered:
            health_trigger_count += 1
            health_triggered_any = True
        if reason:
            health_reason_counter[reason] += 1
        if metric_value is not None:
            health_metric_values.append(float(metric_value))

        ex_obj = report_health.get("execution", {}) if isinstance(report_health.get("execution"), dict) else {}
        health_orders_place += int(_as_float(ex_obj.get("orders_place"), 0.0))
        health_orders_skip += int(_as_float(ex_obj.get("orders_skip"), 0.0))

        prices_obj = report_health.get("prices", {}) if isinstance(report_health.get("prices"), dict) else {}
        health_missing_count += int(_as_float(prices_obj.get("missing_count"), 0.0))
        health_stale_count += int(_as_float(prices_obj.get("stale_count"), 0.0))
        health_live_count += int(_as_float(prices_obj.get("live_count"), 0.0))

        cost_obj = report_health.get("cost", {}) if isinstance(report_health.get("cost"), dict) else {}
        health_cost_total += float(_as_float(cost_obj.get("cost_total"), 0.0))
        cost_bps_val = _as_float(cost_obj.get("cost_bps"), None)  # type: ignore[arg-type]
        if cost_bps_val is not None:
            health_cost_bps_vals.append(float(cost_bps_val))

        cov_obj = report_health.get("coverage", {}) if isinstance(report_health.get("coverage"), dict) else {}
        health_daily_rows.append(
            {
                "date": str(report.get("date", "")),
                "triggered": bool(triggered),
                "reason": reason,
                "metric_value": metric_value,
                "returns_missing_count": int(_as_float(cov_obj.get("returns_missing_count"), 0.0)),
                "orders_place": int(_as_float(ex_obj.get("orders_place"), 0.0)),
                "cost_bps": cost_bps_val,
            }
        )

    top_risky = list(risk_stat.values())
    for item in top_risky:
        cnt = max(1, int(item["count"]))
        item["avg_score"] = float(item["score_sum"] / cnt)
    top_risky.sort(key=lambda x: (int(x["count"]), float(x["avg_score"])), reverse=True)
    top_risky = top_risky[:5]

    long_items = list(long_stat.values())
    long_items.sort(key=lambda x: int(x["count"]), reverse=True)
    short_items = list(short_stat.values())
    short_items.sort(key=lambda x: int(x["count"]), reverse=True)
    blockers_sorted = [{"reason": k, "count": int(v)} for k, v in blocker_counter.most_common(8)]
    gate_reason_final = gate_reason_counter.most_common(1)[0][0] if gate_reason_counter else ""
    returns_missing_top_out = []
    for (ticker, reason), count in returns_missing_counter.most_common(5):
        returns_missing_top_out.append({"ticker": ticker, "reason": reason, "count": int(count)})
    if last_cov_known_weight is None:
        last_cov_known_weight = _as_float(
            ((selected[-1].get("cov_coverage") or {}) if isinstance(selected[-1].get("cov_coverage"), dict) else {}).get("known_weight"),
            None,  # type: ignore[arg-type]
        )
    if last_cov_missing_weight_total is None:
        last_cov_missing_weight_total = _as_float(
            ((selected[-1].get("cov_coverage") or {}) if isinstance(selected[-1].get("cov_coverage"), dict) else {}).get("missing_weight_total"),
            None,  # type: ignore[arg-type]
        )
    if not latest_proxy_scope:
        latest_proxy_scope = "risk_and_execution" if any_execution_proxy_used else "off"
    no_trade_summary = {
        "schema_version": 1,
        "has_trade": bool(nt_has_trade or nt_orders_place > 0),
        "orders_place": int(max(0, nt_orders_place)),
        "orders_skip": int(max(0, nt_orders_skip)),
        "top_blockers": blockers_sorted,
        "gate_reason": gate_reason_final,
        "gate_metric": dict(last_gate_metric),
        "data_issues": {
            "returns_missing_top": returns_missing_top_out,
            "cov_known_weight": last_cov_known_weight,
            "cov_missing_weight_total": last_cov_missing_weight_total,
        },
        "policy": {
            "asset_policy_mode": latest_policy_mode,
            "execution_proxy_used": bool(any_execution_proxy_used),
            "proxy_scope": latest_proxy_scope,
        },
    }
    risk_model_health = {
        "schema_version": 1,
        "window_days": int(window_days),
        "report_count": int(len(selected)),
        "triggered_any": bool(health_triggered_any),
        "trigger_count": int(health_trigger_count),
        "gate_reason_top": [{"reason": k, "count": int(v)} for k, v in health_reason_counter.most_common(5)],
        "metric_avg": float(sum(health_metric_values) / len(health_metric_values)) if health_metric_values else None,
        "metric_max": float(max(health_metric_values)) if health_metric_values else None,
        "execution": {
            "orders_place": int(health_orders_place),
            "orders_skip": int(health_orders_skip),
        },
        "prices": {
            "missing_count": int(health_missing_count),
            "stale_count": int(health_stale_count),
            "live_count": int(health_live_count),
        },
        "cost": {
            "cost_total": float(health_cost_total),
            "cost_bps_avg": float(sum(health_cost_bps_vals) / len(health_cost_bps_vals)) if health_cost_bps_vals else None,
        },
        "daily_rows": health_daily_rows[-7:],
    }

    return {
        "status": "ok",
        "data_quality": aggregate_quality,
        "issues": quality_issues,
        "window_days": int(window_days),
        "from": str(selected[0].get("date", "")),
        "to": str(selected[-1].get("date", "")),
        "report_count": len(selected),
        "metrics": {
            "buy_notional": float(buy_notional),
            "sell_notional": float(sell_notional),
            "net_flow": float(net_flow),
            "trade_count": int(trade_count),
            "pnl": float(pnl_sum) if pnl_available > 0 else None,
            "pnl_pct": float((growth - 1.0) * 100.0) if pnl_pct_available > 0 else None,
            "pnl_available_days": int(pnl_available),
            "cost_total": float(cost_total_sum),
            "cost_fee": float(cost_fee_sum),
            "cost_slippage": float(cost_slippage_sum),
            "cost_bps": float(cost_total_sum / cost_traded_notional_sum * 10000.0) if cost_traded_notional_sum > 1e-12 else 0.0,
        },
        "top_risky_tickers": top_risky,
        "long_term_stats": long_items[:10],
        "short_term_stats": short_items[:10],
        "no_trade_summary": no_trade_summary,
        "risk_model_health": risk_model_health,
    }


def _run_smoke_from_cli(args: argparse.Namespace) -> int:
    if not args.smoke:
        print("Refusing to run without --smoke (to avoid polluting live outputs).")
        return 2
    snapshot_path = args.snapshot or os.path.join("outputs", "snapshot_live.json")
    trades_path = args.trades or os.path.join("outputs", "paper_trades.csv")
    report_date = args.date or datetime.now(_coerce_zone(DEFAULT_TZ)).date().isoformat()
    with tempfile.TemporaryDirectory(prefix="daily_report_smoke_") as td:
        report = generate_daily_report(
            date=report_date,
            snapshot_path=snapshot_path,
            trades_csv_path=trades_path,
            report_dirs=[td],
            tz=args.tz or DEFAULT_TZ,
        )
        paths = write_daily_report(report, [td])
        print(f"SMOKE_OK date={report.get('date')} files={len(paths)} dir={td}")
    return 0


def _run_session_smoke_tests() -> int:
    """Lightweight logic-only tests for market session + close gating."""
    et_tz = _coerce_zone(MARKET_TZ)
    snap = {"stale_ratio": 1.0, "observe_count": 10}

    def _decide_report_date(session_obj: Dict[str, Any], closed: bool, reason_obj: Dict[str, Any]) -> Tuple[str, str]:
        state = str(session_obj.get("state", "")).upper()
        trading_date = str(session_obj.get("trading_date_et", "")).strip()
        last_completed = str(session_obj.get("last_completed_trading_date_et", "")).strip()
        method = str(reason_obj.get("method", "")).strip().lower() if isinstance(reason_obj, dict) else ""
        is_time_close = bool(closed and method == "time")
        is_stale_close = bool(closed and method == "stale_streak" and state == "OPEN")
        trigger_mode = "backfill_last_completed"
        report_date = last_completed
        if is_time_close:
            trigger_mode = "post_close_time"
            report_date = trading_date or last_completed
        elif is_stale_close:
            trigger_mode = "open_stale_streak"
            report_date = trading_date or last_completed
        return trigger_mode, report_date

    # case1: PRE_OPEN must not be closed by stale, and streak must reset.
    tracker1 = {"streak": 10, "ratio_threshold": 0.8, "threshold": 3}
    now1 = datetime(2026, 2, 10, 5, 0, tzinfo=et_tz)  # Tue 05:00 ET
    session1 = get_market_session_state(now1)
    closed1, reason1 = is_market_closed(now=now1, tz=DEFAULT_TZ, snapshot=snap, stale_tracker=tracker1)
    assert session1.get("state") == "PRE_OPEN"
    assert not bool(closed1)
    assert int(_as_float(tracker1.get("streak"), -1)) == 0
    assert str(session1.get("trading_date_et")) == "2026-02-10"
    assert str(session1.get("last_completed_trading_date_et")) == "2026-02-09"
    mode1, report_date1 = _decide_report_date(session1, closed1, reason1)
    assert mode1 == "backfill_last_completed"
    assert report_date1 == "2026-02-09"
    assert report_date1 != str(session1.get("trading_date_et"))

    # case2: OPEN (+ open grace passed) may close by stale streak.
    tracker2 = {"streak": 2, "ratio_threshold": 0.8, "threshold": 3}
    now2 = datetime(2026, 2, 10, 10, 30, tzinfo=et_tz)  # Tue 10:30 ET
    session2 = get_market_session_state(now2)
    closed2, reason2 = is_market_closed(now=now2, tz=DEFAULT_TZ, snapshot=snap, stale_tracker=tracker2)
    assert session2.get("state") == "OPEN"
    assert bool(session2.get("open_grace_passed"))
    assert bool(closed2)
    assert str(reason2.get("method", "")).lower() == "stale_streak"
    mode2, report_date2 = _decide_report_date(session2, closed2, reason2)
    assert mode2 == "open_stale_streak"
    assert report_date2 == str(session2.get("trading_date_et"))

    # case3: POST_CLOSE (+ close grace passed) closes by time.
    tracker3 = {"streak": 0, "ratio_threshold": 0.8, "threshold": 3}
    now3 = datetime(2026, 2, 10, 16, 15, tzinfo=et_tz)  # Tue 16:15 ET
    session3 = get_market_session_state(now3)
    closed3, reason3 = is_market_closed(now=now3, tz=DEFAULT_TZ, snapshot=snap, stale_tracker=tracker3)
    assert session3.get("state") == "POST_CLOSE"
    assert bool(session3.get("close_grace_passed"))
    assert bool(closed3)
    assert str(reason3.get("method", "")).lower() == "time"
    mode3, report_date3 = _decide_report_date(session3, closed3, reason3)
    assert mode3 == "post_close_time"
    assert report_date3 == str(session3.get("trading_date_et"))

    print("SESSION_SMOKE_OK")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily reporter utility (safe smoke mode).")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test in a temporary directory.")
    parser.add_argument("--smoke-session", action="store_true", help="Run session/close logic smoke tests.")
    parser.add_argument("--date", type=str, default=None, help="Report date (YYYY-MM-DD).")
    parser.add_argument("--snapshot", type=str, default=None, help="Snapshot json path.")
    parser.add_argument("--trades", type=str, default=None, help="Trades csv path.")
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ, help="Timezone name.")
    _args = parser.parse_args()
    if _args.smoke_session:
        sys.exit(_run_session_smoke_tests())
    sys.exit(_run_smoke_from_cli(_args))
