"""Daily report generation and aggregation utilities for paper trading."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import argparse
from datetime import date as date_cls
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


DEFAULT_TZ = "America/Vancouver"
MARKET_TZ = "America/New_York"
DEFAULT_MAIN_REPORT_DIR = os.path.join("outputs", "Daily Report")
DEFAULT_MIRROR_REPORT_DIR = r"C:\Users\kyosh\Desktop\Project\News\outputs\Daily Report"
INDEX_FILENAME = "daily_reports_index.json"


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
        if not os.path.exists(path) or os.path.getsize(path) <= 2:
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_report_dirs(report_dirs: Optional[List[str]]) -> List[str]:
    raw = report_dirs or [DEFAULT_MAIN_REPORT_DIR, DEFAULT_MIRROR_REPORT_DIR]
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
        dedup = [os.path.abspath(DEFAULT_MAIN_REPORT_DIR)]
    return dedup


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
    normalized_dirs = _normalize_report_dirs(report_dirs)

    existing, existing_path = _find_existing_report(date_str, normalized_dirs)
    if existing is not None:
        if _is_existing_report_usable(existing):
            existing["_already_exists"] = True
            existing["_existing_path"] = existing_path
            return existing

    snapshot = _safe_read_json(snapshot_path) or {}
    snapshot_account_id = _norm_text(snapshot.get("account_id")) or "paper_main"
    snapshot_session_id = _norm_text(snapshot.get("session_id"))
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

    generated_at = datetime.now(_coerce_zone(tz)).isoformat()
    report = {
        "date": date_str,
        "generated_at_local": generated_at,
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
        "risk": risk,
        "conviction": conviction,
        "meta": {
            "snapshot_path": snapshot_path,
            "trades_csv_path": trades_csv_path,
            "timezone": tz,
            "account_id": snapshot_account_id or None,
            "session_id": snapshot_session_id or None,
            "env": snapshot_env or "live",
        },
    }
    return report


def write_daily_report(report_dict: Dict[str, Any], report_dirs: List[str]) -> List[str]:
    """Write report into all target directories atomically and refresh index."""
    if not isinstance(report_dict, dict):
        return []
    date_str = str(report_dict.get("date", "")).strip()
    if not date_str:
        return []

    raw_dirs = report_dirs or [DEFAULT_MAIN_REPORT_DIR, DEFAULT_MIRROR_REPORT_DIR]
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

    risk_stat: Dict[str, Dict[str, Any]] = {}
    long_stat: Dict[str, Dict[str, Any]] = {}
    short_stat: Dict[str, Dict[str, Any]] = {}
    quality_issues: List[str] = []
    aggregate_quality = "ok"

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
        },
        "top_risky_tickers": top_risky,
        "long_term_stats": long_items[:10],
        "short_term_stats": short_items[:10],
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

    # case2: OPEN (+ open grace passed) may close by stale streak.
    tracker2 = {"streak": 2, "ratio_threshold": 0.8, "threshold": 3}
    now2 = datetime(2026, 2, 10, 10, 30, tzinfo=et_tz)  # Tue 10:30 ET
    session2 = get_market_session_state(now2)
    closed2, reason2 = is_market_closed(now=now2, tz=DEFAULT_TZ, snapshot=snap, stale_tracker=tracker2)
    assert session2.get("state") == "OPEN"
    assert bool(session2.get("open_grace_passed"))
    assert bool(closed2)
    assert str(reason2.get("method", "")).lower() == "stale_streak"

    # case3: POST_CLOSE (+ close grace passed) closes by time.
    tracker3 = {"streak": 0, "ratio_threshold": 0.8, "threshold": 3}
    now3 = datetime(2026, 2, 10, 16, 15, tzinfo=et_tz)  # Tue 16:15 ET
    session3 = get_market_session_state(now3)
    closed3, reason3 = is_market_closed(now=now3, tz=DEFAULT_TZ, snapshot=snap, stale_tracker=tracker3)
    assert session3.get("state") == "POST_CLOSE"
    assert bool(session3.get("close_grace_passed"))
    assert bool(closed3)
    assert str(reason3.get("method", "")).lower() == "time"

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
