#!/usr/bin/env python3
"""A4-10: Build quant alerts from index_timeseries/index."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_io_utils import safe_read_json
from a19_build_index_timeseries import build_index_timeseries

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _load_timeseries(daily_base: Path, lookback_days: int, verbose: bool) -> List[Dict[str, Any]]:
    ts_json = (daily_base / "index_timeseries.json").resolve()
    ts_csv = (daily_base / "index_timeseries.csv").resolve()
    if not ts_json.exists() or not ts_csv.exists():
        build_index_timeseries(daily_base, lookback_days=lookback_days, verbose=verbose)
    obj = safe_read_json(ts_json) or {}
    rows = obj.get("rows") if isinstance(obj.get("rows"), list) else []
    if rows:
        return [dict(r) for r in rows if isinstance(r, dict)]

    out: List[Dict[str, Any]] = []
    if ts_csv.exists():
        with ts_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    out.append(dict(row))
    return out


def _rolling_last(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    return rows[-n:]


def _gate_fail_streak(rows: List[Dict[str, Any]], n: int = 2) -> List[str]:
    streak = 0
    hit_dates: List[str] = []
    for row in rows:
        s = str(row.get("gate_status", "") or "").upper()
        if s == "FAIL":
            streak += 1
            hit_dates.append(str(row.get("date", "")))
        else:
            streak = 0
            hit_dates = []
        if streak >= n:
            return hit_dates[-n:]
    return []


def _drift_missing_streak(rows: List[Dict[str, Any]], n: int = 3) -> List[str]:
    target = {"MISSING", "NOT_RUN"}
    streak = 0
    hit_dates: List[str] = []
    for row in rows:
        s = str(row.get("replay_drift_status", "") or "").upper()
        if s in target:
            streak += 1
            hit_dates.append(str(row.get("date", "")))
        else:
            streak = 0
            hit_dates = []
        if streak >= n:
            return hit_dates[-n:]
    return []


def _exec_blocker_dominant(rows: List[Dict[str, Any]], n: int = 3) -> Optional[Dict[str, Any]]:
    last_n = _rolling_last(rows, n)
    if len(last_n) < n:
        return None
    reasons = [str(r.get("exec_blocker_top1_reason", "") or "").strip().lower() for r in last_n]
    if not reasons or any(not x for x in reasons):
        return None
    if len(set(reasons)) != 1:
        return None
    blocked = [_num_or_none(r.get("exec_blocked_ratio")) for r in last_n]
    if any(x is None or float(x) < 0.8 for x in blocked):
        return None
    top1_ratios = [_num_or_none(r.get("exec_blocker_top1_ratio")) for r in last_n]
    return {
        "reason": reasons[0],
        "dates": [str(r.get("date", "")) for r in last_n],
        "blocked_ratio_list": [float(x) if x is not None else None for x in blocked],
        "top1_ratio_list": [float(x) if x is not None else None for x in top1_ratios],
    }


def _no_trade_streak(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    streak = 0
    dates: List[str] = []
    primary_reason = ""
    for row in reversed(rows):
        if _boolish(row.get("no_trade_flag")):
            streak += 1
            dates.append(str(row.get("date", "")))
            if not primary_reason:
                primary_reason = str(row.get("no_trade_primary_reason", "") or "")
        else:
            break
    dates.reverse()
    return {"streak": int(streak), "dates": dates, "primary_reason": str(primary_reason)}


def build_quant_alerts(
    daily_base: Path,
    lookback_days: int,
    *,
    cost_fragile_threshold_bps: float = 8.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    daily_base = daily_base.resolve()
    rows = _load_timeseries(daily_base, lookback_days=lookback_days, verbose=verbose)
    rows.sort(key=lambda r: str(r.get("date", "")))

    alerts: List[Dict[str, Any]] = []

    # rule1 cooldown_dominant
    last3 = _rolling_last(rows, 3)
    if last3:
        cool = sum(1 for r in last3 if str(r.get("gating_top1", "") or "").strip().lower() == "attempt_cooldown")
        ratio = cool / max(1, len(last3))
        if ratio >= 0.67:
            alerts.append(
                {
                    "rule_id": "cooldown_dominant",
                    "severity": "warn",
                    "window": f"last_{len(last3)}d",
                    "dates": [str(r.get("date", "")) for r in last3],
                    "evidence": {"ratio": ratio, "cooldown_days": cool},
                }
            )

    # rule2 gate_fail_streak
    gate_dates = _gate_fail_streak(rows, n=2)
    if gate_dates:
        alerts.append(
            {
                "rule_id": "gate_fail_streak",
                "severity": "high",
                "window": "streak_2d",
                "dates": gate_dates,
                "evidence": {"gate_status": "FAIL"},
            }
        )

    # rule3 reconcile_gap_large
    gap_dates: List[str] = []
    for r in rows:
        rg = _num_or_none(r.get("reconcile_return_gap"))
        tg = _num_or_none(r.get("reconcile_turnover_gap"))
        if (rg is not None and abs(rg) > 0.005) or (tg is not None and abs(tg) > 0.0):
            gap_dates.append(str(r.get("date", "")))
    if gap_dates:
        alerts.append(
            {
                "rule_id": "reconcile_gap_large",
                "severity": "warn",
                "window": f"{len(gap_dates)}d_hits",
                "dates": gap_dates[-10:],
                "evidence": {"return_gap_threshold": 0.005, "turnover_gap_threshold": 0.0},
            }
        )

    # rule4 drift_missing_streak
    drift_dates = _drift_missing_streak(rows, n=3)
    if drift_dates:
        alerts.append(
            {
                "rule_id": "drift_missing_streak",
                "severity": "warn",
                "window": "streak_3d",
                "dates": drift_dates,
                "evidence": {"statuses": ["MISSING", "NOT_RUN"]},
            }
        )

    # rule5 cost_fragile
    fragile_hits: List[Dict[str, Any]] = []
    for r in rows:
        status = str(r.get("backtest_sweep_status", "") or "").upper()
        be = _num_or_none(r.get("break_even_cost_bps"))
        if status == "OK" and be is not None and float(be) < float(cost_fragile_threshold_bps):
            fragile_hits.append(
                {
                    "date": str(r.get("date", "")),
                    "break_even_cost_bps": float(be),
                    "sensitivity_per_1bp": _num_or_none(r.get("sensitivity_per_1bp")),
                }
            )
    if fragile_hits:
        alerts.append(
            {
                "rule_id": "cost_fragile",
                "severity": "warn",
                "window": f"{len(fragile_hits)}d_hits",
                "dates": [str(x.get("date", "")) for x in fragile_hits[-10:]],
                "evidence": {
                    "threshold_bps": float(cost_fragile_threshold_bps),
                    "hits": fragile_hits[-10:],
                },
            }
        )

    # rule6 exec_blocker_dominant_cyclelevel
    dominant = _exec_blocker_dominant(rows, n=3)
    if dominant is not None:
        alerts.append(
            {
                "rule_id": "exec_blocker_dominant_cyclelevel",
                "severity": "warn",
                "window": "last_3d",
                "dates": dominant["dates"],
                "evidence": {
                    "reason": dominant["reason"],
                    "blocked_ratio_list": dominant["blocked_ratio_list"],
                    "top1_ratio_list": dominant["top1_ratio_list"],
                },
            }
        )

    # rule7 no_trade_day
    nts = _no_trade_streak(rows)
    if int(nts.get("streak", 0)) >= 1:
        streak = int(nts["streak"])
        alerts.append(
            {
                "rule_id": "no_trade_day",
                "severity": "warn" if streak >= 2 else "info",
                "window": f"streak_{streak}d",
                "dates": list(nts.get("dates", [])),
                "evidence": {"primary_reason": str(nts.get("primary_reason", "") or "unknown")},
            }
        )

    # deterministic order by predefined rule then date
    order = {
        "cooldown_dominant": 1,
        "gate_fail_streak": 2,
        "reconcile_gap_large": 3,
        "drift_missing_streak": 4,
        "cost_fragile": 5,
        "exec_blocker_dominant_cyclelevel": 6,
        "no_trade_day": 7,
    }
    alerts.sort(key=lambda a: (order.get(str(a.get("rule_id", "")), 999), ",".join(a.get("dates", []))))

    out_json = {
        "schema_version": 1,
        "generated_at_utc": _now_utc_iso(),
        "daily_base": str(daily_base),
        "rows_considered": len(rows),
        "alerts_count": len(alerts),
        "alerts": alerts,
    }

    lines: List[str] = []
    lines.append("# Quant Alerts")
    lines.append("")
    lines.append(f"- generated_utc: `{out_json['generated_at_utc']}`")
    lines.append(f"- rows_considered: `{len(rows)}`")
    lines.append(f"- alerts_count: `{len(alerts)}`")
    lines.append("")
    if alerts:
        for a in alerts:
            lines.append(f"## {a.get('rule_id')}")
            lines.append(f"- severity: {a.get('severity')}")
            lines.append(f"- window: {a.get('window')}")
            lines.append(f"- dates: {', '.join(a.get('dates', []))}")
            lines.append(f"- evidence: `{json.dumps(a.get('evidence', {}), ensure_ascii=False)}`")
            lines.append("")
    else:
        lines.append("No alerts.")
        lines.append("")

    alerts_json_path = (daily_base / "alerts.json").resolve()
    alerts_md_path = (daily_base / "alerts.md").resolve()
    _write_json_atomic(alerts_json_path, out_json)
    _write_text_atomic(alerts_md_path, "\n".join(lines))

    if verbose:
        print(f"[A20] rows={len(rows)} alerts={len(alerts)}")
        print(f"[A20] alerts_json={alerts_json_path}")
        print("[PASS] a20_build_quant_alerts")

    return {
        "alerts_json": str(alerts_json_path),
        "alerts_md": str(alerts_md_path),
        "alerts_count": len(alerts),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build quant alerts from index timeseries.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--lookback-days", type=int, default=60)
    p.add_argument("--cost-fragile-threshold-bps", type=float, default=8.0)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2
    try:
        build_quant_alerts(
            daily_base,
            lookback_days=int(args.lookback_days),
            cost_fragile_threshold_bps=float(args.cost_fragile_threshold_bps),
            verbose=bool(args.verbose),
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
