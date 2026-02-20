#!/usr/bin/env python3
"""A4-3 utilities: extract deterministic daily target weights from run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from quant_io_utils import iter_jsonl, parse_iso_to_utc, safe_read_json


WEIGHTS_COLUMNS = ["date", "ticker", "weight"]
SNAPSHOT_TS_KEYS = ["time_utc", "timestamp_utc", "ts_utc", "now_utc", "time", "timestamp", "ts"]
TARGET_WEIGHT_KEYS = ["target_weights", "final_target_weights", "weights", "target_allocation"]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(v: Any) -> Optional[float]:
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None


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
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                out = {c: row.get(c, "") for c in columns}
                w.writerow(out)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _resolve_tz(name: str) -> timezone:
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(str(name or "America/New_York"))
    except Exception:
        return timezone.utc


def _extract_ts_utc(obj: Dict[str, Any], fallback_path: Optional[Path] = None) -> Optional[datetime]:
    for key in SNAPSHOT_TS_KEYS:
        dt = parse_iso_to_utc(obj.get(key))
        if dt is not None:
            return dt
    payload = obj.get("payload")
    if isinstance(payload, dict):
        for key in SNAPSHOT_TS_KEYS:
            dt = parse_iso_to_utc(payload.get(key))
            if dt is not None:
                return dt
    if fallback_path is not None and fallback_path.exists():
        try:
            return datetime.fromtimestamp(float(fallback_path.stat().st_mtime), tz=timezone.utc)
        except Exception:
            return None
    return None


def _extract_target_weights(obj: Dict[str, Any]) -> Dict[str, float]:
    candidate: Any = None
    for key in TARGET_WEIGHT_KEYS:
        if isinstance(obj.get(key), dict):
            candidate = obj.get(key)
            break
    if candidate is None and isinstance(obj.get("payload"), dict):
        payload = obj.get("payload") or {}
        for key in TARGET_WEIGHT_KEYS:
            if isinstance(payload.get(key), dict):
                candidate = payload.get(key)
                break
    if not isinstance(candidate, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in candidate.items():
        t = str(k or "").strip().upper()
        if not t:
            continue
        fv = _as_float(v)
        if fv is None:
            continue
        out[t] = float(fv)
    return out


def _discover_source(run_dir: Path) -> Tuple[str, Optional[Path]]:
    direct = run_dir / "portfolio_snapshots.jsonl"
    if direct.exists():
        return "portfolio_snapshots_jsonl", direct

    candidates = list(run_dir.rglob("portfolio_snapshots.jsonl"))
    if candidates:
        candidates.sort(key=lambda p: (float(p.stat().st_mtime), str(p)), reverse=True)
        return "portfolio_snapshots_jsonl", candidates[0]

    live = run_dir / "snapshot_live.json"
    if live.exists():
        obj = safe_read_json(live)
        if isinstance(obj, dict):
            tw = _extract_target_weights(obj)
            if tw:
                return "snapshot_live_json", live
    return "missing", None


def _iter_snapshots(source_kind: str, source_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {"bad_lines": 0, "rows_total": 0, "rows_used": 0}
    if source_kind == "snapshot_live_json":
        obj = safe_read_json(source_path) or {}
        stats["rows_total"] = 1
        ts_utc = _extract_ts_utc(obj, fallback_path=source_path)
        tw = _extract_target_weights(obj)
        if ts_utc is not None and tw:
            rows.append({"ts_utc": ts_utc, "target_weights": tw})
            stats["rows_used"] = 1
        return rows, stats

    for _lineno, obj, err in iter_jsonl(source_path):
        stats["rows_total"] += 1
        if err is not None or not isinstance(obj, dict):
            stats["bad_lines"] += 1
            continue
        ts_utc = _extract_ts_utc(obj)
        tw = _extract_target_weights(obj)
        if ts_utc is None or not tw:
            continue
        rows.append({"ts_utc": ts_utc, "target_weights": tw})
        stats["rows_used"] += 1
    return rows, stats


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_day_weights(raw: Dict[str, float], warnings: List[str], date_str: str) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for ticker, w in raw.items():
        t = str(ticker or "").strip().upper()
        if not t:
            continue
        fv = _as_float(w)
        if fv is None:
            continue
        clean[t] = max(0.0, float(fv))

    cash = float(clean.pop("CASH", 0.0))
    non_cash_sum = sum(clean.values())
    total = non_cash_sum + cash
    if total <= 0:
        warnings.append(f"{date_str}:empty_or_non_positive_weights->cash_1")
        return {"CASH": 1.0}

    if total > 1.0 + 1e-12:
        scale = 1.0 / total
        for t in list(clean.keys()):
            clean[t] = clean[t] * scale
        cash = cash * scale
        warnings.append(f"{date_str}:scaled_weights_sum_gt_1")

    total_after = sum(clean.values()) + cash
    residual = 1.0 - total_after
    cash = cash + residual
    if cash < 0 and abs(cash) < 1e-9:
        cash = 0.0

    out = {t: float(w) for t, w in sorted(clean.items(), key=lambda kv: kv[0])}
    out["CASH"] = float(cash)
    return out


def build_daily_weights(
    run_dir: Path,
    *,
    report_tz: str = "America/New_York",
    date_start: str = "",
    date_end: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(run_dir).resolve()
    source_kind, source_path = _discover_source(root)
    if source_path is None:
        raise FileNotFoundError(f"no snapshot source found under run_dir={root}")

    snapshots, stats = _iter_snapshots(source_kind, source_path)
    if not snapshots:
        raise ValueError(f"no usable snapshots with target_weights found in {source_path}")

    tz = _resolve_tz(report_tz)
    daily_last: Dict[str, Dict[str, Any]] = {}
    for row in snapshots:
        ts_utc = row.get("ts_utc")
        if not isinstance(ts_utc, datetime):
            continue
        date_local = ts_utc.astimezone(tz).date().isoformat()
        if date_start and date_local < date_start:
            continue
        if date_end and date_local > date_end:
            continue
        prev = daily_last.get(date_local)
        if prev is None or ts_utc > prev.get("ts_utc"):
            daily_last[date_local] = {"ts_utc": ts_utc, "target_weights": dict(row.get("target_weights") or {})}

    if not daily_last:
        raise ValueError("no snapshots left after date filtering")

    warnings: List[str] = []
    out_rows: List[Dict[str, Any]] = []
    for date_local in sorted(daily_last.keys()):
        weights = _normalize_day_weights(daily_last[date_local]["target_weights"], warnings, date_local)
        for ticker, weight in weights.items():
            out_rows.append({"date": date_local, "ticker": ticker, "weight": float(weight)})

    out_rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    rows_hash = _canonical_hash(
        [{"date": r["date"], "ticker": r["ticker"], "weight": round(float(r["weight"]), 12)} for r in out_rows]
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "run_dir": str(root),
        "source_kind": source_kind,
        "source_file": str(source_path.resolve()),
        "report_tz": str(report_tz),
        "date_start": str(date_start or ""),
        "date_end": str(date_end or ""),
        "days": len(sorted(daily_last.keys())),
        "rows": len(out_rows),
        "warnings": warnings,
        "stats": stats,
        "hash": rows_hash,
    }
    return out_rows, manifest


def write_weights(out_dir: Path, rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Dict[str, str]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "weights.csv"
    manifest_path = out / "weights_manifest.json"
    _write_csv(csv_path, rows, WEIGHTS_COLUMNS)
    _write_json_atomic(manifest_path, manifest)
    return {"weights_csv": str(csv_path), "manifest_json": str(manifest_path)}

