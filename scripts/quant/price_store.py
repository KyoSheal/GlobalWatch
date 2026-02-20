#!/usr/bin/env python3
"""A4-1 backtest data layer: prices/returns cache utilities."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PRICE_COLUMNS = ["date", "ticker", "adj_close"]
RET_COLUMNS = ["date", "ticker", "ret"]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
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


def _normalize_price_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        d = str(row.get("date") or "").strip()
        t = str(row.get("ticker") or "").strip().upper()
        px = _num_or_none(row.get("adj_close"))
        if not d or not t or px is None:
            continue
        out.append({"date": d, "ticker": t, "adj_close": float(px)})
    out.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return out


def _normalize_ret_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        d = str(row.get("date") or "").strip()
        t = str(row.get("ticker") or "").strip().upper()
        rv = _num_or_none(row.get("ret"))
        if not d or not t or rv is None:
            continue
        out.append({"date": d, "ticker": t, "ret": float(rv)})
    out.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return out


def load_prices(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load prices_daily.csv from cache_dir."""
    path = Path(cache_dir).resolve() / "prices_daily.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return _normalize_price_rows(rows)


def load_returns(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load returns_daily.csv from cache_dir."""
    path = Path(cache_dir).resolve() / "returns_daily.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return _normalize_ret_rows(rows)


def validate_prices(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate prices and return warnings/stats."""
    normalized = _normalize_price_rows(rows)
    warnings: List[str] = []
    key_seen = set()
    duplicates = 0
    bad_dates = 0
    by_ticker: Dict[str, List[str]] = {}

    for row in normalized:
        d = str(row["date"])
        t = str(row["ticker"])
        key = (d, t)
        if key in key_seen:
            duplicates += 1
        key_seen.add(key)
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except Exception:
            bad_dates += 1
        by_ticker.setdefault(t, []).append(d)

    non_monotonic_tickers: List[str] = []
    for t, dates in by_ticker.items():
        if dates != sorted(dates):
            non_monotonic_tickers.append(t)

    missing_rate = 0.0
    if by_ticker:
        all_dates = sorted(set([r["date"] for r in normalized]))
        total_expected = len(all_dates) * len(by_ticker)
        total_actual = len(normalized)
        if total_expected > 0:
            missing_rate = max(0.0, float((total_expected - total_actual) / total_expected))

    if duplicates > 0:
        warnings.append(f"duplicates:{duplicates}")
    if bad_dates > 0:
        warnings.append(f"bad_dates:{bad_dates}")
    if non_monotonic_tickers:
        warnings.append(f"non_monotonic:{','.join(sorted(non_monotonic_tickers)[:10])}")
    if missing_rate > 0.2:
        warnings.append(f"high_missing_rate:{missing_rate:.4f}")

    return {
        "rows": len(normalized),
        "tickers": sorted(by_ticker.keys()),
        "date_range": {
            "start": min([r["date"] for r in normalized]) if normalized else "",
            "end": max([r["date"] for r in normalized]) if normalized else "",
        },
        "duplicates": duplicates,
        "bad_dates": bad_dates,
        "non_monotonic_tickers": sorted(non_monotonic_tickers),
        "missing_rate": float(missing_rate),
        "warnings": warnings,
    }


def compute_returns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute simple daily returns by ticker from normalized price rows."""
    normalized = _normalize_price_rows(rows)
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for row in normalized:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)

    ret_rows: List[Dict[str, Any]] = []
    for ticker, trows in by_ticker.items():
        trows.sort(key=lambda r: str(r["date"]))
        prev = None
        for row in trows:
            px = float(row["adj_close"])
            if prev is None or prev <= 0:
                prev = px
                continue
            ret = (px / prev) - 1.0
            ret_rows.append({"date": str(row["date"]), "ticker": ticker, "ret": float(ret)})
            prev = px

    ret_rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return ret_rows


def save_prices(
    rows: List[Dict[str, Any]],
    cache_dir: Path,
    *,
    source: str = "unknown",
    tickers: Optional[List[str]] = None,
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save prices_daily.csv + manifest.json into cache_dir."""
    cache = Path(cache_dir).resolve()
    normalized = _normalize_price_rows(rows)
    validation = validate_prices(normalized)
    _write_csv(cache / "prices_daily.csv", normalized, PRICE_COLUMNS)

    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "source": str(source),
        "rows": int(validation.get("rows", 0)),
        "tickers": list(tickers or validation.get("tickers", [])),
        "date_range": validation.get("date_range", {}),
        "validation": validation,
        "request": request or {},
    }
    _write_json_atomic(cache / "manifest.json", manifest)
    return manifest


def save_returns(rows: List[Dict[str, Any]], cache_dir: Path) -> Dict[str, Any]:
    """Save returns_daily.csv and update manifest with returns stats."""
    cache = Path(cache_dir).resolve()
    normalized = _normalize_ret_rows(rows)
    _write_csv(cache / "returns_daily.csv", normalized, RET_COLUMNS)

    manifest_path = cache / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        except Exception:
            manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest["returns"] = {
        "rows": len(normalized),
        "date_range": {
            "start": min([r["date"] for r in normalized]) if normalized else "",
            "end": max([r["date"] for r in normalized]) if normalized else "",
        },
    }
    manifest["updated_utc"] = _now_utc_iso()
    _write_json_atomic(manifest_path, manifest)
    return manifest
