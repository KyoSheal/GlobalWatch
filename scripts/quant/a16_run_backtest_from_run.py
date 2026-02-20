#!/usr/bin/env python3
"""A4-4 CLI: one-shot offline backtest from run_dir artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from backtest_engine import run_backtest, write_backtest
from price_store import compute_returns, save_prices, save_returns
from quant_io_utils import safe_read_json
from weights_from_run import build_daily_weights, write_weights


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_float(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _load_prices_rows(cache_dir: Path) -> List[Dict[str, Any]]:
    path = cache_dir / "prices_daily.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            d = str(row.get("date") or "").strip()
            t = str(row.get("ticker") or "").strip().upper()
            px = _as_float(row.get("adj_close"))
            if not d or not t or px is None:
                continue
            rows.append({"date": d, "ticker": t, "adj_close": float(px)})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def _load_returns_rows(cache_dir: Path) -> List[Dict[str, Any]]:
    path = cache_dir / "returns_daily.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            d = str(row.get("date") or "").strip()
            t = str(row.get("ticker") or "").strip().upper()
            rv = _as_float(row.get("ret"))
            if not d or not t or rv is None:
                continue
            rows.append({"date": d, "ticker": t, "ret": float(rv)})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def _filter_rows_by_request(
    rows: List[Dict[str, Any]],
    tickers: List[str],
    date_start: str,
    date_end: str,
    value_key: str,
) -> List[Dict[str, Any]]:
    tick = set([str(t).upper() for t in tickers])
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = str(row.get("date") or "")
        t = str(row.get("ticker") or "").upper()
        if t not in tick:
            continue
        if date_start and d < date_start:
            continue
        if date_end and d > date_end:
            continue
        out.append({"date": d, "ticker": t, value_key: row.get(value_key)})
    out.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return out


def _read_prices_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"price csv not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            d = str(row.get("date") or "").strip()
            t = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            px = _as_float(row.get("adj_close") or row.get("close") or row.get("price"))
            if not d or not t or px is None:
                continue
            rows.append({"date": d, "ticker": t, "adj_close": float(px)})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def _discover_cache_candidates(price_store_root: Path) -> List[Path]:
    if not price_store_root.exists():
        return []
    out: List[Path] = []
    for p in price_store_root.rglob("manifest.json"):
        cache_dir = p.parent
        if (cache_dir / "returns_daily.csv").exists() and (cache_dir / "prices_daily.csv").exists():
            out.append(cache_dir)
    out.sort(key=lambda p: (float(p.stat().st_mtime), str(p)), reverse=True)
    return out


def _cache_satisfies(cache_dir: Path, tickers: List[str], date_start: str, date_end: str) -> bool:
    prices = _load_prices_rows(cache_dir)
    filtered = _filter_rows_by_request(prices, tickers, date_start, date_end, "adj_close")
    if not filtered:
        return False
    by_ticker_dates: Dict[str, set] = {}
    for row in filtered:
        by_ticker_dates.setdefault(str(row["ticker"]), set()).add(str(row["date"]))
    if not all(t in by_ticker_dates for t in tickers):
        return False
    # at least one row in [date_start, date_end] for each ticker
    return True


def _ensure_cache_for_request(
    price_store_root: Path,
    tickers: List[str],
    date_start: str,
    date_end: str,
    *,
    price_csv: str = "",
    out_prices_dir: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    warnings: List[str] = []
    for cache_dir in _discover_cache_candidates(price_store_root):
        if _cache_satisfies(cache_dir, tickers, date_start, date_end):
            return cache_dir, warnings

    if not str(price_csv or "").strip():
        raise FileNotFoundError(
            f"no suitable price cache under {price_store_root}; provide --price-csv to seed local cache"
        )

    seeded_dir = (out_prices_dir or (price_store_root / "seeded")).resolve()
    source_rows = _read_prices_csv(Path(price_csv).resolve())
    source_rows = _filter_rows_by_request(source_rows, tickers, date_start, date_end, "adj_close")
    if not source_rows:
        raise ValueError("price csv has no rows matching requested tickers/date range")

    cache_hash = _canonical_hash(
        {
            "tickers": sorted(tickers),
            "date_start": date_start,
            "date_end": date_end,
            "source_csv": str(Path(price_csv).resolve()),
        }
    )
    cache_dir = seeded_dir / cache_hash
    save_prices(
        source_rows,
        cache_dir,
        source="csv",
        tickers=tickers,
        request={"hash": cache_hash, "start": date_start, "end": date_end, "tickers": tickers, "source": "csv"},
    )
    save_returns(compute_returns(source_rows), cache_dir)
    warnings.append("price_cache_seeded_from_csv")
    return cache_dir, warnings


def _weights_tickers_and_range(rows: List[Dict[str, Any]]) -> Tuple[List[str], str, str]:
    dates = sorted(set([str(r.get("date") or "") for r in rows if str(r.get("date") or "")]))
    tickers = sorted(set([str(r.get("ticker") or "").upper() for r in rows if str(r.get("ticker") or "").upper() not in ("", "CASH")]))
    if not dates:
        raise ValueError("weights rows are empty")
    if not tickers:
        raise ValueError("weights rows contain only CASH; no tradable tickers")
    return tickers, dates[0], dates[-1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one-shot offline backtest directly from run_dir.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--price-store", default="outputs/price_store")
    p.add_argument("--out-dir", default="")
    p.add_argument("--report-tz", default="America/New_York")
    p.add_argument("--date-start", default="")
    p.add_argument("--date-end", default="")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--initial-equity", type=float, default=100000.0)
    p.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    p.add_argument("--price-csv", default="", help="Optional local CSV to seed price cache (no network)")
    p.add_argument("--verbose", action="store_true")
    return p


def run_backtest_from_run(
    *,
    run_dir: Path,
    price_store_root: Path,
    out_root: Optional[Path] = None,
    report_tz: str = "America/New_York",
    date_start: str = "",
    date_end: str = "",
    cost_bps: float = 5.0,
    initial_equity: float = 100000.0,
    rebalance: str = "daily",
    price_csv: str = "",
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        return 2, {"error": f"run dir not found: {run_dir}"}
    price_store_root = Path(price_store_root).resolve()

    # Step A: extract daily weights from run
    try:
        weights_rows, weights_manifest = build_daily_weights(
            run_dir,
            report_tz=str(report_tz),
            date_start=str(date_start or ""),
            date_end=str(date_end or ""),
        )
    except Exception as exc:
        return 2, {"error": f"failed to extract weights from run: {exc}"}

    try:
        tickers, inferred_start, inferred_end = _weights_tickers_and_range(weights_rows)
    except Exception as exc:
        return 2, {"error": f"invalid extracted weights: {exc}"}

    date_start = str(date_start or inferred_start)
    date_end = str(date_end or inferred_end)

    run_hash = _canonical_hash(
        {
            "run_dir": str(run_dir),
            "weights_hash": weights_manifest.get("hash"),
            "tickers": tickers,
            "date_start": date_start,
            "date_end": date_end,
            "cost_bps": float(cost_bps),
            "initial_equity": float(initial_equity),
            "rebalance": str(rebalance),
        }
    )
    out_root = Path(out_root).resolve() if out_root is not None and str(out_root).strip() else (ROOT / "outputs" / "backtests" / run_hash).resolve()
    weights_out = out_root / "weights"
    prices_out = out_root / "prices"
    backtest_out = out_root / "backtest"
    out_root.mkdir(parents=True, exist_ok=True)

    write_weights(weights_out, weights_rows, weights_manifest)

    # Step B/C: ensure price cache (offline only; existing cache or local CSV seed)
    warnings: List[str] = []
    try:
        cache_dir, cache_warnings = _ensure_cache_for_request(
            price_store_root,
            tickers,
            date_start,
            date_end,
            price_csv=str(price_csv or ""),
            out_prices_dir=prices_out / "cache",
        )
        warnings.extend(cache_warnings)
    except Exception as exc:
        return 2, {"error": f"failed to prepare price cache: {exc}"}

    # Create prices manifest in output tree (reference + small summary)
    source_manifest = safe_read_json(cache_dir / "manifest.json") or {}
    prices_ref_manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "price_store_path": str(cache_dir),
        "source_manifest": source_manifest,
        "request": {"tickers": tickers, "date_start": date_start, "date_end": date_end},
    }
    _write_json_atomic(prices_out / "manifest.json", prices_ref_manifest)

    # Step D: run offline backtest with filtered returns
    returns_rows = _load_returns_rows(cache_dir)
    returns_rows = _filter_rows_by_request(returns_rows, tickers, date_start, date_end, "ret")
    if not returns_rows:
        return 2, {"error": "no returns rows available after filtering"}

    weights_rows_for_bt = [dict(r) for r in weights_rows if date_start <= str(r.get("date") or "") <= date_end]
    try:
        eq_rows, tr_rows, bt_manifest = run_backtest(
            returns_rows,
            weights_rows_for_bt,
            initial_equity=float(initial_equity),
            cost_bps=float(cost_bps),
            rebalance_rule=str(rebalance),
        )
    except Exception as exc:
        return 2, {"error": f"backtest run failed: {exc}"}

    bt_manifest["inputs"] = {
        "run_dir": str(run_dir),
        "weights_path": str((weights_out / "weights.csv").resolve()),
        "price_store_path": str(cache_dir),
        "returns_rows": len(returns_rows),
        "weights_rows": len(weights_rows_for_bt),
    }
    bt_manifest["params"] = {
        "initial_equity": float(initial_equity),
        "cost_bps": float(cost_bps),
        "rebalance": str(rebalance),
        "date_start": date_start,
        "date_end": date_end,
    }
    bt_write_info = write_backtest(backtest_out, eq_rows, tr_rows, bt_manifest)

    manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "run_dir": str(run_dir),
        "weights_path": str((weights_out / "weights.csv").resolve()),
        "price_store_path": str(cache_dir),
        "backtest_out_dir": str(backtest_out.resolve()),
        "date_range": {"start": date_start, "end": date_end},
        "tickers_count": len(tickers),
        "tickers": tickers,
        "warnings": sorted(set(list(weights_manifest.get("warnings") or []) + warnings)),
        "hash": run_hash,
        "outputs": {
            "weights_manifest": str((weights_out / "weights_manifest.json").resolve()),
            "prices_manifest": str((prices_out / "manifest.json").resolve()),
            "backtest_report": bt_write_info.get("report_md"),
            "backtest_manifest": bt_write_info.get("manifest_json"),
        },
    }
    _write_json_atomic(out_root / "backtest_from_run_manifest.json", manifest)

    if verbose:
        print(f"[INFO] run_dir={run_dir}")
        print(f"[INFO] out_dir={out_root}")
        print(f"[INFO] date_range={date_start}..{date_end} tickers={len(tickers)}")
        print(f"[INFO] price_store_path={cache_dir}")
        print(f"[INFO] report={backtest_out / 'backtest_report.md'}")
        print("[PASS] a16_run_backtest_from_run")
    return 0, manifest


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    price_store_root = Path(args.price_store).resolve()
    out_root = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else None
    rc, info = run_backtest_from_run(
        run_dir=run_dir,
        price_store_root=price_store_root,
        out_root=out_root,
        report_tz=str(args.report_tz),
        date_start=str(args.date_start or ""),
        date_end=str(args.date_end or ""),
        cost_bps=float(args.cost_bps),
        initial_equity=float(args.initial_equity),
        rebalance=str(args.rebalance),
        price_csv=str(args.price_csv or ""),
        verbose=bool(args.verbose),
    )
    if rc != 0:
        print(f"[ERROR] {info.get('error', 'a16 failed')}", file=sys.stderr)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
