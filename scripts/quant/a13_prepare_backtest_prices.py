#!/usr/bin/env python3
"""A4-1 CLI: prepare reusable backtest prices/returns cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from price_store import compute_returns, load_prices, save_prices, save_returns


def _parse_date(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        raise ValueError("date is required")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().isoformat()
    except Exception as exc:
        raise ValueError(f"invalid date {s!r}; expected YYYY-MM-DD") from exc


def _load_tickers_from_universe_file(path: Path) -> List[str]:
    p = path.resolve()
    if not p.exists():
        raise FileNotFoundError(f"universe file not found: {p}")
    tickers: List[str] = []
    if p.suffix.lower() == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, list):
            for x in obj:
                t = str(x or "").strip().upper()
                if t:
                    tickers.append(t)
        elif isinstance(obj, dict):
            seq = obj.get("tickers")
            if not isinstance(seq, list):
                seq = obj.get("universe")
            if not isinstance(seq, list):
                seq = obj.get("assets")
            if isinstance(seq, list):
                for x in seq:
                    t = str(x or "").strip().upper()
                    if t:
                        tickers.append(t)
    else:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
                if t:
                    tickers.append(t)
    out = sorted(set(tickers))
    if not out:
        raise ValueError(f"no tickers found in universe file: {p}")
    return out


def _parse_tickers_arg(text: str) -> List[str]:
    out = [x.strip().upper() for x in str(text or "").split(",") if x.strip()]
    return sorted(set(out))


def _request_hash(tickers: List[str], start: str, end: str, source: str) -> str:
    payload = json.dumps(
        {"tickers": sorted(set(tickers)), "start": start, "end": end, "source": source},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_dir(base: Path, req_hash: str) -> Path:
    return (base.resolve() / req_hash).resolve()


def _filter_rows(rows: List[Dict[str, Any]], tickers: List[str], start: str, end: str) -> List[Dict[str, Any]]:
    tick = set([x.upper() for x in tickers])
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        d = str(row.get("date") or "")
        t = str(row.get("ticker") or "").upper()
        if t not in tick:
            continue
        if d < start or d > end:
            continue
        out.append({"date": d, "ticker": t, "adj_close": row.get("adj_close")})
    out.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return out


def _fetch_yfinance_rows(tickers: List[str], start: str, end: str) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise RuntimeError("yfinance not available; install yfinance or use --source csv/offline") from exc

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() + timedelta(days=1)
    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        hist = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
        if hist is None or getattr(hist, "empty", True):
            continue
        cols = set([str(c) for c in getattr(hist, "columns", [])])
        price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
        if not price_col:
            continue
        series = hist[price_col]
        for idx, value in series.items():
            d = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
            try:
                px = float(value)
            except Exception:
                continue
            rows.append({"date": d, "ticker": ticker.upper(), "adj_close": px})
    rows.sort(key=lambda r: (str(r["date"]), str(r["ticker"])))
    return rows


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    p = path.resolve()
    if not p.exists():
        raise FileNotFoundError(f"csv source not found: {p}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "date": row.get("date"),
                    "ticker": row.get("ticker") or row.get("symbol"),
                    "adj_close": row.get("adj_close") or row.get("close") or row.get("price"),
                }
            )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare backtest price/returns cache.")
    p.add_argument("--tickers", default="", help="Comma-separated tickers")
    p.add_argument("--universe-file", default="", help="JSON/CSV file containing tickers")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--cache-base", default="outputs/backtest_cache")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "csv", "offline"])
    p.add_argument("--csv-path", default="", help="CSV source path when --source csv")
    p.add_argument("--offline", action="store_true", default=False, help="Validate/reuse existing cache only")
    p.add_argument("--force", action="store_true", default=False, help="Force refresh even when cache exists")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("[ERROR] --end must be >= --start")

    tickers = _parse_tickers_arg(args.tickers)
    if str(args.universe_file or "").strip():
        from_file = _load_tickers_from_universe_file(Path(args.universe_file))
        tickers = sorted(set(tickers + from_file))
    if not tickers:
        raise SystemExit("[ERROR] no tickers provided (use --tickers or --universe-file)")

    src = str(args.source or "yfinance").strip().lower()
    if bool(args.offline):
        src = "offline"

    req_hash = _request_hash(tickers, start, end, src)
    cache_dir = _cache_dir(Path(args.cache_base), req_hash)
    manifest_path = cache_dir / "manifest.json"
    prices_path = cache_dir / "prices_daily.csv"
    returns_path = cache_dir / "returns_daily.csv"

    if manifest_path.exists() and prices_path.exists() and returns_path.exists() and not bool(args.force):
        existing = json.load(open(manifest_path, "r", encoding="utf-8"))
        existing_req = existing.get("request") if isinstance(existing.get("request"), dict) else {}
        if (
            str(existing_req.get("hash") or "") == req_hash
            and str(existing_req.get("start") or "") == start
            and str(existing_req.get("end") or "") == end
        ):
            if args.verbose:
                print(f"[INFO] reuse cache_dir={cache_dir}")
                print("[PASS] a13_prepare_backtest_prices")
            return 0

    if src == "offline":
        if not prices_path.exists():
            raise SystemExit(f"[ERROR] offline mode requires existing cache prices: {prices_path}")
        price_rows = _filter_rows(load_prices(cache_dir), tickers, start, end)
    elif src == "csv":
        csv_path = Path(args.csv_path).resolve() if str(args.csv_path or "").strip() else prices_path
        raw_rows = _load_csv_rows(csv_path)
        price_rows = _filter_rows(raw_rows, tickers, start, end)
    else:
        price_rows = _fetch_yfinance_rows(tickers, start, end)
        price_rows = _filter_rows(price_rows, tickers, start, end)

    if not price_rows:
        raise SystemExit("[ERROR] no price rows prepared for the requested range/tickers")

    manifest = save_prices(
        price_rows,
        cache_dir,
        source=src,
        tickers=tickers,
        request={"hash": req_hash, "start": start, "end": end, "tickers": tickers, "source": src},
    )
    ret_rows = compute_returns(price_rows)
    manifest = save_returns(ret_rows, cache_dir)
    validation = manifest.get("validation", {}) if isinstance(manifest, dict) else {}

    if args.verbose:
        print(f"[INFO] cache_dir={cache_dir}")
        print(f"[INFO] prices_rows={len(price_rows)} returns_rows={len(ret_rows)}")
        print(f"[INFO] warnings={validation.get('warnings', [])}")
        print("[PASS] a13_prepare_backtest_prices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
