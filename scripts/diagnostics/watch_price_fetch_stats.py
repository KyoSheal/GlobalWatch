#!/usr/bin/env python3
"""Watch snapshot_live.json and print per-cycle price fetch diagnostics."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from atomic_io import safe_read_json
from price_service import PriceService


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_cycle_id(snapshot: Dict[str, Any]) -> Optional[int]:
    for key in ("cycle", "current_cycle", "cycle_index"):
        if key in snapshot:
            try:
                return int(float(snapshot.get(key)))
            except Exception:
                continue
    return None


def _format_row(cycle_id: int, stats: Dict[str, Any]) -> str:
    status = str(stats.get("status", "unknown"))
    calls = _to_int(stats.get("batch_calls"), 0)
    hits = _to_int(stats.get("cache_hits"), 0)
    misses = _to_int(stats.get("cache_misses"), 0)
    uniq = _to_int(stats.get("symbols_unique"), 0)
    miss5m = _to_int(stats.get("missing_after_pass1"), 0)
    fill1m = _to_int(stats.get("fetched_by_1m"), 0)
    tz_bad = _to_int(stats.get("tz_ok_false"), 0)
    elapsed_ms = _to_int(stats.get("elapsed_ms"), 0)
    denom = hits + misses
    hit_rate = (100.0 * hits / denom) if denom > 0 else 0.0
    return (
        f"[PRICE_FETCH_DIAG] cycle={cycle_id} status={status} calls={calls} "
        f"hit={hits} miss={misses} hit_rate={hit_rate:.1f}% uniq={uniq} "
        f"miss5m={miss5m} fill1m={fill1m} tz_bad={tz_bad} ms={elapsed_ms}"
    )


def _warn_if_needed(rows: List[Dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        if i == 0:
            continue
        cycle_id = row.get("cycle_id")
        stats = row.get("stats", {})
        if not isinstance(stats, dict):
            continue
        calls = _to_int(stats.get("batch_calls"), 0)
        hits = _to_int(stats.get("cache_hits"), 0)
        misses = _to_int(stats.get("cache_misses"), 0)
        status = str(stats.get("status", "unknown"))
        denom = hits + misses
        hit_rate = (100.0 * hits / denom) if denom > 0 else 0.0
        if calls > 2:
            print(f"[WARN] cycle={cycle_id} batch_calls={calls} > 2")
        if hit_rate < 70.0:
            print(f"[WARN] cycle={cycle_id} hit_rate={hit_rate:.1f}% < 70%")
        if status != "ok":
            print(f"[WARN] cycle={cycle_id} status={status}")


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    calls = []
    hit_rates = []
    miss5m_total = 0
    fill1m_total = 0
    max_ms = 0
    for row in rows:
        stats = row.get("stats", {})
        if not isinstance(stats, dict):
            continue
        c = _to_int(stats.get("batch_calls"), 0)
        h = _to_int(stats.get("cache_hits"), 0)
        m = _to_int(stats.get("cache_misses"), 0)
        denom = h + m
        hr = (100.0 * h / denom) if denom > 0 else 0.0
        calls.append(c)
        hit_rates.append(hr)
        miss5m_total += _to_int(stats.get("missing_after_pass1"), 0)
        fill1m_total += _to_int(stats.get("fetched_by_1m"), 0)
        max_ms = max(max_ms, _to_int(stats.get("elapsed_ms"), 0))

    avg_calls = statistics.fmean(calls) if calls else 0.0
    avg_hit_rate = statistics.fmean(hit_rates) if hit_rates else 0.0
    print(
        f"[PRICE_FETCH_SUMMARY] n={len(rows)} avg_calls={avg_calls:.2f} "
        f"avg_hit_rate={avg_hit_rate:.1f}% total_miss5m={miss5m_total} "
        f"total_fill1m={fill1m_total} max_ms={max_ms}"
    )


def run(snapshot_path: str, cycles: int, poll_ms: int, timeout_s: int) -> int:
    target = max(1, int(cycles))
    poll_sleep = max(1, int(poll_ms)) / 1000.0
    timeout_sec = max(1, int(timeout_s))
    started = time.time()

    rows: List[Dict[str, Any]] = []
    seen_cycles = set()

    while len(rows) < target and (time.time() - started) < timeout_sec:
        payload = safe_read_json(snapshot_path, retries=2, sleep_ms=15)
        if isinstance(payload, dict):
            cycle_id = _extract_cycle_id(payload)
            stats = payload.get("price_fetch_stats")
            if cycle_id is not None and isinstance(stats, dict) and bool(stats) and cycle_id not in seen_cycles:
                seen_cycles.add(cycle_id)
                row = {"cycle_id": cycle_id, "stats": dict(stats)}
                rows.append(row)
                print(_format_row(cycle_id, stats))
        time.sleep(poll_sleep)

    _warn_if_needed(rows)
    _print_summary(rows)

    if len(rows) >= target:
        return 0

    print(
        f"[PRICE_FETCH_DIAG_TIMEOUT] collected={len(rows)} required={target} "
        f"timeout_s={timeout_sec} snapshot={os.path.abspath(snapshot_path)}"
    )
    return 1


def _parse_tickers(raw: str) -> List[str]:
    out: List[str] = []
    for item in str(raw or "").split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def run_active(
    tickers: List[str],
    rounds: int,
    sleep_s: float,
    *,
    interval: str,
    period: str,
    chunk: int,
    allow_1m_fallback: bool,
) -> int:
    target = max(1, int(rounds))
    sleep_sec = max(0.0, float(sleep_s))
    svc = PriceService()

    rows: List[Dict[str, Any]] = []
    for i in range(target):
        stats = svc.prefetch(
            tickers,
            interval=interval,
            period=period,
            max_chunk=max(1, int(chunk)),
            allow_1m_fallback=bool(allow_1m_fallback),
        )
        row = {"cycle_id": i + 1, "stats": dict(stats) if isinstance(stats, dict) else {}}
        rows.append(row)
        print(_format_row(i + 1, row["stats"]))
        if i < target - 1 and sleep_sec > 0:
            time.sleep(sleep_sec)

    _warn_if_needed(rows)
    _print_summary(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch snapshot price_fetch_stats across cycles.")
    parser.add_argument("--mode", choices=["passive", "active"], default="passive")
    parser.add_argument("--snapshot", default=os.path.join("outputs", "snapshot_live.json"))
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--poll-ms", type=int, default=250)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--tickers", default="SPY,QQQ,AAPL,MSFT,GLD")
    parser.add_argument("--sleep-s", type=float, default=1.0)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--period", default="1d")
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--no-1m-fallback", action="store_true")
    args = parser.parse_args()
    if args.mode == "active":
        tickers = _parse_tickers(args.tickers)
        if not tickers:
            print("[PRICE_FETCH_ACTIVE] empty tickers list")
            return 1
        return run_active(
            tickers=tickers,
            rounds=int(args.cycles),
            sleep_s=float(args.sleep_s),
            interval=str(args.interval),
            period=str(args.period),
            chunk=int(args.chunk),
            allow_1m_fallback=not bool(args.no_1m_fallback),
        )
    return run(
        snapshot_path=str(args.snapshot),
        cycles=int(args.cycles),
        poll_ms=int(args.poll_ms),
        timeout_s=int(args.timeout_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
