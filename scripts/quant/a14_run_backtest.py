#!/usr/bin/env python3
"""A4-2 CLI: run offline backtest from returns + target weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from backtest_engine import load_returns, load_weights, run_backtest, write_backtest


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.load(open(path, "r", encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _filter_by_date(rows: List[Dict[str, Any]], start: str, end: str, key: str = "date") -> List[Dict[str, Any]]:
    if not start and not end:
        return list(rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = str(r.get(key) or "")
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(r)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run deterministic offline backtest.")
    p.add_argument("--cache-dir", required=True, help="A4-1 cache directory containing returns_daily.csv")
    p.add_argument("--weights", required=True, help="weights.csv or weights.json")
    p.add_argument("--out-dir", default="", help="default outputs/backtests/<hash>")
    p.add_argument("--initial-equity", type=float, default=100000.0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    weights_path = Path(args.weights).resolve()
    if not cache_dir.exists():
        print(f"[ERROR] cache dir not found: {cache_dir}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"[ERROR] weights file not found: {weights_path}", file=sys.stderr)
        return 2

    returns_rows = load_returns(cache_dir)
    weights_rows = load_weights(weights_path)
    if args.start or args.end:
        returns_rows = _filter_by_date(returns_rows, str(args.start or ""), str(args.end or ""))
        weights_rows = _filter_by_date(weights_rows, "", str(args.end or ""))

    cache_manifest = _read_json(cache_dir / "manifest.json")
    req_hash = str(((cache_manifest.get("request") or {}).get("hash")) or "")
    if not req_hash:
        req_hash = _canonical_hash(
            [{"date": r.get("date"), "ticker": r.get("ticker"), "ret": r.get("ret")} for r in returns_rows]
        )
    weights_hash = _canonical_hash(
        [{"date": r.get("date"), "ticker": r.get("ticker"), "weight": r.get("weight")} for r in weights_rows]
    )
    params = {
        "initial_equity": float(args.initial_equity),
        "cost_bps": float(args.cost_bps),
        "rebalance": str(args.rebalance),
        "start": str(args.start or ""),
        "end": str(args.end or ""),
    }
    run_hash = _canonical_hash({"request_hash": req_hash, "weights_hash": weights_hash, "params": params})

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (ROOT / "outputs" / "backtests" / run_hash).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        equity_rows, trades_rows, manifest = run_backtest(
            returns_rows,
            weights_rows,
            initial_equity=float(args.initial_equity),
            cost_bps=float(args.cost_bps),
            rebalance_rule=str(args.rebalance),
        )
    except Exception as exc:
        print(f"[ERROR] backtest failed: {exc}", file=sys.stderr)
        return 2

    manifest["inputs"] = {
        "cache_dir": str(cache_dir),
        "weights_path": str(weights_path),
        "returns_rows": len(returns_rows),
        "weights_rows": len(weights_rows),
        "request_hash": req_hash,
        "weights_hash": weights_hash,
        "run_hash": run_hash,
    }
    manifest["params"] = params
    write_info = write_backtest(out_dir, equity_rows, trades_rows, manifest)

    if args.verbose:
        print(f"[INFO] out_dir={out_dir}")
        print(f"[INFO] equity_rows={len(equity_rows)} trades_rows={len(trades_rows)}")
        print(f"[INFO] report={write_info.get('report_md')}")
        print("[PASS] a14_run_backtest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
