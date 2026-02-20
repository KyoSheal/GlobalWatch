#!/usr/bin/env python3
"""T51: minimal regression test for A4-5 attach backtest summary to daily + index."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str((ROOT / "scripts" / "quant").resolve()) not in sys.path:
    sys.path.insert(0, str((ROOT / "scripts" / "quant").resolve()))

from price_store import compute_returns, save_prices, save_returns


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="backtest_attach_daily_min_"))
    try:
        outputs_base = (tmp / "outputs").resolve()
        daily_base = (outputs_base / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)

        date_str = "2026-02-18"
        daily_json = daily_base / f"{date_str}.json"
        with daily_json.open("w", encoding="utf-8") as f:
            json.dump({"date": date_str, "summary": {}}, f, ensure_ascii=False, indent=2)

        run_dir = outputs_base / "2026-02" / "20260218-1200-a45demo"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "portfolio_snapshots.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"time_utc": "2026-02-18T15:00:00+00:00", "target_weights": {"AAA": 0.8}}) + "\n")
            f.write(json.dumps({"time_utc": "2026-02-18T20:00:00+00:00", "target_weights": {"AAA": 0.7, "BBB": 0.2}}) + "\n")

        price_store = outputs_base / "price_store" / "cache_demo"
        prices = [
            {"date": "2026-02-17", "ticker": "AAA", "adj_close": 100.0},
            {"date": "2026-02-18", "ticker": "AAA", "adj_close": 101.0},
            {"date": "2026-02-17", "ticker": "BBB", "adj_close": 50.0},
            {"date": "2026-02-18", "ticker": "BBB", "adj_close": 49.0},
        ]
        save_prices(
            prices,
            price_store,
            source="csv",
            tickers=["AAA", "BBB"],
            request={"hash": "t51-demo", "start": "2026-02-17", "end": "2026-02-18", "tickers": ["AAA", "BBB"]},
        )
        save_returns(compute_returns(prices), price_store)

        out_pack_dir = daily_base / "quant_packs" / date_str / "backtest_from_run"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a17_attach_backtest_to_daily.py"),
            "--daily-base",
            str(daily_base),
            "--date",
            date_str,
            "--outputs-base",
            str(outputs_base),
            "--price-store",
            str(outputs_base / "price_store"),
            "--out-pack-dir",
            str(out_pack_dir),
            "--strict",
            "--verbose",
        ]
        p1 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a17 first run failed rc={p1.returncode}")

        # idempotent second run
        p2 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a17 second run failed rc={p2.returncode}")

        obj = json.load(open(daily_json, "r", encoding="utf-8"))
        qpack = obj.get("quant_pack") if isinstance(obj.get("quant_pack"), dict) else {}
        bfr = qpack.get("backtest_from_run") if isinstance(qpack.get("backtest_from_run"), dict) else {}
        if str(bfr.get("status", "")) != "OK":
            return _fail(f"backtest_from_run status not OK: {bfr.get('status')}")
        for key in ("total_return", "max_drawdown", "days", "trade_rows", "rebalance_count", "total_cost"):
            if key not in bfr:
                return _fail(f"missing key in backtest_from_run: {key}")

        bak = daily_json.with_name(daily_json.name + ".bak")
        if not bak.exists():
            return _fail("missing daily json .bak after attach")

        attach_manifest = out_pack_dir / "attach_manifest.json"
        if not attach_manifest.exists():
            return _fail("missing attach_manifest.json")

        # update index and validate quant.backtest_from_run projection
        p3 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "365",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p3.returncode != 0:
            print(p3.stdout)
            print(p3.stderr)
            return _fail(f"a7 failed rc={p3.returncode}")

        index_obj = json.load(open(daily_base / "daily_reports_index.json", "r", encoding="utf-8"))
        reports = index_obj.get("reports") if isinstance(index_obj.get("reports"), list) else []
        hit = None
        for r in reports:
            if isinstance(r, dict) and str(r.get("date", "")) == date_str:
                hit = r
                break
        if not isinstance(hit, dict):
            return _fail("index missing date row")
        q = hit.get("quant") if isinstance(hit.get("quant"), dict) else {}
        q_bt = q.get("backtest_from_run") if isinstance(q.get("backtest_from_run"), dict) else {}
        if "total_return" not in q_bt:
            return _fail("index quant.backtest_from_run.total_return missing")

        print("[PASS] backtest_attach_daily_minimal")
        print(f"[INFO] daily_json={daily_json}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

