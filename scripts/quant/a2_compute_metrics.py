#!/usr/bin/env python3
"""A1-2: Compute performance/risk metrics from extracted run dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_metrics import compute_metrics, load_dataset, render_metrics_markdown

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_daily_returns(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["date_local", "close_equity", "daily_return"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "date_local": row.get("date_local", ""),
                    "close_equity": row.get("close_equity", ""),
                    "daily_return": row.get("daily_return", ""),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute quant metrics from A1 dataset output.")
    p.add_argument("--dataset-dir", required=True, help="Path to dataset dir with manifest/equity/cycles/trades.")
    p.add_argument("--out-dir", default="", help="Output directory. Defaults to <dataset-dir>/metrics.")
    p.add_argument("--report-tz", default="America/New_York", help="Timezone for daily aggregation.")
    p.add_argument("--annualization", type=int, default=252, help="Annualization factor (default 252).")
    p.add_argument("--rf", type=float, default=0.0, help="Annualized risk-free rate.")
    p.add_argument("--min-points", type=int, default=5, help="Minimum equity points before marking insufficient.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (dataset_dir / "metrics").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_dir)
    metrics, daily_returns = compute_metrics(
        dataset,
        dataset_dir=dataset_dir,
        report_tz=str(args.report_tz or "America/New_York"),
        annualization=int(args.annualization),
        rf_annual=float(args.rf),
        min_points=int(args.min_points),
    )
    metrics_md = render_metrics_markdown(metrics)

    metrics_json_path = out_dir / "metrics.json"
    metrics_md_path = out_dir / "metrics.md"
    daily_returns_path = out_dir / "daily_returns.csv"

    _write_json(metrics_json_path, metrics)
    _write_text(metrics_md_path, metrics_md)
    _write_daily_returns(daily_returns_path, daily_returns)

    if args.verbose:
        meta = metrics.get("meta", {}) or {}
        perf = metrics.get("performance", {}) or {}
        risk = metrics.get("risk", {}) or {}
        trading = metrics.get("trading", {}) or {}
        print(f"[INFO] dataset_dir={dataset_dir}")
        print(f"[INFO] out_dir={out_dir}")
        print(
            "[INFO] "
            f"run_id={meta.get('run_id')} "
            f"equity_points={meta.get('equity_points')} "
            f"trades={meta.get('trade_rows')} "
            f"cycles={meta.get('cycle_rows')}"
        )
        print(
            "[INFO] "
            f"total_return={perf.get('total_return')} "
            f"max_drawdown={risk.get('max_drawdown')} "
            f"sharpe={risk.get('sharpe')} "
            f"trades_total={trading.get('trades_total')}"
        )
        print("[PASS] a2_compute_metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

