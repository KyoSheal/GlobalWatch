#!/usr/bin/env python3
"""A4-5: attach A4-4 backtest summary into flat Daily Report JSON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from a16_run_backtest_from_run import run_backtest_from_run
from quant_daily_pack import discover_run_dir_for_date
from quant_io_utils import safe_read_json


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_date(s: str) -> Optional[str]:
    text = str(s or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _discover_latest_date(daily_base: Path) -> Optional[str]:
    candidates: List[str] = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        d = _parse_date(p.stem)
        if d is not None:
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


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


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_name(path.name + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        return bak
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bak2 = path.with_name(path.name + f".{stamp}.bak")
    shutil.copy2(path, bak2)
    return bak2


def _ensure_daily_json(path: Path, *, date_str: str, strict: bool) -> Tuple[Optional[Dict[str, Any]], List[str], bool]:
    warnings: List[str] = []
    created = False
    obj = safe_read_json(path)
    if obj is not None:
        return obj, warnings, created
    if path.exists() and strict:
        return None, [f"invalid_daily_json:{path}"], created
    if not path.exists() and strict:
        return None, [f"missing_daily_json:{path}"], created
    obj = {
        "schema_version": 1,
        "date": date_str,
        "generated_at_utc": _now_utc_iso(),
        "summary": {},
    }
    warnings.append("daily_json_created_shell")
    created = True
    return obj, warnings, created


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _extract_bt_summary(*, out_pack_dir: Path, run_dir: Optional[Path], cost_bps: float, status: str, warnings: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": str(status),
        "generated_utc": _now_utc_iso(),
        "run_dir": str(run_dir.resolve()) if run_dir is not None else "",
        "out_dir": str(out_pack_dir.resolve()),
        "date_range": {"start": "", "end": ""},
        "tickers": {"count": 0},
        "cost_bps": float(cost_bps),
        "start_equity": None,
        "end_equity": None,
        "total_return": None,
        "max_drawdown": None,
        "days": 0,
        "trade_rows": 0,
        "rebalance_count": 0,
        "turnover_notional": None,
        "total_cost": None,
        "warnings": list(warnings),
    }
    if status != "OK":
        return summary

    bt_manifest = safe_read_json(out_pack_dir / "backtest" / "backtest_manifest.json") or {}
    from_run_manifest = safe_read_json(out_pack_dir / "backtest_from_run_manifest.json") or {}

    bt_sum = bt_manifest.get("summary") if isinstance(bt_manifest.get("summary"), dict) else {}
    bt_cost = bt_manifest.get("cost_summary") if isinstance(bt_manifest.get("cost_summary"), dict) else {}
    date_range = from_run_manifest.get("date_range") if isinstance(from_run_manifest.get("date_range"), dict) else {}
    tickers = from_run_manifest.get("tickers") if isinstance(from_run_manifest.get("tickers"), list) else []

    summary.update(
        {
            "date_range": {"start": str(date_range.get("start", "")), "end": str(date_range.get("end", ""))},
            "tickers": {"count": len(tickers)},
            "start_equity": _num_or_none(bt_sum.get("start_equity")),
            "end_equity": _num_or_none(bt_sum.get("end_equity")),
            "total_return": _num_or_none(bt_sum.get("total_return")),
            "max_drawdown": _num_or_none(bt_sum.get("max_drawdown")),
            "days": int(_num_or_none(bt_sum.get("days")) or 0),
            "trade_rows": int(_num_or_none(bt_cost.get("trade_rows")) or 0),
            "rebalance_count": int(_num_or_none(bt_cost.get("rebalance_count")) or 0),
            "turnover_notional": _num_or_none(bt_cost.get("total_turnover_notional")),
            "total_cost": _num_or_none(bt_cost.get("total_cost")),
        }
    )
    return summary


def attach_backtest_to_daily(
    *,
    daily_base: Path,
    date_str: str,
    outputs_base: Path,
    price_store: Path,
    out_pack_dir: Path,
    cost_bps: float,
    strict: bool,
    verbose: bool,
) -> Tuple[int, Dict[str, Any]]:
    date_norm = _parse_date(date_str)
    if not date_norm:
        return 2, {"error": f"invalid date: {date_str}"}

    daily_base = daily_base.resolve()
    outputs_base = outputs_base.resolve()
    price_store = price_store.resolve()
    out_pack_dir = out_pack_dir.resolve()
    report_path = (daily_base / f"{date_norm}.json").resolve()

    report_obj, pre_warnings, created_shell = _ensure_daily_json(report_path, date_str=date_norm, strict=bool(strict))
    if report_obj is None:
        return 2, {"error": ";".join(pre_warnings) if pre_warnings else "daily report unavailable"}

    run_dir, run_reason, run_trace = discover_run_dir_for_date(
        date_str=date_norm,
        daily_report_path=report_path,
        base_out_dir=outputs_base,
    )

    warnings: List[str] = list(pre_warnings)
    backtest_status = "OK"
    step_rc = 0
    step_info: Dict[str, Any] = {}

    if run_dir is None:
        backtest_status = "MISSING_RUN"
        warnings.append(f"missing_run:{run_reason}")
        if strict:
            step_rc = 2
        else:
            step_rc = 0
    else:
        step_rc, step_info = run_backtest_from_run(
            run_dir=run_dir,
            price_store_root=price_store,
            out_root=out_pack_dir,
            report_tz="America/New_York",
            date_start="",
            date_end="",
            cost_bps=float(cost_bps),
            initial_equity=100000.0,
            rebalance="daily",
            price_csv="",
            verbose=bool(verbose),
        )
        if step_rc != 0:
            backtest_status = "ERROR"
            warnings.append(str(step_info.get("error") or f"backtest_rc_{step_rc}"))
            if strict:
                step_rc = 2
            else:
                step_rc = 0

    summary = _extract_bt_summary(
        out_pack_dir=out_pack_dir,
        run_dir=run_dir,
        cost_bps=float(cost_bps),
        status=backtest_status,
        warnings=warnings,
    )

    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    qp["backtest_from_run"] = summary
    report_obj["quant_pack"] = qp
    report_obj.setdefault("date", date_norm)
    report_obj["updated_at_utc"] = _now_utc_iso()

    backup = _backup_file(report_path)
    _write_json_atomic(report_path, report_obj)

    attach_manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "daily_base": str(daily_base),
        "date": date_norm,
        "daily_report_path": str(report_path),
        "daily_report_backup": str(backup) if backup else "",
        "created_daily_shell": bool(created_shell),
        "outputs_base": str(outputs_base),
        "price_store": str(price_store),
        "out_pack_dir": str(out_pack_dir),
        "run_discovery": {
            "run_dir": str(run_dir) if run_dir is not None else "",
            "reason": run_reason,
            "trace": run_trace,
        },
        "backtest_status": backtest_status,
        "warnings": warnings,
        "step": step_info,
        "exit_code": int(step_rc),
    }
    out_pack_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_pack_dir / "attach_manifest.json", attach_manifest)

    if verbose:
        print(f"[A17] date={date_norm} status={backtest_status} report={report_path}")
        print(f"[A17] run_dir={(str(run_dir) if run_dir is not None else '-')}")
        print(f"[A17] out_pack_dir={out_pack_dir}")
    return int(step_rc), attach_manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach A4-4 backtest summary into flat Daily Report JSON.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest")
    p.add_argument("--outputs-base", default="outputs")
    p.add_argument("--price-store", default="outputs/price_store")
    p.add_argument("--out-pack-dir", default="", help="default outputs/Daily Report/quant_packs/<date>/backtest_from_run")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    if not daily_base.exists():
        print(f"[ERROR] daily base not found: {daily_base}")
        return 2
    date_norm = _parse_date(args.date) or _discover_latest_date(daily_base)
    if not date_norm:
        print(f"[ERROR] no valid date found under {daily_base}")
        return 2
    out_pack_dir = (
        Path(args.out_pack_dir).resolve()
        if str(args.out_pack_dir or "").strip()
        else (daily_base / "quant_packs" / date_norm / "backtest_from_run").resolve()
    )
    rc, info = attach_backtest_to_daily(
        daily_base=daily_base,
        date_str=date_norm,
        outputs_base=Path(args.outputs_base),
        price_store=Path(args.price_store),
        out_pack_dir=out_pack_dir,
        cost_bps=float(args.cost_bps),
        strict=bool(args.strict),
        verbose=bool(args.verbose),
    )
    if rc != 0:
        print(f"[ERROR] {info.get('error', 'attach failed')}")
    elif args.verbose:
        print("[PASS] a17_attach_backtest_to_daily")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

