#!/usr/bin/env python3
"""A4-12: attach backtest sweep summary into daily report + quant pack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quant_backtest_sweep import parse_list_csv, run_sweep, write_outputs
from quant_daily_pack import discover_run_dir_for_date
from quant_io_utils import safe_read_json
from weights_from_run import build_daily_weights, write_weights


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
    days: List[str] = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        d = _parse_date(p.stem)
        if d:
            days.append(d)
    if not days:
        return None
    days.sort()
    return days[-1]


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


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def _row_for_cost(rows: List[Dict[str, Any]], cost_bps: float) -> Optional[Dict[str, Any]]:
    for row in rows:
        cb = _num_or_none(row.get("cost_bps"))
        if cb is None:
            continue
        if abs(float(cb) - float(cost_bps)) <= 1e-9:
            return row
    return None


def _compute_break_even_cost_bps(rows: List[Dict[str, Any]]) -> Optional[float]:
    pts: List[Tuple[float, float]] = []
    for row in rows:
        cb = _num_or_none(row.get("cost_bps"))
        rt = _num_or_none(row.get("total_return"))
        if cb is None or rt is None:
            continue
        pts.append((float(cb), float(rt)))
    pts.sort(key=lambda x: x[0])
    if not pts:
        return None
    for i, (b0, r0) in enumerate(pts):
        if abs(r0) <= 1e-12:
            return float(b0)
        if i + 1 >= len(pts):
            continue
        b1, r1 = pts[i + 1]
        if r0 > 0.0 and r1 < 0.0:
            if abs(r1 - r0) <= 1e-12:
                return float((b0 + b1) / 2.0)
            frac = (0.0 - r0) / (r1 - r0)
            return float(b0 + frac * (b1 - b0))
    return None


def _build_summary(
    *,
    status: str,
    rows: List[Dict[str, Any]],
    cost_bps_list: List[float],
    warnings: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": str(status),
        "generated_utc": _now_utc_iso(),
        "cost_bps_list": [float(x) for x in cost_bps_list],
        "best_cost_bps": None,
        "return_at_0bps": None,
        "return_at_5bps": None,
        "return_at_10bps": None,
        "return_at_20bps": None,
        "sensitivity_per_1bp": None,
        "break_even_cost_bps": None,
        "warnings": list(warnings),
    }
    if str(status).upper() != "OK" or not rows:
        return summary

    best_row = max(
        rows,
        key=lambda r: (
            float(_num_or_none(r.get("end_equity")) or 0.0),
            -float(_num_or_none(r.get("cost_bps")) or 0.0),
        ),
    )
    summary["best_cost_bps"] = _num_or_none(best_row.get("cost_bps"))

    for fixed in (0.0, 5.0, 10.0, 20.0):
        row = _row_for_cost(rows, fixed)
        key = f"return_at_{int(fixed)}bps"
        summary[key] = _num_or_none(row.get("total_return")) if row is not None else None

    ret0 = _num_or_none(summary.get("return_at_0bps"))
    ret20 = _num_or_none(summary.get("return_at_20bps"))
    if ret0 is not None and ret20 is not None:
        summary["sensitivity_per_1bp"] = float((ret20 - ret0) / 20.0)
    else:
        pts = []
        for row in rows:
            cb = _num_or_none(row.get("cost_bps"))
            rt = _num_or_none(row.get("total_return"))
            if cb is None or rt is None:
                continue
            pts.append((float(cb), float(rt)))
        pts.sort(key=lambda x: x[0])
        if len(pts) >= 2 and abs(pts[-1][0] - pts[0][0]) > 1e-9:
            summary["sensitivity_per_1bp"] = float((pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]))

    summary["break_even_cost_bps"] = _compute_break_even_cost_bps(rows)
    return summary


def _resolve_run_dir(
    *,
    explicit_run_dir: str,
    report_obj: Dict[str, Any],
    date_str: str,
    report_path: Path,
    outputs_base: Path,
) -> Tuple[Optional[Path], str]:
    if str(explicit_run_dir or "").strip():
        run_path = Path(str(explicit_run_dir)).resolve()
        if run_path.exists() and run_path.is_dir():
            return run_path, "arg_run_dir"
        return None, "arg_run_dir_missing"
    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    bt = qp.get("backtest_from_run") if isinstance(qp.get("backtest_from_run"), dict) else {}
    run_val = str(bt.get("run_dir", "") or "").strip()
    if run_val:
        run_path = Path(run_val)
        if not run_path.is_absolute():
            run_path = (outputs_base / run_path).resolve()
        if run_path.exists() and run_path.is_dir():
            return run_path, "report_backtest_from_run"
    run_discovered, reason, _ = discover_run_dir_for_date(
        date_str=str(date_str),
        daily_report_path=report_path,
        base_out_dir=outputs_base,
    )
    if run_discovered is not None:
        return run_discovered.resolve(), f"discover:{reason}"
    return None, f"missing:{reason}"


def _resolve_price_store(
    *,
    explicit_price_store: str,
    report_obj: Dict[str, Any],
    outputs_base: Path,
) -> Tuple[Optional[Path], str]:
    if str(explicit_price_store or "").strip():
        p = Path(str(explicit_price_store)).resolve()
        if p.exists():
            return p, "arg_price_store"
        return None, "arg_price_store_missing"
    qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    bt = qp.get("backtest_from_run") if isinstance(qp.get("backtest_from_run"), dict) else {}
    for key in ("price_store_path", "price_store", "cache_dir"):
        v = str(bt.get(key, "") or "").strip()
        if not v:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = (outputs_base / p).resolve()
        if p.exists():
            return p, f"report_backtest_from_run_{key}"
    candidate = (outputs_base / "price_store").resolve()
    if candidate.exists():
        return candidate, "outputs_base_price_store"
    return None, "missing_price_store"


def attach_backtest_sweep_to_daily(
    *,
    daily_base: Path,
    date_str: str,
    cost_bps_list: List[float],
    price_store: str = "",
    run_dir: str = "",
    out_dir: str = "",
    embed: bool = True,
    strict: bool = False,
    verbose: bool = False,
    outputs_base: str = "",
) -> Tuple[int, Dict[str, Any]]:
    daily_base = Path(daily_base).resolve()
    date_norm = _parse_date(date_str)
    if not date_norm:
        return 2, {"error": f"invalid date: {date_str}"}
    report_path = (daily_base / f"{date_norm}.json").resolve()
    report_obj = safe_read_json(report_path) or {}
    if not report_obj and strict:
        return 2, {"error": f"missing_or_invalid_daily_report:{report_path}"}
    if not report_obj:
        report_obj = {"date": date_norm, "schema_version": 1}

    outputs_root = Path(outputs_base).resolve() if str(outputs_base or "").strip() else daily_base.parent.resolve()
    out_pack_dir = (
        Path(out_dir).resolve()
        if str(out_dir or "").strip()
        else (daily_base / "quant_packs" / date_norm / "backtest_sweep").resolve()
    )
    warnings: List[str] = []

    resolved_run_dir, run_reason = _resolve_run_dir(
        explicit_run_dir=str(run_dir or ""),
        report_obj=report_obj,
        date_str=date_norm,
        report_path=report_path,
        outputs_base=outputs_root,
    )
    if resolved_run_dir is None:
        warnings.append(f"run_dir:{run_reason}")
    resolved_price_store, ps_reason = _resolve_price_store(
        explicit_price_store=str(price_store or ""),
        report_obj=report_obj,
        outputs_base=outputs_root,
    )
    if resolved_price_store is None:
        warnings.append(f"price_store:{ps_reason}")

    status = "OK"
    rows: List[Dict[str, Any]] = []
    sweep_manifest: Dict[str, Any] = {}
    sweep_paths: Dict[str, str] = {}

    if resolved_run_dir is None or resolved_price_store is None:
        status = "MISSING"
    else:
        try:
            weights_rows, weights_manifest = build_daily_weights(
                resolved_run_dir,
                report_tz="America/New_York",
                date_start="",
                date_end="",
            )
            weights_info = write_weights((out_pack_dir / "weights").resolve(), weights_rows, weights_manifest)
            weights_csv = Path(weights_info["weights_csv"]).resolve()
            rows, sweep_manifest = run_sweep(
                weights_csv=weights_csv,
                price_store_dir=resolved_price_store,
                start="",
                end="",
                cost_bps_list=cost_bps_list,
                out_dir=out_pack_dir,
                initial_equity=100000.0,
                rebalance_rule="daily",
            )
            sweep_paths = write_outputs(out_pack_dir, rows, sweep_manifest)
        except Exception as exc:
            status = "ERROR"
            warnings.append(f"sweep_error:{exc}")

    summary = _build_summary(
        status=status,
        rows=rows,
        cost_bps_list=cost_bps_list,
        warnings=warnings,
    )
    if sweep_paths:
        summary["out_dir"] = str(out_pack_dir)
        summary["report_md"] = str(sweep_paths.get("report_md", ""))
        summary["request_hash"] = str((sweep_manifest.get("request_hash") if isinstance(sweep_manifest, dict) else "") or "")

    backup = None
    if embed:
        qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
        qp["backtest_sweep"] = summary
        report_obj["quant_pack"] = qp
        report_obj["updated_at_utc"] = _now_utc_iso()
        backup = _backup_file(report_path) if report_path.exists() else None
        _write_json_atomic(report_path, report_obj)

    attach_manifest = {
        "schema_version": 1,
        "generated_utc": _now_utc_iso(),
        "daily_base": str(daily_base),
        "date": date_norm,
        "daily_report_path": str(report_path),
        "daily_report_backup": str(backup) if backup else "",
        "resolved_run_dir": str(resolved_run_dir) if resolved_run_dir else "",
        "resolved_price_store": str(resolved_price_store) if resolved_price_store else "",
        "run_reason": run_reason,
        "price_store_reason": ps_reason,
        "status": status,
        "warnings": warnings,
        "sweep_out_dir": str(out_pack_dir),
        "sweep_paths": sweep_paths,
        "summary": summary,
    }
    out_pack_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic((out_pack_dir / "attach_manifest.json").resolve(), attach_manifest)

    rc = 0
    if strict and status in ("MISSING", "ERROR"):
        rc = 2
    elif status == "ERROR":
        rc = 2

    if verbose:
        print(f"[A20-ATTACH] date={date_norm} status={status} report={report_path}")
        print(f"[A20-ATTACH] run_dir={(str(resolved_run_dir) if resolved_run_dir else '-')}")
        print(f"[A20-ATTACH] price_store={(str(resolved_price_store) if resolved_price_store else '-')}")
        print(f"[A20-ATTACH] out_dir={out_pack_dir}")
    return rc, attach_manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach backtest sweep summary to flat daily report JSON.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest")
    p.add_argument("--cost-bps-list", default="0,5,10,20")
    p.add_argument("--price-store", default="")
    p.add_argument("--run-dir", default="")
    p.add_argument("--outputs-base", default="", help="Optional outputs root for run_dir discovery")
    p.add_argument("--out-dir", default="")
    p.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
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
    try:
        cost_bps_list = parse_list_csv(str(args.cost_bps_list))
    except Exception as exc:
        print(f"[ERROR] invalid --cost-bps-list: {exc}")
        return 2
    rc, info = attach_backtest_sweep_to_daily(
        daily_base=daily_base,
        date_str=date_norm,
        cost_bps_list=cost_bps_list,
        price_store=str(args.price_store or ""),
        run_dir=str(args.run_dir or ""),
        out_dir=str(args.out_dir or ""),
        embed=bool(args.embed),
        strict=bool(args.strict),
        verbose=bool(args.verbose),
        outputs_base=str(args.outputs_base or ""),
    )
    if rc != 0:
        print(f"[ERROR] {info.get('error', info.get('status', 'attach_failed'))}")
    elif args.verbose:
        print("[PASS] a20_attach_backtest_sweep_to_daily")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

