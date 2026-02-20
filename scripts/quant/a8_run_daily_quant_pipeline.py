#!/usr/bin/env python3
"""A2-2: Daily Quant Pipeline (build pack + embed + update index)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_daily_embed import discover_quant_md, embed_quant_into_daily_report, write_embed_manifest
from quant_daily_pack import build_daily_pack
from quant_exec_blockers import attach_exec_blockers_to_daily
from quant_replay_drift_daily import attach_replay_drift_to_daily
from a17_attach_backtest_to_daily import attach_backtest_to_daily
from a20_attach_backtest_sweep_to_daily import attach_backtest_sweep_to_daily
from a20_build_quant_alerts import build_quant_alerts
from a18_reconcile_live_vs_backtest import reconcile_live_vs_backtest
from a7_update_daily_reports_index import update_daily_reports_index

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
    from atomic_io import safe_read_json as io_safe_read_json
except Exception:
    io_atomic_write_json = None
    io_safe_read_json = None


DATE_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if io_safe_read_json is not None:
        return io_safe_read_json(str(path), retries=2, sleep_ms=15)
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if io_atomic_write_json is not None:
        io_atomic_write_json(str(path), obj, indent=2)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


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
        m = DATE_FILE_RE.match(p.name)
        if not m:
            continue
        candidates.append(m.group(1))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _derive_base_out_dir(daily_base: Path) -> Path:
    # typical: outputs/Daily Report -> outputs
    return daily_base.parent.resolve()


def _annotate_missing_report_in_index(
    *,
    daily_base: Path,
    date_str: str,
    report_path: Path,
    verbose: bool,
) -> Dict[str, Any]:
    index_path = (daily_base / "daily_reports_index.json").resolve()
    obj = _read_json(index_path) or {}
    reports = obj.get("reports") if isinstance(obj.get("reports"), list) else []
    normalized: List[Dict[str, Any]] = []
    found = False
    for row in reports:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if str(item.get("date", "")).strip() == date_str:
            item["path"] = str(report_path)
            item["missing_report"] = True
            quant = item.get("quant") if isinstance(item.get("quant"), dict) else {}
            quant["source"] = str(quant.get("source") or "missing_report")
            quant["updated_at_utc"] = _now_utc_iso()
            item["quant"] = quant
            warns = item.get("warnings") if isinstance(item.get("warnings"), list) else []
            if "missing_report" not in warns:
                warns.append("missing_report")
            item["warnings"] = warns
            found = True
        normalized.append(item)
    if not found:
        normalized.append(
            {
                "date": date_str,
                "path": str(report_path),
                "missing_report": True,
                "warnings": ["missing_report"],
                "quant": {
                    "total_return": None,
                    "sharpe": None,
                    "max_drawdown": None,
                    "trades_total": None,
                    "gate_status": "NA",
                    "pack_path": "",
                    "updated_at_utc": _now_utc_iso(),
                    "source": "missing_report",
                },
            }
        )
    normalized.sort(key=lambda x: str(x.get("date", "")), reverse=True)
    obj["reports"] = normalized
    obj["updated_at"] = datetime.now().astimezone().isoformat()
    bak = _backup_file(index_path)
    _write_json_atomic(index_path, obj)
    if verbose:
        print(f"[A8] index annotate missing_report: date={date_str} bak={bak}")
    return {"index_path": str(index_path), "backup_path": str(bak) if bak else "", "annotated": True}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run daily quant pipeline: build pack + embed + update index.")
    p.add_argument("--daily-base", default="outputs/Daily Report")
    p.add_argument("--date", default="", help="YYYY-MM-DD; default latest date file under daily-base")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--auto-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-leaderboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-replay-drift", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-backtest-from-run", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-backtest-sweep", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-reconcile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-exec-blockers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-alerts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--replay-drift-strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--replay-drift-fail-on-drift", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--replay-drift-rules-json", default="")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    daily_base = Path(args.daily_base).resolve()
    started = _now_utc_iso()

    if not daily_base.exists():
        print(f"[A8] ERROR: daily base not found: {daily_base}")
        return 2

    date_str = _parse_date(args.date)
    if not date_str:
        date_str = _discover_latest_date(daily_base)
    if not date_str:
        print(f"[A8] ERROR: no daily report date files under {daily_base}")
        return 2

    daily_report_path = (daily_base / f"{date_str}.json").resolve()
    pack_dir = (daily_base / "quant_packs" / date_str).resolve()
    pipeline_manifest_path = (pack_dir / "pipeline_manifest.json").resolve()
    base_out_dir = _derive_base_out_dir(daily_base)

    warnings: List[str] = []
    steps_ok: Dict[str, bool] = {
        "build_pack": False,
        "embed": False,
        "backtest_from_run": True,
        "backtest_sweep": True,
        "reconcile": True,
        "exec_blockers": True,
        "replay_drift": True,
        "update_index": False,
        "alerts": True,
    }
    step_results: Dict[str, Any] = {}

    print(f"[A8] date={date_str} daily_base={daily_base}")

    # Step 1: build pack
    build_rc, build_manifest = build_daily_pack(
        daily_dir_arg="",
        daily_base=daily_base,
        date_str=date_str,
        dataset_dir_arg="",
        baseline_dataset_arg="",
        baseline_mode="prev_day",
        base_dir=daily_base,
        lookback_days=int(args.lookback_days),
        auto_extract=True,
        base_out_dir=base_out_dir,
        auto_metrics=bool(args.auto_metrics),
        auto_gate=bool(args.auto_gate),
        auto_leaderboard=bool(args.auto_leaderboard),
        out_dir=pack_dir,
        strict=bool(args.strict),
        report_tz="America/New_York",
        annualization=252,
        rf=0.0,
        min_points=5,
        verbose=bool(args.verbose),
    )
    steps_ok["build_pack"] = int(build_rc) == 0
    step_results["build_pack"] = {
        "rc": int(build_rc),
        "pack_dir": str(pack_dir),
        "manifest_path": str((pack_dir / "pack_manifest.json").resolve()),
        "status": str((build_manifest or {}).get("status", "")),
        "gate_status": str((build_manifest or {}).get("gate_status", "")),
    }
    print(f"[A8] build_pack: {'ok' if steps_ok['build_pack'] else 'fail'} path={pack_dir}")
    if int(build_rc) != 0:
        warnings.append(f"build_pack_rc={build_rc}")

    # Step 2: embed
    report_exists = daily_report_path.exists()
    embed_backup = ""
    if report_exists:
        quant_md = discover_quant_md(
            daily_dir=None,
            daily_base=daily_base,
            date_str=date_str,
            quant_md_arg="",
        )
        embed_result = embed_quant_into_daily_report(
            daily_dir=None,
            daily_base=daily_base,
            date_str=date_str,
            quant_md=quant_md,
            report_file=daily_report_path,
            mode="replace",
            out_file=None,
            strict=bool(args.strict),
        )
        write_embed_manifest(
            daily_dir=None,
            daily_base=daily_base,
            date_str=date_str,
            result=embed_result,
        )
        steps_ok["embed"] = int(embed_result.exit_code) == 0
        embed_backup_path = daily_report_path.with_name(daily_report_path.name + ".bak")
        if embed_backup_path.exists():
            embed_backup = str(embed_backup_path)
        step_results["embed"] = {
            "rc": int(embed_result.exit_code),
            "report": str(embed_result.daily_report_out) if embed_result.daily_report_out else str(daily_report_path),
            "backup": embed_backup,
            "warnings": list(embed_result.warnings),
        }
        print(f"[A8] embed: {'ok' if steps_ok['embed'] else 'fail'} report={daily_report_path} backup={embed_backup or '-'}")
        if int(embed_result.exit_code) != 0:
            warnings.append(f"embed_rc={embed_result.exit_code}")
    else:
        steps_ok["embed"] = False
        step_results["embed"] = {"rc": 0, "status": "skipped", "reason": "missing_daily_report"}
        warnings.append("missing_daily_report")
        print(f"[A8] embed: skipped report_missing={daily_report_path}")

    # Step 3: backtest-from-run attach (optional)
    if bool(args.auto_exec_blockers):
        exec_rc, exec_manifest = attach_exec_blockers_to_daily(
            daily_base=daily_base,
            date_str=date_str,
            strict=bool(args.strict),
            auto_compute=True,
            verbose=bool(args.verbose),
        )
        steps_ok["exec_blockers"] = int(exec_rc) in (0, 1)
        step_results["exec_blockers"] = {
            "rc": int(exec_rc),
            "status": str(exec_manifest.get("exec_blockers_status", "")),
            "manifest_path": str((daily_base / "quant_packs" / date_str / "exec_blockers_attach_manifest.json").resolve()),
            "warnings": list(exec_manifest.get("warnings", []))
            if isinstance(exec_manifest.get("warnings"), list)
            else [],
        }
        print(
            f"[A8] exec_blockers: {'ok' if steps_ok['exec_blockers'] else 'fail'} "
            f"status={exec_manifest.get('exec_blockers_status')} rc={exec_rc}"
        )
        if int(exec_rc) != 0:
            warnings.append(f"exec_blockers_rc={exec_rc}")
    else:
        step_results["exec_blockers"] = {"rc": 0, "status": "skipped", "reason": "auto_exec_blockers_disabled"}

    # Step 3: backtest-from-run attach (optional)
    backtest_attach_rc = 0
    if bool(args.auto_backtest_from_run):
        bt_out_dir = (daily_base / "quant_packs" / date_str / "backtest_from_run").resolve()
        backtest_attach_rc, backtest_attach_manifest = attach_backtest_to_daily(
            daily_base=daily_base,
            date_str=date_str,
            outputs_base=base_out_dir.resolve(),
            price_store=(base_out_dir / "price_store").resolve(),
            out_pack_dir=bt_out_dir,
            cost_bps=5.0,
            strict=bool(args.strict),
            verbose=bool(args.verbose),
        )
        steps_ok["backtest_from_run"] = int(backtest_attach_rc) in (0, 1)
        step_results["backtest_from_run"] = {
            "rc": int(backtest_attach_rc),
            "status": str(backtest_attach_manifest.get("backtest_status", "")),
            "manifest_path": str((bt_out_dir / "attach_manifest.json").resolve()),
            "warnings": list(backtest_attach_manifest.get("warnings", []))
            if isinstance(backtest_attach_manifest.get("warnings"), list)
            else [],
        }
        print(
            f"[A8] backtest_from_run: {'ok' if steps_ok['backtest_from_run'] else 'fail'} "
            f"status={backtest_attach_manifest.get('backtest_status')} rc={backtest_attach_rc}"
        )
        if int(backtest_attach_rc) != 0:
            warnings.append(f"backtest_from_run_rc={backtest_attach_rc}")
    else:
        step_results["backtest_from_run"] = {"rc": 0, "status": "skipped", "reason": "auto_backtest_from_run_disabled"}

    # Step 3.5: backtest sweep attach (optional)
    if bool(args.auto_backtest_sweep):
        sweep_out_dir = (daily_base / "quant_packs" / date_str / "backtest_sweep").resolve()
        sweep_rc, sweep_manifest = attach_backtest_sweep_to_daily(
            daily_base=daily_base,
            date_str=date_str,
            cost_bps_list=[0.0, 5.0, 10.0, 20.0],
            price_store="",
            run_dir="",
            out_dir=str(sweep_out_dir),
            embed=True,
            strict=bool(args.strict),
            verbose=bool(args.verbose),
            outputs_base=str(base_out_dir.resolve()),
        )
        steps_ok["backtest_sweep"] = int(sweep_rc) in (0, 1)
        step_results["backtest_sweep"] = {
            "rc": int(sweep_rc),
            "status": str(sweep_manifest.get("status", "")),
            "manifest_path": str((sweep_out_dir / "attach_manifest.json").resolve()),
            "warnings": list(sweep_manifest.get("warnings", []))
            if isinstance(sweep_manifest.get("warnings"), list)
            else [],
        }
        print(
            f"[A8] backtest_sweep: {'ok' if steps_ok['backtest_sweep'] else 'fail'} "
            f"status={sweep_manifest.get('status')} rc={sweep_rc}"
        )
        if int(sweep_rc) != 0:
            warnings.append(f"backtest_sweep_rc={sweep_rc}")
    else:
        step_results["backtest_sweep"] = {"rc": 0, "status": "skipped", "reason": "auto_backtest_sweep_disabled"}

    # Step 4: reconcile live vs backtest (optional)
    reconcile_rc = 0
    if bool(args.auto_reconcile):
        reconcile_rc, reconcile_manifest = reconcile_live_vs_backtest(
            daily_base=daily_base,
            date_str=date_str,
            strict=bool(args.strict),
            verbose=bool(args.verbose),
        )
        steps_ok["reconcile"] = int(reconcile_rc) in (0, 1)
        step_results["reconcile"] = {
            "rc": int(reconcile_rc),
            "status": str(reconcile_manifest.get("status", "")),
            "manifest_path": str((daily_base / "quant_packs" / date_str / "reconcile" / "reconcile_manifest.json").resolve()),
        }
        print(
            f"[A8] reconcile: {'ok' if steps_ok['reconcile'] else 'fail'} "
            f"status={reconcile_manifest.get('status')} rc={reconcile_rc}"
        )
        if int(reconcile_rc) != 0:
            warnings.append(f"reconcile_rc={reconcile_rc}")
    else:
        step_results["reconcile"] = {"rc": 0, "status": "skipped", "reason": "auto_reconcile_disabled"}

    # Step 5: replay drift attach (optional)
    replay_rc = 0
    if bool(args.auto_replay_drift):
        rules_obj = None
        rules_path = str(args.replay_drift_rules_json or "").strip()
        if rules_path:
            try:
                rules_candidate = _read_json(Path(rules_path).resolve())
                if isinstance(rules_candidate, dict):
                    rules_obj = rules_candidate
                else:
                    warnings.append(f"replay_drift_rules_invalid:{rules_path}")
            except Exception as exc:
                warnings.append(f"replay_drift_rules_error:{exc}")

        replay_rc, replay_manifest = attach_replay_drift_to_daily(
            daily_base=daily_base,
            date_str=date_str,
            strict=bool(args.replay_drift_strict),
            fail_on_drift=bool(args.replay_drift_fail_on_drift),
            rules=rules_obj,
            out_base=(daily_base / "quant_packs" / date_str / "replay_drift").resolve(),
            verbose=bool(args.verbose),
        )
        steps_ok["replay_drift"] = int(replay_rc) in (0, 1)
        step_results["replay_drift"] = {
            "rc": int(replay_rc),
            "status": str(replay_manifest.get("status", "")),
            "manifest_path": str((Path(replay_manifest.get("drift_gate_out_dir")) / "replay_drift_manifest.json").resolve())
            if replay_manifest.get("drift_gate_out_dir")
            else "",
            "warnings": list(replay_manifest.get("warnings", [])) if isinstance(replay_manifest.get("warnings"), list) else [],
        }
        print(
            f"[A8] replay_drift: {'ok' if steps_ok['replay_drift'] else 'fail'} "
            f"status={replay_manifest.get('status')} rc={replay_rc}"
        )
        if int(replay_rc) != 0:
            warnings.append(f"replay_drift_rc={replay_rc}")
    else:
        step_results["replay_drift"] = {"rc": 0, "status": "skipped", "reason": "auto_replay_drift_disabled"}

    # Step 6: update index
    index_result: Optional[Dict[str, Any]] = None
    try:
        index_result = update_daily_reports_index(
            daily_base,
            lookback_days=max(int(args.lookback_days), 1),
            verbose=False,
        )
        if not report_exists:
            _annotate_missing_report_in_index(
                daily_base=daily_base,
                date_str=date_str,
                report_path=daily_report_path,
                verbose=bool(args.verbose),
            )
        steps_ok["update_index"] = True
        step_results["update_index"] = index_result
        print(f"[A8] update_index: ok path={index_result.get('index_path') if isinstance(index_result, dict) else (daily_base / 'daily_reports_index.json')}")
    except Exception as exc:
        steps_ok["update_index"] = False
        step_results["update_index"] = {"error": str(exc)}
        warnings.append(f"update_index_error:{exc}")
        print(f"[A8] update_index: fail error={exc}")

    # Step 7: alerts (optional)
    if bool(args.auto_alerts):
        try:
            alerts_result = build_quant_alerts(
                daily_base=daily_base,
                lookback_days=max(int(args.lookback_days), 1),
                verbose=bool(args.verbose),
            )
            steps_ok["alerts"] = True
            step_results["alerts"] = alerts_result
            print(
                f"[A8] alerts: ok path={alerts_result.get('alerts_json') if isinstance(alerts_result, dict) else ''}"
            )
        except Exception as exc:
            steps_ok["alerts"] = False
            step_results["alerts"] = {"error": str(exc)}
            warnings.append(f"alerts_error:{exc}")
            print(f"[A8] alerts: fail error={exc}")
    else:
        step_results["alerts"] = {"rc": 0, "status": "skipped", "reason": "auto_alerts_disabled"}

    finished = _now_utc_iso()
    pipeline_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "date": date_str,
        "daily_base": str(daily_base),
        "daily_report_path": str(daily_report_path),
        "pack_dir": str(pack_dir),
        "index_path": str((daily_base / "daily_reports_index.json").resolve()),
        "steps_ok": steps_ok,
        "step_results": step_results,
        "warnings": warnings,
        "strict": bool(args.strict),
        "options": {
            "lookback_days": int(args.lookback_days),
            "auto_metrics": bool(args.auto_metrics),
            "auto_gate": bool(args.auto_gate),
            "auto_leaderboard": bool(args.auto_leaderboard),
            "auto_replay_drift": bool(args.auto_replay_drift),
            "auto_backtest_from_run": bool(args.auto_backtest_from_run),
            "auto_backtest_sweep": bool(args.auto_backtest_sweep),
            "auto_reconcile": bool(args.auto_reconcile),
            "auto_exec_blockers": bool(args.auto_exec_blockers),
            "auto_alerts": bool(args.auto_alerts),
            "replay_drift_strict": bool(args.replay_drift_strict),
            "replay_drift_fail_on_drift": bool(args.replay_drift_fail_on_drift),
            "replay_drift_rules_json": str(args.replay_drift_rules_json or ""),
        },
    }
    _write_json_atomic(pipeline_manifest_path, pipeline_manifest)

    final_rc = 0
    if bool(args.strict) and not report_exists:
        final_rc = 2
    elif int(build_rc) != 0:
        final_rc = 2 if bool(args.strict) else 1
    elif not steps_ok["update_index"]:
        final_rc = 2
    elif bool(args.auto_reconcile) and not steps_ok["reconcile"]:
        final_rc = 2 if bool(args.strict) else 1
    elif bool(args.auto_backtest_from_run) and not steps_ok["backtest_from_run"]:
        final_rc = 2 if bool(args.strict) else 1
    elif bool(args.auto_backtest_sweep) and not steps_ok["backtest_sweep"]:
        final_rc = 2 if bool(args.strict) else 1
    elif bool(args.auto_exec_blockers) and not steps_ok["exec_blockers"]:
        final_rc = 2 if bool(args.strict) else 1
    elif bool(args.auto_alerts) and not steps_ok["alerts"]:
        final_rc = 2 if bool(args.strict) else 1
    elif not report_exists or (not steps_ok["embed"]):
        final_rc = 2 if bool(args.strict) else 1
    elif warnings:
        final_rc = 1

    if int(replay_rc) == 3:
        final_rc = 3
    elif int(replay_rc) == 2 and final_rc < 2:
        final_rc = 2
    elif int(replay_rc) == 1 and final_rc == 0:
        final_rc = 1

    if args.verbose:
        print(f"[A8] pipeline_manifest={pipeline_manifest_path}")
        if warnings:
            print(f"[A8] warnings={warnings}")

    return int(final_rc)


if __name__ == "__main__":
    raise SystemExit(main())
