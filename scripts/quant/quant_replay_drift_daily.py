#!/usr/bin/env python3
"""A3-4 helpers: attach replay drift gate summary to flat daily report json."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

from quant_io_utils import safe_read_json
from quant_replay_drift import DEFAULT_RULES, run_drift_gate

try:
    from atomic_io import atomic_write_json as io_atomic_write_json
except Exception:
    io_atomic_write_json = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except Exception:
        return ""


def _latest_daily_date(daily_base: Path) -> str:
    dates: List[str] = []
    for p in daily_base.glob("*.json"):
        if p.name == "daily_reports_index.json":
            continue
        d = _parse_date(p.stem)
        if d:
            dates.append(d)
    if not dates:
        return ""
    dates.sort()
    return dates[-1]


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


def _candidate_replay_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.exists() or not root.is_dir():
        return out
    for p in root.rglob("replay_window_summary.csv"):
        if p.is_file():
            out.append(p.parent.resolve())
    for p in root.rglob("replay_window_manifest.json"):
        if p.is_file():
            out.append(p.parent.resolve())
    dedup: List[Path] = []
    seen = set()
    for p in out:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup


def _candidate_mtime(path: Path) -> float:
    ts = 0.0
    for p in (
        (path / "replay_window_manifest.json"),
        (path / "replay_window_summary.csv"),
    ):
        if p.exists():
            try:
                ts = max(ts, p.stat().st_mtime)
            except Exception:
                continue
    if ts <= 0.0:
        try:
            ts = path.stat().st_mtime
        except Exception:
            ts = 0.0
    return ts


def discover_replay_window_dir(
    daily_base: Path,
    date_str: str,
    *,
    extra_search_dirs: Optional[Sequence[Path]] = None,
    verbose: bool = False,
) -> Optional[Path]:
    date_norm = _parse_date(date_str)
    if not date_norm:
        return None

    candidates: List[Path] = []

    # 1) pack-local replay windows
    pack_root = (daily_base / "quant_packs" / date_norm / "replay_window").resolve()
    candidates.extend(_candidate_replay_dirs(pack_root))

    # 2) from daily report pack path
    report_path = (daily_base / f"{date_norm}.json").resolve()
    report_obj = safe_read_json(report_path)
    if isinstance(report_obj, dict):
        qp = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
        pack_path = str(qp.get("pack_md_path") or "").strip()
        if pack_path:
            p = Path(pack_path)
            if not p.is_absolute():
                p = (daily_base / p).resolve()
            maybe_pack = p.parent
            if maybe_pack.exists():
                candidates.extend(_candidate_replay_dirs(maybe_pack / "replay_window"))

        run_id = str(report_obj.get("run_id") or "").strip()
        if run_id:
            base_out = daily_base.parent.resolve()
            run_path = (base_out / date_norm[:7] / run_id).resolve()
            candidates.extend(_candidate_replay_dirs(run_path / "replay_window"))

    # 3) explicit extra search dirs
    for root in list(extra_search_dirs or []):
        candidates.extend(_candidate_replay_dirs(Path(root).resolve()))

    # 4) fallback broad search under outputs (bounded by replay_window folders only)
    if not candidates:
        base_out = daily_base.parent.resolve()
        candidates.extend(_candidate_replay_dirs(base_out / date_norm[:7]))
        if not candidates:
            candidates.extend(_candidate_replay_dirs(base_out))

    if not candidates:
        return None

    candidates.sort(key=lambda p: (_candidate_mtime(p), str(p).lower()), reverse=True)
    chosen = candidates[0]
    if verbose:
        print(f"[A12] replay candidates={len(candidates)} chosen={chosen}")
    return chosen


def _load_rules(rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_RULES)
    if isinstance(rules, dict):
        merged.update(rules)
    return merged


def _status_to_rc(status: str) -> int:
    s = str(status or "").upper()
    if s == "PASS":
        return 0
    if s == "WARN":
        return 1
    if s == "FAIL":
        return 3
    return 2


def run_or_load_drift_gate(
    replay_window_dir: Path,
    out_dir: Path,
    *,
    strict: bool,
    fail_on_drift: bool,
    rules: Optional[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any], bool]:
    out_dir = out_dir.resolve()
    result_path = (out_dir / "drift_gate_result.json").resolve()
    replay_manifest = (replay_window_dir / "replay_window_manifest.json").resolve()
    replay_summary = (replay_window_dir / "replay_window_summary.csv").resolve()

    src_ts = 0.0
    for p in (replay_manifest, replay_summary):
        if p.exists():
            try:
                src_ts = max(src_ts, p.stat().st_mtime)
            except Exception:
                continue

    if result_path.exists():
        cached = safe_read_json(result_path)
        try:
            cached_ts = result_path.stat().st_mtime
        except Exception:
            cached_ts = 0.0
        if isinstance(cached, dict) and cached_ts >= src_ts:
            rc = _status_to_rc(str(cached.get("status") or "PASS"))
            if verbose:
                print(f"[A12] reuse drift gate result: {result_path}")
            return rc, cached, True

    merged_rules = _load_rules(rules)
    rc, result = run_drift_gate(
        replay_window_dir=replay_window_dir.resolve(),
        out_dir=out_dir,
        strict=bool(strict),
        fail_on_drift=bool(fail_on_drift),
        rules=merged_rules,
    )
    return int(rc), result, False


def extract_drift_summary(result_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result_obj, dict):
        return {
            "status": "MISSING",
            "strict": False,
            "fail_on_drift": False,
            "cycles": 0,
            "fails": 0,
            "warns": 0,
            "worst_cycle": {},
            "tag_top": [],
            "generated_utc": _now_utc_iso(),
            "warnings": ["drift_result_missing"],
        }

    window = result_obj.get("window") if isinstance(result_obj.get("window"), dict) else {}
    worst_w = (window.get("worst_by_weights_l1") if isinstance(window.get("worst_by_weights_l1"), list) else [])
    worst_t = (window.get("worst_by_trade_delta") if isinstance(window.get("worst_by_trade_delta"), list) else [])
    worst_cycle: Dict[str, Any] = {}
    if worst_w:
        first = worst_w[0] if isinstance(worst_w[0], dict) else {}
        worst_cycle = {
            "cycle": int(float(first.get("cycle", 0) or 0)),
            "weights_l1": float(first.get("weights_l1", 0.0) or 0.0),
            "tags": list(first.get("tags") or []),
        }
    if worst_t:
        first_t = worst_t[0] if isinstance(worst_t[0], dict) else {}
        if not worst_cycle:
            worst_cycle = {"cycle": int(float(first_t.get("cycle", 0) or 0))}
        if first_t.get("trades_notional_delta") is not None:
            worst_cycle["trades_notional_delta"] = float(first_t.get("trades_notional_delta", 0.0) or 0.0)
        if first_t.get("trades_notional_delta_ratio") is not None:
            worst_cycle["trades_notional_delta_ratio"] = float(first_t.get("trades_notional_delta_ratio", 0.0) or 0.0)

    tag_counts = result_obj.get("tag_counts") if isinstance(result_obj.get("tag_counts"), dict) else {}
    tag_top = [{"tag": str(k), "count": int(v)} for k, v in sorted(tag_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:3]]
    return {
        "status": str(result_obj.get("status") or "MISSING").upper(),
        "strict": bool(result_obj.get("strict", False)),
        "fail_on_drift": bool(result_obj.get("fail_on_drift", False)),
        "cycles": int(float(window.get("cycles", 0) or 0)),
        "fails": int(float(window.get("fails", 0) or 0)),
        "warns": int(float(window.get("warns", 0) or 0)),
        "fail_cycle_ratio": float(window.get("fail_cycle_ratio", 0.0) or 0.0),
        "worst_cycle": worst_cycle,
        "tag_top": tag_top,
        "generated_utc": str(result_obj.get("generated_utc") or _now_utc_iso()),
    }


def write_daily_json_quant_replay_drift(
    daily_report_path: Path,
    summary: Dict[str, Any],
    *,
    backup: bool = True,
    atomic: bool = True,
) -> Dict[str, Any]:
    report_obj = safe_read_json(daily_report_path)
    if not isinstance(report_obj, dict):
        report_obj = {}
    quant_pack = report_obj.get("quant_pack") if isinstance(report_obj.get("quant_pack"), dict) else {}
    quant_pack["replay_drift"] = dict(summary)
    report_obj["quant_pack"] = quant_pack

    bak = _backup_file(daily_report_path) if backup else None
    if atomic:
        _write_json_atomic(daily_report_path, report_obj)
    else:
        daily_report_path.parent.mkdir(parents=True, exist_ok=True)
        daily_report_path.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    return {
        "report_path": str(daily_report_path.resolve()),
        "backup_path": str(bak) if bak else "",
    }


def attach_replay_drift_to_daily(
    *,
    daily_base: Path,
    date_str: str = "",
    strict: bool = False,
    fail_on_drift: Optional[bool] = None,
    rules: Optional[Dict[str, Any]] = None,
    out_base: Optional[Path] = None,
    extra_search_dirs: Optional[Sequence[Path]] = None,
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    daily_base = daily_base.resolve()
    date_norm = _parse_date(date_str) if date_str else _latest_daily_date(daily_base)
    if not date_norm:
        return 2, {
            "schema_version": 1,
            "status": "FAIL",
            "reason": "date_unavailable",
            "daily_base": str(daily_base),
            "generated_at_utc": _now_utc_iso(),
        }

    strict_flag = bool(strict)
    fail_flag = bool(fail_on_drift) if fail_on_drift is not None else bool(strict_flag)
    daily_report_path = (daily_base / f"{date_norm}.json").resolve()
    drift_out_dir = out_base.resolve() if isinstance(out_base, Path) else (daily_base / "quant_packs" / date_norm / "replay_drift").resolve()
    drift_out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    replay_dir = discover_replay_window_dir(
        daily_base,
        date_norm,
        extra_search_dirs=extra_search_dirs,
        verbose=verbose,
    )

    summary: Dict[str, Any]
    gate_rc = 0
    gate_result: Dict[str, Any] = {}
    reused = False

    if replay_dir is None:
        summary = {
            "status": "MISSING",
            "strict": strict_flag,
            "fail_on_drift": fail_flag,
            "cycles": 0,
            "fails": 0,
            "warns": 0,
            "worst_cycle": {},
            "tag_top": [],
            "generated_utc": _now_utc_iso(),
            "warnings": ["replay_window_not_found"],
        }
        warnings.append("replay_window_not_found")
        if strict_flag:
            gate_rc = 2
    else:
        gate_rc, gate_result, reused = run_or_load_drift_gate(
            replay_window_dir=replay_dir,
            out_dir=drift_out_dir,
            strict=strict_flag,
            fail_on_drift=fail_flag,
            rules=rules,
            verbose=verbose,
        )
        summary = extract_drift_summary(gate_result)

    write_info: Dict[str, Any] = {}
    if daily_report_path.exists():
        write_info = write_daily_json_quant_replay_drift(
            daily_report_path,
            summary,
            backup=True,
            atomic=True,
        )
    else:
        warnings.append("daily_report_missing")
        if strict_flag:
            gate_rc = 2

    status = str(summary.get("status") or "MISSING").upper()
    rc = 0
    if gate_rc == 2:
        rc = 2
    elif fail_flag and status == "FAIL":
        rc = 3
    elif status in ("WARN", "MISSING", "FAIL") or warnings:
        rc = 1
    else:
        rc = 0

    manifest = {
        "schema_version": 1,
        "date": date_norm,
        "daily_base": str(daily_base),
        "daily_report_path": str(daily_report_path),
        "replay_window_dir": str(replay_dir) if replay_dir is not None else "",
        "drift_gate_out_dir": str(drift_out_dir),
        "strict": strict_flag,
        "fail_on_drift": fail_flag,
        "status": status,
        "rc": int(rc),
        "warnings": warnings,
        "summary": summary,
        "write_info": write_info,
        "reused_existing_result": bool(reused),
        "generated_at_utc": _now_utc_iso(),
    }
    _write_json_atomic((drift_out_dir / "replay_drift_manifest.json").resolve(), manifest)
    return int(rc), manifest
