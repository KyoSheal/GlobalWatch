#!/usr/bin/env python3
"""Safely clean temporary test artifacts under outputs/ without touching formal runs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json


MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
ROOT_DIR_TEST_RE = re.compile(r"^test_", re.IGNORECASE)
ROOT_FILE_PATTERNS = [
    re.compile(r"^cfg_.*\.json$", re.IGNORECASE),
    re.compile(r"^tmp_.*\.json$", re.IGNORECASE),
    re.compile(r"^.*_config\.json$", re.IGNORECASE),
    re.compile(r"^.*_io_test\.json$", re.IGNORECASE),
]
DIR_MARKERS = ["_test", "_diag", "_smoke", "_probe", "_regression", "_io_test"]
PREVIEW_LIMIT = 50


@dataclass
class Candidate:
    path: Path
    explicit_test: bool
    source: str


def _is_month_name(name: str) -> bool:
    return bool(MONTH_RE.match(str(name or "").strip()))


def _is_root_test_dir_name(name: str) -> bool:
    n = str(name or "").strip().lower()
    if ROOT_DIR_TEST_RE.match(n):
        return True
    return any(marker in n for marker in DIR_MARKERS)


def _is_root_temp_file_name(name: str) -> bool:
    n = str(name or "").strip()
    return any(pat.match(n) for pat in ROOT_FILE_PATTERNS)


def _dir_contains_run_summary(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "run_summary.json").exists():
        return True
    try:
        for hit in path.rglob("run_summary.json"):
            if hit.is_file():
                return True
    except Exception:
        return False
    return False


def _dir_contains_key_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    checks = [
        path / "run_summary.json",
        path / "trade_history.jsonl",
        path / "telemetry" / "events.jsonl",
    ]
    return any(p.exists() for p in checks)


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        return total
    except Exception:
        return 0


def _path_mtime(path: Path) -> datetime:
    try:
        ts = float(path.stat().st_mtime)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _is_whitelisted_path(path: Path, base_out_dir: Path) -> bool:
    try:
        rel = path.resolve().relative_to(base_out_dir.resolve())
    except Exception:
        return True
    rel_parts = rel.parts
    if not rel_parts:
        return True
    top = rel_parts[0]
    if top in {"LATEST.json", "registry.jsonl"}:
        return True
    if top == "reports":
        return True
    if _is_month_name(top):
        # Month roots and all formal run folders under month must be preserved by default.
        if len(rel_parts) == 1:
            return True
        if len(rel_parts) >= 2:
            run_dir = base_out_dir / rel_parts[0] / rel_parts[1]
            if run_dir.is_dir() and (run_dir / "run_summary.json").exists():
                return True
            if len(rel_parts) == 2 and run_dir.is_dir() and (run_dir / "run_summary.json").exists():
                return True
            if len(rel_parts) == 2 and rel_parts[1] == "month_summary.json":
                return True
    return False


def _scan_candidates(base_out_dir: Path, include_dryrun: bool) -> List[Candidate]:
    candidates: List[Candidate] = []
    if not base_out_dir.exists() or not base_out_dir.is_dir():
        return candidates

    for child in sorted(base_out_dir.iterdir(), key=lambda p: p.name.lower()):
        if _is_month_name(child.name):
            continue
        if child.name in {"LATEST.json", "registry.jsonl"}:
            continue
        if child.name == "reports":
            continue
        if child.is_dir() and _is_root_test_dir_name(child.name):
            candidates.append(Candidate(path=child, explicit_test=True, source="root_test_dir"))
            continue
        if child.is_file() and _is_root_temp_file_name(child.name):
            candidates.append(Candidate(path=child, explicit_test=True, source="root_temp_file"))
            continue

    if include_dryrun:
        for month_dir in sorted(base_out_dir.iterdir(), key=lambda p: p.name.lower()):
            if not month_dir.is_dir() or not _is_month_name(month_dir.name):
                continue
            for run_dir in sorted(month_dir.iterdir(), key=lambda p: p.name.lower()):
                if not run_dir.is_dir():
                    continue
                if not run_dir.name.startswith("DRYRUN-"):
                    continue
                candidates.append(Candidate(path=run_dir, explicit_test=True, source="month_dryrun_dir"))
    return candidates


def _make_trash_root(base_out_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return base_out_dir / "_trash" / stamp


def _validate_candidate(
    c: Candidate,
    base_out_dir: Path,
    recent_cutoff: datetime,
) -> Tuple[bool, str]:
    path = c.path
    if _is_whitelisted_path(path, base_out_dir):
        return False, "kept_by_whitelist"

    if _path_mtime(path) >= recent_cutoff:
        return False, "kept_by_recent"

    if path.is_dir():
        # Hard protection: any formal run dir under month with run_summary.json is never removable.
        try:
            rel = path.resolve().relative_to(base_out_dir.resolve())
        except Exception:
            return False, "kept_by_whitelist"
        parts = rel.parts
        if len(parts) >= 2 and _is_month_name(parts[0]) and (path / "run_summary.json").exists():
            return False, "kept_by_contains_run_summary"

        if _dir_contains_run_summary(path) and not c.explicit_test:
            return False, "kept_by_contains_run_summary"

        if _dir_contains_key_files(path) and not c.explicit_test:
            return False, "kept_by_contains_run_summary"

    return True, "ok"


def _apply_action(path: Path, mode: str, base_out_dir: Path, trash_root: Optional[Path]) -> None:
    if mode == "move":
        if trash_root is None:
            raise RuntimeError("trash_root is required for move mode")
        rel = path.resolve().relative_to(base_out_dir.resolve())
        dst = trash_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dst))
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def cleanup_outputs(
    base_out_dir: str = "outputs",
    apply: bool = False,
    mode: str = "delete",
    include_dryrun: bool = False,
    keep_days: int = 7,
) -> Dict[str, Any]:
    base = Path(str(base_out_dir or "outputs")).resolve()
    mode_norm = str(mode or "delete").strip().lower()
    if mode_norm not in {"delete", "move"}:
        mode_norm = "delete"
    dry_run = not bool(apply)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=max(0, int(keep_days)))

    raw_candidates = _scan_candidates(base, include_dryrun=include_dryrun)
    selected: List[Candidate] = []
    skipped: List[Dict[str, Any]] = []
    skip_stats = {
        "kept_by_whitelist": 0,
        "kept_by_recent": 0,
        "kept_by_contains_run_summary": 0,
        "unknown": 0,
    }

    for cand in raw_candidates:
        ok, reason = _validate_candidate(cand, base, cutoff)
        if ok:
            selected.append(cand)
            continue
        if reason not in skip_stats:
            reason = "unknown"
        skip_stats[reason] += 1
        skipped.append({"path": str(cand.path), "reason": reason, "source": cand.source})

    planned_items: List[Dict[str, Any]] = []
    for cand in selected:
        planned_items.append(
            {
                "path": str(cand.path),
                "type": "dir" if cand.path.is_dir() else "file",
                "source": cand.source,
                "explicit_test": bool(cand.explicit_test),
                "size_bytes": _path_size_bytes(cand.path),
                "mtime_utc": _path_mtime(cand.path).isoformat(),
            }
        )

    trash_root: Optional[Path] = None
    moved_count = 0
    deleted_count = 0
    action_errors: List[Dict[str, str]] = []
    if apply and mode_norm == "move":
        trash_root = _make_trash_root(base)
        trash_root.mkdir(parents=True, exist_ok=True)

    if apply:
        for cand in selected:
            try:
                _apply_action(cand.path, mode_norm, base, trash_root)
                if mode_norm == "move":
                    moved_count += 1
                else:
                    deleted_count += 1
            except Exception as exc:
                action_errors.append({"path": str(cand.path), "error": str(exc)})

    preview_path = base / "_cleanup_preview.json"
    preview_written = False
    if len(planned_items) > PREVIEW_LIMIT:
        payload = {
            "schema_version": 1,
            "updated_at_utc": now_utc.isoformat(),
            "base_out_dir": str(base),
            "dry_run": dry_run,
            "mode": mode_norm,
            "include_dryrun": bool(include_dryrun),
            "keep_days": int(keep_days),
            "candidate_count": len(raw_candidates),
            "selected_count": len(selected),
            "planned_items": planned_items,
            "skipped": skipped,
        }
        atomic_write_json(str(preview_path), payload, indent=2)
        preview_written = True

    total_size = int(sum(int(i.get("size_bytes", 0) or 0) for i in planned_items))

    return {
        "schema_version": 1,
        "updated_at_utc": now_utc.isoformat(),
        "base_out_dir": str(base),
        "dry_run": dry_run,
        "apply": bool(apply),
        "mode": mode_norm,
        "include_dryrun": bool(include_dryrun),
        "keep_days": int(keep_days),
        "candidate_count": len(raw_candidates),
        "selected_count": len(selected),
        "planned_total_size_bytes": total_size,
        "preview_limit": PREVIEW_LIMIT,
        "preview_written": preview_written,
        "preview_path": str(preview_path) if preview_written else None,
        "planned_items_preview": planned_items[:PREVIEW_LIMIT],
        "planned_items_omitted": max(0, len(planned_items) - PREVIEW_LIMIT),
        "skipped_reason_stats": skip_stats,
        "skipped_preview": skipped[:PREVIEW_LIMIT],
        "skipped_omitted": max(0, len(skipped) - PREVIEW_LIMIT),
        "deleted_count": deleted_count,
        "moved_count": moved_count,
        "trash_dir": str(trash_root) if trash_root is not None else None,
        "action_error_count": len(action_errors),
        "action_errors_preview": action_errors[:PREVIEW_LIMIT],
    }


def _print_summary(result: Dict[str, Any]) -> None:
    print("[CLEANUP OUTPUTS] summary")
    print(f"  base_out_dir: {result.get('base_out_dir')}")
    print(
        f"  mode: {result.get('mode')} apply={result.get('apply')} dry_run={result.get('dry_run')} "
        f"include_dryrun={result.get('include_dryrun')} keep_days={result.get('keep_days')}"
    )
    print(
        f"  candidates={result.get('candidate_count')} selected={result.get('selected_count')} "
        f"size_bytes={result.get('planned_total_size_bytes')}"
    )
    print(f"  deleted={result.get('deleted_count')} moved={result.get('moved_count')}")
    if result.get("trash_dir"):
        print(f"  trash_dir: {result.get('trash_dir')}")
    if result.get("preview_written"):
        print(f"  preview_path: {result.get('preview_path')}")

    stats = result.get("skipped_reason_stats") or {}
    print(
        "  skipped_reason_stats: "
        f"kept_by_whitelist={stats.get('kept_by_whitelist', 0)} "
        f"kept_by_recent={stats.get('kept_by_recent', 0)} "
        f"kept_by_contains_run_summary={stats.get('kept_by_contains_run_summary', 0)} "
        f"unknown={stats.get('unknown', 0)}"
    )

    planned_preview = result.get("planned_items_preview") or []
    if planned_preview:
        print("  candidates_preview:")
        for item in planned_preview:
            if not isinstance(item, dict):
                continue
            print(
                f"    - {item.get('path')} [{item.get('type')}] "
                f"source={item.get('source')} size={item.get('size_bytes')}"
            )
    omitted = int(result.get("planned_items_omitted") or 0)
    if omitted > 0:
        print(f"  ... and {omitted} more candidates (see preview json)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely cleanup temporary outputs artifacts.")
    parser.add_argument("--base-out-dir", type=str, default="outputs")
    parser.add_argument("--apply", action="store_true", help="Apply delete/move. Without this, dry-run only.")
    parser.add_argument("--mode", type=str, default="delete", choices=["delete", "move"])
    parser.add_argument("--include-dryrun", action="store_true", help="Include outputs/YYYY-MM/DRYRUN-* candidates.")
    parser.add_argument("--keep-days", type=int, default=7, help="Keep recent test artifacts within N days.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    result = cleanup_outputs(
        base_out_dir=args.base_out_dir,
        apply=bool(args.apply),
        mode=args.mode,
        include_dryrun=bool(args.include_dryrun),
        keep_days=int(args.keep_days),
    )
    if args.json:
        import json

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

