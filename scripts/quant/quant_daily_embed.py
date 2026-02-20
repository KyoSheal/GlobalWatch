#!/usr/bin/env python3
"""A1-6 helper: embed quant pack output into markdown/txt/json daily reports."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MARKER_BEGIN = "<!-- QUANT_PACK_BEGIN -->"
MARKER_END = "<!-- QUANT_PACK_END -->"


@dataclass
class EmbedResult:
    exit_code: int
    mode: str
    is_json: bool
    daily_report_in: Optional[Path]
    daily_report_out: Optional[Path]
    quant_md: Optional[Path]
    quant_pack_dir: Optional[Path]
    created_fallback: bool
    warnings: List[str]
    notes: List[str]

    def to_manifest(self, *, daily_dir: Path) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "daily_dir": str(daily_dir.resolve()),
            "daily_report_in": str(self.daily_report_in.resolve()) if self.daily_report_in else "",
            "daily_report_out": str(self.daily_report_out.resolve()) if self.daily_report_out else "",
            "mode": self.mode,
            "is_json": bool(self.is_json),
            "quant_md_used": str(self.quant_md.resolve()) if self.quant_md else "",
            "quant_pack_dir": str(self.quant_pack_dir.resolve()) if self.quant_pack_dir else "",
            "created_fallback": bool(self.created_fallback),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "exit_code": int(self.exit_code),
        }


def _parse_date_str(s: str) -> Optional[str]:
    t = str(s or "").strip()
    if not t:
        return None
    try:
        return datetime.fromisoformat(t).date().isoformat()
    except Exception:
        pass
    try:
        return datetime.strptime(t, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def _write_json_atomic(path: Path, obj: Any) -> None:
    _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _backup_existing(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    primary = path.with_name(path.name + ".bak")
    if not primary.exists():
        shutil.copy2(path, primary)
        return primary
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    fallback = path.with_name(path.name + f".{stamp}.bak")
    shutil.copy2(path, fallback)
    return fallback


def _latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)[0]


def discover_daily_report_file(daily_dir: Path) -> Optional[Path]:
    p1 = daily_dir / "daily_report.md"
    if p1.exists() and p1.is_file():
        return p1
    p2 = daily_dir / "daily_report.txt"
    if p2.exists() and p2.is_file():
        return p2
    candidates: List[Path] = []
    for pat in ("*daily*report*.md", "*Daily*Report*.md", "*report*.md"):
        for p in daily_dir.glob(pat):
            if p.is_file():
                candidates.append(p)
    return _latest_by_mtime(candidates)


def discover_quant_md(
    *,
    daily_dir: Optional[Path],
    daily_base: Optional[Path],
    date_str: str,
    quant_md_arg: str,
) -> Optional[Path]:
    if str(quant_md_arg or "").strip():
        p = Path(quant_md_arg).resolve()
        return p if p.exists() else p

    candidates: List[Path] = []
    if daily_dir is not None:
        p = (daily_dir / "quant" / "daily_quant_report.md").resolve()
        if p.exists():
            candidates.append(p)

    if daily_base is not None and date_str:
        c1 = (daily_base / "quant_packs" / date_str / "daily_quant_report.md").resolve()
        c2 = (daily_base / "quant" / date_str / "daily_quant_report.md").resolve()
        if c1.exists():
            candidates.append(c1)
        if c2.exists():
            candidates.append(c2)
        for p in daily_base.rglob("daily_quant_report.md"):
            if date_str in str(p):
                candidates.append(p.resolve())

    return _latest_by_mtime(candidates)


def _build_quant_block(quant_md_text: str, *, links: Dict[str, str]) -> str:
    quant_body = str(quant_md_text or "").strip()
    link_lines = [
        f"- metrics: `{links.get('metrics_md', 'quant/metrics/metrics.md')}`",
        f"- gate: `{links.get('gate_report_md', 'quant/gate/gate_report.md')}`",
        f"- leaderboard: `{links.get('leaderboard_md', 'quant/leaderboard/leaderboard.md')}`",
    ]
    body = "## Quant Pack (Auto)\n\n" + quant_body + "\n\n### Quant Links\n" + "\n".join(link_lines) + "\n"
    return f"{MARKER_BEGIN}\n{body}\n{MARKER_END}\n"


def _replace_marker_block(text: str, block: str) -> str:
    src = str(text or "")
    bi = src.find(MARKER_BEGIN)
    ei = src.find(MARKER_END)
    if bi >= 0 and ei > bi:
        ei2 = ei + len(MARKER_END)
        if ei2 < len(src) and src[ei2 : ei2 + 1] == "\n":
            ei2 += 1
        return src[:bi].rstrip() + "\n\n" + block
    if src.strip():
        return src.rstrip() + "\n\n" + block
    return block


def _append_marker_block(text: str, block: str) -> str:
    src = str(text or "")
    if src.strip():
        return src.rstrip() + "\n\n" + block
    return block


def _read_json_safely(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _relative_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _build_quant_pack_payload(
    *,
    quant_md: Path,
    report_out: Path,
) -> Dict[str, Any]:
    pack_dir = quant_md.parent
    metrics_json = pack_dir / "metrics" / "metrics.json"
    gate_result_json = pack_dir / "gate" / "gate_result.json"
    pack_manifest_json = pack_dir / "pack_manifest.json"
    leaderboard_md = pack_dir / "leaderboard" / "leaderboard.md"
    gate_report_md = pack_dir / "gate" / "gate_report.md"
    metrics_md = pack_dir / "metrics" / "metrics.md"

    metrics_obj = _read_json_safely(metrics_json) or {}
    gate_obj = _read_json_safely(gate_result_json) or {}
    pack_obj = _read_json_safely(pack_manifest_json) or {}

    perf = (metrics_obj.get("performance") or {}) if isinstance(metrics_obj, dict) else {}
    risk = (metrics_obj.get("risk") or {}) if isinstance(metrics_obj, dict) else {}
    trading = (metrics_obj.get("trading") or {}) if isinstance(metrics_obj, dict) else {}
    gate_status = str(gate_obj.get("status", "") or "").upper() if isinstance(gate_obj, dict) else ""
    if not gate_status:
        gate_status = "NA"

    artifacts = {
        "dataset_dir": str(pack_obj.get("dataset_dir", "") or ""),
        "metrics_md": _relative_or_abs(metrics_md, report_out.parent) if metrics_md.exists() else "",
        "leaderboard_md": _relative_or_abs(leaderboard_md, report_out.parent) if leaderboard_md.exists() else "",
        "gate_report_md": _relative_or_abs(gate_report_md, report_out.parent) if gate_report_md.exists() else "",
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pack_md_path": _relative_or_abs(quant_md, report_out.parent),
        "pack_dir": str(pack_dir.resolve()),
        "summary": {
            "total_return": perf.get("total_return"),
            "cagr": perf.get("cagr"),
            "vol_annualized": risk.get("vol_annualized"),
            "sharpe": risk.get("sharpe"),
            "max_drawdown": risk.get("max_drawdown"),
            "trades_total": trading.get("trades_total"),
            "gate_status": gate_status,
        },
        "artifacts": artifacts,
    }


def embed_quant_into_daily_report(
    *,
    daily_dir: Optional[Path],
    daily_base: Optional[Path],
    date_str: str,
    quant_md: Optional[Path],
    report_file: Optional[Path],
    mode: str,
    out_file: Optional[Path],
    strict: bool,
) -> EmbedResult:
    warnings: List[str] = []
    notes: List[str] = []
    created_fallback = False
    mode_norm = str(mode or "replace").strip().lower()
    if mode_norm not in {"append", "replace"}:
        mode_norm = "replace"

    date_norm = _parse_date_str(date_str) if date_str else None
    daily_dir_r = daily_dir.resolve() if daily_dir is not None else None
    daily_base_r = daily_base.resolve() if daily_base is not None else None
    quant_md_r = quant_md.resolve() if quant_md is not None else None
    report_file_r = report_file.resolve() if report_file is not None else None
    out_file_r = out_file.resolve() if out_file is not None else None

    # Default report file resolution
    if report_file_r is None:
        if daily_dir_r is not None:
            report_file_r = discover_daily_report_file(daily_dir_r)
        elif daily_base_r is not None and date_norm:
            report_file_r = (daily_base_r / f"{date_norm}.json").resolve()

    if quant_md_r is None:
        quant_md_r = discover_quant_md(
            daily_dir=daily_dir_r,
            daily_base=daily_base_r,
            date_str=date_norm or "",
            quant_md_arg="",
        )

    if quant_md_r is None or not quant_md_r.exists():
        warnings.append(f"quant_md_not_found:{quant_md_r}")
        if strict:
            return EmbedResult(3, mode_norm, False, report_file_r, out_file_r, quant_md_r, None, False, warnings, notes)
        return EmbedResult(0, mode_norm, False, report_file_r, out_file_r, quant_md_r, None, False, warnings, notes)

    # Decide output file
    target_out = out_file_r
    if target_out is None:
        if report_file_r is not None and report_file_r.exists():
            target_out = report_file_r
        else:
            if strict:
                warnings.append("report_file_not_found_strict")
                return EmbedResult(3, mode_norm, False, report_file_r, target_out, quant_md_r, quant_md_r.parent, False, warnings, notes)
            if report_file_r is not None and report_file_r.suffix.lower() == ".json":
                stem = report_file_r.stem
                target_out = report_file_r.with_name(stem + "_with_quant.json")
            elif daily_base_r is not None and date_norm:
                target_out = (daily_base_r / f"{date_norm}_with_quant.json").resolve()
            elif daily_dir_r is not None:
                target_out = (daily_dir_r / "daily_report_with_quant.md").resolve()
            else:
                target_out = (quant_md_r.parent / "daily_report_with_quant.md").resolve()
            created_fallback = True
            notes.append("fallback_output_created")

    is_json = str(target_out.suffix).lower() == ".json"
    if not is_json:
        base_text = ""
        if target_out.exists():
            base_text = _read_text(target_out)
        elif report_file_r is not None and report_file_r.exists():
            base_text = _read_text(report_file_r)
        elif created_fallback:
            base_text = "# Daily Report (Auto)\n\n"

        links = {
            "metrics_md": _relative_or_abs(quant_md_r.parent / "metrics" / "metrics.md", target_out.parent),
            "gate_report_md": _relative_or_abs(quant_md_r.parent / "gate" / "gate_report.md", target_out.parent),
            "leaderboard_md": _relative_or_abs(quant_md_r.parent / "leaderboard" / "leaderboard.md", target_out.parent),
        }
        block = _build_quant_block(_read_text(quant_md_r), links=links)
        out_text = _append_marker_block(base_text, block) if mode_norm == "append" else _replace_marker_block(base_text, block)
        _backup_existing(target_out)
        _write_text_atomic(target_out, out_text)
        return EmbedResult(0, mode_norm, False, report_file_r, target_out, quant_md_r, quant_md_r.parent, created_fallback, warnings, notes)

    # JSON mode
    base_obj: Dict[str, Any] = {}
    if target_out.exists():
        parsed = _read_json_safely(target_out)
        if parsed is None:
            if strict:
                warnings.append("target_json_invalid_strict")
                return EmbedResult(3, mode_norm, True, report_file_r, target_out, quant_md_r, quant_md_r.parent, created_fallback, warnings, notes)
            notes.append("target_json_invalid_reinitialized")
            base_obj = {}
        else:
            base_obj = parsed
    elif report_file_r is not None and report_file_r.exists():
        parsed = _read_json_safely(report_file_r)
        if parsed is None:
            if strict:
                warnings.append("source_json_invalid_strict")
                return EmbedResult(3, mode_norm, True, report_file_r, target_out, quant_md_r, quant_md_r.parent, created_fallback, warnings, notes)
            notes.append("source_json_invalid_reinitialized")
            base_obj = {}
        else:
            base_obj = parsed
    else:
        if strict and out_file_r is None:
            warnings.append("source_json_missing_strict")
            return EmbedResult(3, mode_norm, True, report_file_r, target_out, quant_md_r, quant_md_r.parent, created_fallback, warnings, notes)
        base_obj = {}

    payload = _build_quant_pack_payload(quant_md=quant_md_r, report_out=target_out)
    base_obj["quant_pack"] = payload
    _backup_existing(target_out)
    _write_json_atomic(target_out, base_obj)
    return EmbedResult(0, mode_norm, True, report_file_r, target_out, quant_md_r, quant_md_r.parent, created_fallback, warnings, notes)


def write_embed_manifest(
    *,
    daily_dir: Optional[Path],
    daily_base: Optional[Path],
    date_str: str,
    result: EmbedResult,
) -> Path:
    if result.quant_pack_dir is not None:
        manifest_path = (result.quant_pack_dir / "embed_manifest.json").resolve()
    elif daily_dir is not None:
        manifest_path = (daily_dir.resolve() / "quant" / "embed_manifest.json").resolve()
    elif daily_base is not None and date_str:
        manifest_path = (daily_base.resolve() / "quant_packs" / date_str / "embed_manifest.json").resolve()
    else:
        manifest_path = Path("embed_manifest.json").resolve()

    ref_dir = daily_dir.resolve() if daily_dir is not None else (daily_base.resolve() if daily_base is not None else manifest_path.parent)
    _write_json_atomic(manifest_path, result.to_manifest(daily_dir=ref_dir))
    return manifest_path

