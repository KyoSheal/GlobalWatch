#!/usr/bin/env python3
"""T38: minimal regression test for embedding quant report into daily report."""

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


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _run_embed(*, daily_dir: Path, strict: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a6_embed_quant_into_daily_report.py"),
        "--daily-dir",
        str(daily_dir),
        "--verbose",
    ]
    if strict:
        cmd.append("--strict")
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_daily_embed_min_"))
    try:
        # Case 1: report exists + quant exists, run twice and ensure idempotent replace
        daily_ok = tmp_root / "Daily Report" / "2026-02-20"
        report_path = daily_ok / "daily_report.md"
        quant_md = daily_ok / "quant" / "daily_quant_report.md"
        _write_text(
            report_path,
            "# Daily Report\n\nExisting body.\n",
        )
        _write_text(
            quant_md,
            "# Daily Quant Pack\n\n## Today Metrics\n- Total Return: 1.23%\n\n## Gate\n- PASS\n",
        )

        p1 = _run_embed(daily_dir=daily_ok, strict=False)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first embed expected rc=0, got rc={p1.returncode}")

        content1 = report_path.read_text(encoding="utf-8")
        if "<!-- QUANT_PACK_BEGIN -->" not in content1 or "<!-- QUANT_PACK_END -->" not in content1:
            return _fail("marker block not inserted on first run")

        p2 = _run_embed(daily_dir=daily_ok, strict=False)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second embed expected rc=0, got rc={p2.returncode}")
        content2 = report_path.read_text(encoding="utf-8")

        if content2.count("<!-- QUANT_PACK_BEGIN -->") != 1 or content2.count("<!-- QUANT_PACK_END -->") != 1:
            return _fail("marker block duplicated after second run")
        if len(content2.splitlines()) > len(content1.splitlines()) + 5:
            return _fail("report appears to grow unexpectedly after second run")

        manifest_path = daily_ok / "quant" / "embed_manifest.json"
        if not manifest_path.exists():
            return _fail("embed_manifest.json missing")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_obj = json.load(f)
        out_field = str(manifest_obj.get("daily_report_out", "") or manifest_obj.get("out_file", ""))
        if not out_field.endswith("daily_report.md"):
            return _fail("embed_manifest output path not pointing to daily_report.md")

        # Case 2: report missing, strict=false -> fallback file created
        daily_missing = tmp_root / "Daily Report" / "2026-02-21"
        _write_text(
            daily_missing / "quant" / "daily_quant_report.md",
            "# Daily Quant Pack\n\n## Gate\n- PASS\n",
        )
        p3 = _run_embed(daily_dir=daily_missing, strict=False)
        if p3.returncode != 0:
            print(p3.stdout)
            print(p3.stderr)
            return _fail(f"missing report strict=false expected rc=0, got rc={p3.returncode}")
        fallback = daily_missing / "daily_report_with_quant.md"
        if not fallback.exists():
            return _fail("fallback daily_report_with_quant.md not created")

        # Case 3: report missing, strict=true -> rc=3
        daily_strict = tmp_root / "Daily Report" / "2026-02-22"
        _write_text(
            daily_strict / "quant" / "daily_quant_report.md",
            "# Daily Quant Pack\n\n## Gate\n- PASS\n",
        )
        p4 = _run_embed(daily_dir=daily_strict, strict=True)
        if p4.returncode != 3:
            print(p4.stdout)
            print(p4.stderr)
            return _fail(f"missing report strict=true expected rc=3, got rc={p4.returncode}")

        print("[PASS] quant_daily_embed_minimal")
        print(f"[INFO] report={report_path}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
