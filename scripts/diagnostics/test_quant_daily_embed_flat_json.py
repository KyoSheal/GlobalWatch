#!/usr/bin/env python3
"""T39: flat JSON daily report embedding for quant pack."""

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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _run_embed(*, daily_base: Path, date_str: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "quant" / "a6_embed_quant_into_daily_report.py"),
        "--daily-base",
        str(daily_base),
        "--date",
        str(date_str),
        "--verbose",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="quant_daily_embed_flat_"))
    try:
        daily_base = tmp_root / "outputs" / "Daily Report"
        date_str = "2026-02-18"
        daily_json = daily_base / f"{date_str}.json"
        quant_md = daily_base / "quant_packs" / date_str / "daily_quant_report.md"
        metrics_json = daily_base / "quant_packs" / date_str / "metrics" / "metrics.json"
        gate_json = daily_base / "quant_packs" / date_str / "gate" / "gate_result.json"

        _write_json(daily_json, {"date": date_str, "title": "Daily Report"})
        _write_text(
            quant_md,
            "# Daily Quant Pack\n\n## Today Metrics\n- Total Return: 1.23%\n\n## Gate\n- PASS\n",
        )
        _write_json(
            metrics_json,
            {
                "performance": {"total_return": 0.0123, "cagr": 0.10},
                "risk": {"vol_annualized": 0.2, "sharpe": 1.5, "max_drawdown": -0.08},
                "trading": {"trades_total": 4},
            },
        )
        _write_json(gate_json, {"status": "PASS"})

        p1 = _run_embed(daily_base=daily_base, date_str=date_str)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first embed expected rc=0 got={p1.returncode}")

        # Backup created and quant_pack inserted
        bak = daily_json.with_name(daily_json.name + ".bak")
        if not bak.exists():
            return _fail(".bak backup not created")
        obj1 = json.loads(daily_json.read_text(encoding="utf-8"))
        if "quant_pack" not in obj1:
            return _fail("quant_pack missing after embed")
        qp = obj1.get("quant_pack", {})
        if str((qp.get("summary") or {}).get("gate_status", "")).upper() != "PASS":
            return _fail("quant_pack.summary.gate_status mismatch")

        # idempotent overwrite
        p2 = _run_embed(daily_base=daily_base, date_str=date_str)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second embed expected rc=0 got={p2.returncode}")
        obj2 = json.loads(daily_json.read_text(encoding="utf-8"))
        if "quant_pack" not in obj2:
            return _fail("quant_pack missing after second embed")
        if len([k for k in obj2.keys() if k == "quant_pack"]) != 1:
            return _fail("quant_pack appears duplicated")

        manifest = daily_base / "quant_packs" / date_str / "embed_manifest.json"
        if not manifest.exists():
            return _fail("embed_manifest.json missing in quant_packs/date")

        print("[PASS] quant_daily_embed_flat_json")
        print(f"[INFO] daily_json={daily_json}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

