#!/usr/bin/env python3
"""T40: minimal test for a7_update_daily_reports_index.py."""

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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as _f: _f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="daily_index_min_"))
    try:
        daily_base = tmp_root / "outputs" / "Daily Report"
        daily_base.mkdir(parents=True, exist_ok=True)

        # Date A: report has quant_pack directly
        _write_json(
            daily_base / "2026-02-18.json",
            {
                "date": "2026-02-18",
                "summary": {"pnl": 10},
                "quant_pack": {
                    "generated_at_utc": "2026-02-18T10:00:00+00:00",
                    "pack_md_path": "quant_packs/2026-02-18/daily_quant_report.md",
                    "summary": {
                        "total_return": 0.01,
                        "sharpe": 1.2,
                        "max_drawdown": -0.05,
                        "trades_total": 3,
                        "gate_status": "PASS",
                    },
                },
            },
        )

        # Date B: no quant_pack in daily json, but quant_packs has metrics/gate
        _write_json(daily_base / "2026-02-19.json", {"date": "2026-02-19", "summary": {"pnl": 20}})
        _write_json(
            daily_base / "quant_packs" / "2026-02-19" / "metrics" / "metrics.json",
            {
                "performance": {"total_return": 0.02},
                "risk": {"sharpe": 1.5, "max_drawdown": -0.04},
                "trading": {"trades_total": 5},
            },
        )
        _write_json(
            daily_base / "quant_packs" / "2026-02-19" / "gate" / "gate_result.json",
            {"status": "FAIL"},
        )
        (daily_base / "quant_packs" / "2026-02-19" / "daily_quant_report.md").parent.mkdir(parents=True, exist_ok=True)
        (daily_base / "quant_packs" / "2026-02-19" / "daily_quant_report.md").write_text("# Quant\n", encoding="utf-8")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
            "--daily-base",
            str(daily_base),
            "--lookback-days",
            "5000",
            "--verbose",
        ]
        p1 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"first run failed rc={p1.returncode}")

        idx = daily_base / "daily_reports_index.json"
        if not idx.exists():
            return _fail("daily_reports_index.json not created")
        bak = daily_base / "daily_reports_index.json.bak"
        if not bak.exists():
            return _fail("index backup .bak not created")

        obj1 = json.loads(idx.read_text(encoding="utf-8"))
        reports = obj1.get("reports", [])
        if len(reports) != 2:
            return _fail(f"expected 2 reports, got {len(reports)}")
        dates = [r.get("date") for r in reports]
        if dates != sorted(dates, reverse=True):
            return _fail("reports not sorted by date desc")

        by_date = {r.get("date"): r for r in reports}
        q18 = (by_date.get("2026-02-18", {}) or {}).get("quant", {})
        q19 = (by_date.get("2026-02-19", {}) or {}).get("quant", {})
        if float(q18.get("total_return")) != 0.01 or str(q18.get("gate_status")) != "PASS":
            return _fail("quant for 2026-02-18 not loaded from report.quant_pack")
        if float(q19.get("total_return")) != 0.02 or str(q19.get("gate_status")) != "FAIL":
            return _fail("quant for 2026-02-19 not loaded from quant_packs")

        # idempotent: second run keeps same number of rows and one quant block per date
        p2 = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"second run failed rc={p2.returncode}")
        obj2 = json.loads(idx.read_text(encoding="utf-8"))
        reports2 = obj2.get("reports", [])
        if len(reports2) != 2:
            return _fail("idempotency failed: report count changed")
        for r in reports2:
            if "quant" not in r or not isinstance(r["quant"], dict):
                return _fail("idempotency failed: quant field missing")

        print("[PASS] update_daily_reports_index_minimal")
        print(f"[INFO] index={idx}")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

