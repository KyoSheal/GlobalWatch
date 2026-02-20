#!/usr/bin/env python3
"""T60: minimal regression for A4-12 cost_fragile alerts rule."""

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


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_alerts_cost_fragile_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)
        idx = {
            "updated_at": "2026-02-20T00:00:00+00:00",
            "report_dir": str(daily_base),
            "reports": [
                {
                    "date": "2026-02-19",
                    "quant": {
                        "total_return": 0.01,
                        "sharpe": 0.8,
                        "max_drawdown": -0.03,
                        "trades_total": 2,
                        "gate_status": "PASS",
                        "backtest_sweep": {
                            "status": "OK",
                            "break_even_cost_bps": 5.0,
                            "sensitivity_per_1bp": -0.0004,
                            "return_at_10bps": 0.001,
                            "warnings_count": 0,
                        },
                        "reconcile": {
                            "status": "OK",
                            "return_gap_live_minus_backtest": -0.001,
                            "turnover_gap": 0.0,
                            "cost_gap": 0.0,
                            "warnings_count": 0,
                            "evidence_summary": {
                                "gate_status": "PASS",
                                "replay_drift_status": "PASS",
                                "gating_top3": [{"reason": "attempt_cooldown", "count": 1}],
                            },
                        },
                    },
                }
            ],
        }
        (daily_base / "daily_reports_index.json").write_text(
            json.dumps(idx, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        p1 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a19_build_index_timeseries.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "3650",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p1.returncode != 0:
            print(p1.stdout)
            print(p1.stderr)
            return _fail(f"a19 failed rc={p1.returncode}")

        p2 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a20_build_quant_alerts.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "3650",
                "--cost-fragile-threshold-bps",
                "8",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a20 failed rc={p2.returncode}")

        alerts = json.load(open(daily_base / "alerts.json", "r", encoding="utf-8"))
        arr = alerts.get("alerts") if isinstance(alerts.get("alerts"), list) else []
        ids = {str(a.get("rule_id", "")) for a in arr if isinstance(a, dict)}
        if "cost_fragile" not in ids:
            return _fail(f"expected cost_fragile alert, got={sorted(ids)}")
        md_text = (daily_base / "alerts.md").read_text(encoding="utf-8")
        if "cost_fragile" not in md_text:
            return _fail("alerts.md missing cost_fragile section")

        print("[PASS] quant_alerts_cost_fragile_minimal")
        print(f"[INFO] alerts_json={daily_base / 'alerts.json'}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

