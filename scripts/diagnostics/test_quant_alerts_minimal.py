#!/usr/bin/env python3
"""T57: minimal regression for A4-10 quant alerts builder."""

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


def _build_index_obj(base: Path) -> dict:
    # 3 days -> trigger cooldown_dominant and drift_missing_streak
    return {
        "updated_at": "2026-02-20T00:00:00+00:00",
        "report_dir": str(base),
        "reports": [
            {
                "date": "2026-02-17",
                "quant": {
                    "total_return": 0.0,
                    "sharpe": 0.2,
                    "max_drawdown": -0.04,
                    "trades_total": 0,
                    "gate_status": "PASS",
                    "reconcile": {
                        "status": "OK",
                        "return_gap_live_minus_backtest": -0.001,
                        "turnover_gap": 0.0,
                        "cost_gap": 0.0,
                        "warnings_count": 0,
                        "evidence_summary": {
                            "gate_status": "PASS",
                            "replay_drift_status": "MISSING",
                            "gating_top3": [{"reason": "attempt_cooldown", "count": 2}],
                        },
                    },
                },
            },
            {
                "date": "2026-02-18",
                "quant": {
                    "total_return": -0.003,
                    "sharpe": 0.1,
                    "max_drawdown": -0.05,
                    "trades_total": 0,
                    "gate_status": "PASS",
                    "reconcile": {
                        "status": "OK",
                        "return_gap_live_minus_backtest": -0.006,
                        "turnover_gap": -10.0,
                        "cost_gap": -0.5,
                        "warnings_count": 1,
                        "evidence_summary": {
                            "gate_status": "PASS",
                            "replay_drift_status": "NOT_RUN",
                            "gating_top3": [{"reason": "attempt_cooldown", "count": 3}],
                        },
                    },
                },
            },
            {
                "date": "2026-02-19",
                "quant": {
                    "total_return": -0.002,
                    "sharpe": 0.3,
                    "max_drawdown": -0.03,
                    "trades_total": 1,
                    "gate_status": "PASS",
                    "reconcile": {
                        "status": "OK",
                        "return_gap_live_minus_backtest": -0.002,
                        "turnover_gap": 0.0,
                        "cost_gap": 0.0,
                        "warnings_count": 0,
                        "evidence_summary": {
                            "gate_status": "PASS",
                            "replay_drift_status": "MISSING",
                            "gating_top3": [{"reason": "attempt_cooldown", "count": 1}],
                        },
                    },
                },
            },
        ],
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_alerts_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        daily_base.mkdir(parents=True, exist_ok=True)
        (daily_base / "daily_reports_index.json").write_text(
            json.dumps(_build_index_obj(daily_base), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # build timeseries first
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
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p2.returncode != 0:
            print(p2.stdout)
            print(p2.stderr)
            return _fail(f"a20 failed rc={p2.returncode}")

        alerts_json = daily_base / "alerts.json"
        alerts_md = daily_base / "alerts.md"
        if not alerts_json.exists() or not alerts_md.exists():
            return _fail("missing alerts outputs")

        obj = json.load(open(alerts_json, "r", encoding="utf-8"))
        alerts = obj.get("alerts") if isinstance(obj.get("alerts"), list) else []
        rule_ids = {str(a.get("rule_id", "")) for a in alerts if isinstance(a, dict)}
        if "cooldown_dominant" not in rule_ids and "drift_missing_streak" not in rule_ids:
            return _fail(f"expected cooldown_dominant or drift_missing_streak, got {sorted(rule_ids)}")

        md_text = alerts_md.read_text(encoding="utf-8")
        if ("cooldown_dominant" not in md_text) and ("drift_missing_streak" not in md_text):
            return _fail("alerts.md missing expected rule headings")

        print("[PASS] quant_alerts_minimal")
        print(f"[INFO] alerts_count={len(alerts)} alerts_md={alerts_md}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

