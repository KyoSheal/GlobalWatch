#!/usr/bin/env python3
"""T64: execution blockers + no-trade attach minimal regression."""

from __future__ import annotations

import csv
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


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="exec_blockers_no_trade_min_"))
    try:
        daily_base = (tmp / "outputs" / "Daily Report").resolve()
        date_str = "2026-02-18"
        report_path = (daily_base / f"{date_str}.json").resolve()
        _write_json(
            report_path,
            {
                "date": date_str,
                "schema_version": 1,
                "quant_pack": {"summary": {"trades_total": 0}},
            },
        )

        run_dataset = (daily_base / "quant_packs" / date_str / "run_dataset").resolve()
        cycles_rows = [
            {"cycle_id": "1", "time_utc": "2026-02-18T14:30:00+00:00", "skip_reason": "attempt_cooldown", "status": "skip"},
            {"cycle_id": "2", "time_utc": "2026-02-18T15:00:00+00:00", "skip_reason": "attempt_cooldown", "status": "skip"},
            {"cycle_id": "3", "time_utc": "2026-02-18T15:30:00+00:00", "skip_reason": "attempt_cooldown", "status": "skip"},
            {"cycle_id": "4", "time_utc": "2026-02-18T16:00:00+00:00", "skip_reason": "attempt_cooldown", "status": "skip"},
            {"cycle_id": "5", "time_utc": "2026-02-18T16:30:00+00:00", "skip_reason": "attempt_cooldown", "status": "skip"},
            {"cycle_id": "6", "time_utc": "2026-02-18T17:00:00+00:00", "skip_reason": "market_closed", "status": "skip"},
        ]
        _write_csv(run_dataset / "cycles.csv", ["cycle_id", "time_utc", "skip_reason", "status"], cycles_rows)
        _write_csv(
            run_dataset / "trades.csv",
            ["time_utc", "cycle_id", "ticker", "side", "qty", "price", "notional"],
            [],
        )

        cmds = [
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a19_compute_exec_blockers.py"),
                "--daily-base",
                str(daily_base),
                "--date",
                date_str,
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a20_attach_exec_blockers_to_daily.py"),
                "--daily-base",
                str(daily_base),
                "--date",
                date_str,
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a7_update_daily_reports_index.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "3650",
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "quant" / "a20_build_quant_alerts.py"),
                "--daily-base",
                str(daily_base),
                "--lookback-days",
                "3650",
            ],
        ]
        for cmd in cmds:
            p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            if p.returncode not in (0, 1):
                print(p.stdout)
                print(p.stderr)
                return _fail(f"command failed rc={p.returncode}: {' '.join(cmd)}")

        # idempotency: rerun compute+attach
        for cmd in cmds[:2]:
            p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            if p.returncode not in (0, 1):
                print(p.stdout)
                print(p.stderr)
                return _fail(f"idempotency command failed rc={p.returncode}: {' '.join(cmd)}")

        daily_obj = json.loads(report_path.read_text(encoding="utf-8"))
        qp = daily_obj.get("quant_pack") if isinstance(daily_obj.get("quant_pack"), dict) else {}
        eb = qp.get("execution_blockers") if isinstance(qp.get("execution_blockers"), dict) else {}
        nt = qp.get("no_trade") if isinstance(qp.get("no_trade"), dict) else {}
        top3 = eb.get("top3") if isinstance(eb.get("top3"), list) else []
        top1 = top3[0] if top3 and isinstance(top3[0], dict) else {}
        if str(top1.get("reason", "")) != "attempt_cooldown":
            return _fail(f"unexpected top1 reason: {top1}")
        if not bool(nt.get("is_no_trade_day", False)):
            return _fail("expected is_no_trade_day=true")
        if str(nt.get("primary_reason", "")) != "attempt_cooldown":
            return _fail(f"unexpected no_trade primary_reason: {nt.get('primary_reason')}")

        idx_obj = json.loads((daily_base / "daily_reports_index.json").read_text(encoding="utf-8"))
        rows = idx_obj.get("reports") if isinstance(idx_obj.get("reports"), list) else []
        row = next((r for r in rows if isinstance(r, dict) and str(r.get("date", "")) == date_str), None)
        if not isinstance(row, dict):
            return _fail("index row missing for test date")
        quant = row.get("quant") if isinstance(row.get("quant"), dict) else {}
        if str(quant.get("exec_blocker_top1_reason", "")) != "attempt_cooldown":
            return _fail("index quant.exec_blocker_top1_reason mismatch")
        if not bool(quant.get("no_trade_flag", False)):
            return _fail("index quant.no_trade_flag expected true")

        alerts_obj = json.loads((daily_base / "alerts.json").read_text(encoding="utf-8"))
        alerts = alerts_obj.get("alerts") if isinstance(alerts_obj.get("alerts"), list) else []
        rule_ids = {str(a.get("rule_id", "")) for a in alerts if isinstance(a, dict)}
        if ("no_trade_day" not in rule_ids) and ("exec_blocker_dominant_cyclelevel" not in rule_ids):
            return _fail(f"expected no_trade_day or exec_blocker_dominant_cyclelevel in alerts, got {sorted(rule_ids)}")

        print("[PASS] exec_blockers_no_trade_attach_minimal")
        print(f"[INFO] daily_json={report_path}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

