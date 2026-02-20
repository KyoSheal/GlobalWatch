#!/usr/bin/env python3
"""Write compact GitHub Actions job summary for quant/replay CI artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _status_emoji(status: str) -> str:
    s = str(status or "").strip().upper()
    if s in {"PASS", "SUCCESS", "OK"}:
        return "✅"
    if s in {"WARN", "WARNING", "SKIPPED"}:
        return "⚠️"
    if s in {"FAIL", "FAILED", "ERROR"}:
        return "❌"
    return "•"


def _fmt_rules_from_report(report_path: Path, max_lines: int = 3) -> list[str]:
    if not report_path.exists():
        return []
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "## rules":
            start = i + 1
            break
    if start < 0:
        return []
    out: list[str] = []
    for line in lines[start:]:
        t = line.strip()
        if not t:
            if out:
                break
            continue
        if t.startswith("## "):
            break
        if t.startswith("- "):
            out.append(t)
        if len(out) >= max_lines:
            break
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write CI job summary for replay drift artifacts.")
    p.add_argument("--artifacts-dir", default="outputs/ci_artifacts")
    p.add_argument("--tests-status", default=os.getenv("GW_CI_TESTS_STATUS", "UNKNOWN"))
    p.add_argument("--strict-replay-status", default=os.getenv("GW_CI_STRICT_REPLAY_STATUS", "SKIPPED"))
    p.add_argument("--strict-replay-enabled", default=os.getenv("GW_CI_STRICT_REPLAY", "0"))
    return p


def main() -> int:
    args = _build_parser().parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()

    t46_result = _read_json(artifacts_dir / "T46" / "drift_gate_result.json") or {}
    t46_status = str(t46_result.get("status", "MISSING") or "MISSING").upper()

    strict_result = _read_json(artifacts_dir / "STRICT_REPLAY" / "drift_gate_result.json") or {}
    strict_status = str(strict_result.get("status", "MISSING") or "MISSING").upper()

    strict_enabled = str(args.strict_replay_enabled or "0").strip() == "1"
    strict_step_status = str(args.strict_replay_status or "SKIPPED").strip().upper()
    tests_status = str(args.tests_status or "UNKNOWN").strip().upper()

    strict_report = artifacts_dir / "STRICT_REPLAY" / "drift_gate_report.md"
    strict_rules_lines = _fmt_rules_from_report(strict_report, max_lines=4)

    lines: list[str] = []
    lines.append("# Quant/Replay CI Summary")
    lines.append("")
    lines.append(f"- {_status_emoji(tests_status)} run_all_tests: **{tests_status}**")
    lines.append(
        f"- {_status_emoji(strict_step_status)} STRICT_REPLAY step: **{strict_step_status}** "
        f"(enabled={strict_enabled})"
    )
    lines.append(f"- artifacts: `{artifacts_dir}`")
    lines.append(f"- generated_utc: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Drift Gate Snapshot")
    lines.append(f"- T46 drift status: **{t46_status}**")
    lines.append(f"- STRICT_REPLAY drift status: **{strict_status}**")
    lines.append("")
    if strict_rules_lines:
        lines.append("## Strict Replay Rules (excerpt)")
        for row in strict_rules_lines:
            lines.append(row)
        lines.append("")

    summary_text = "\n".join(lines) + "\n"
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(summary_text)
    else:
        print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
