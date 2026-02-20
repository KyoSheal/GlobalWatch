#!/usr/bin/env python3
"""T00: compile key modules with py_compile."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path


FILES_TO_COMPILE = [
    "paper_trading.py",
    "GlobalWatch_V2.py",
    "daily_reporter.py",
    "telemetry.py",
    "atomic_io.py",
    "price_service.py",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    failures = []

    for rel in FILES_TO_COMPILE:
        target = repo_root / rel
        if not target.exists():
            continue
        try:
            py_compile.compile(str(target), doraise=True)
            print(f"[OK] {rel}")
        except Exception as exc:
            failures.append((rel, exc))
            print(f"[FAIL] {rel}: {exc}")

    if failures:
        print("[FAIL] compile_all")
        return 2

    print("[PASS] compile_all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

