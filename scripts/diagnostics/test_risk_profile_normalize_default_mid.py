#!/usr/bin/env python3
"""T02: normalize risk profile to strict low/mid/high/ultra with default mid."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trading import PaperTradingEngine, RISK_PROFILE_CHOICES, RISK_PROFILE_DEFAULT


def _build_engine_stub() -> PaperTradingEngine:
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine._risk_profile_default_events = []
    engine._risk_profile_default_event_keys = set()
    engine.config = {}
    return engine


def _normalize_runtime_request(value) -> str:
    return str(value or "").strip().lower()


def _is_valid_runtime_requested(value) -> bool:
    normalized = _normalize_runtime_request(value)
    return normalized in set(RISK_PROFILE_CHOICES)


def main() -> int:
    failures: list[str] = []
    engine = _build_engine_stub()

    # A) empty / None -> mid
    for raw in ("", None):
        cfg = {"execution": {"risk_profile": raw}}
        try:
            normalized = engine._normalize_execution_risk_profile(source="unit_t02", config_obj=cfg)
        except Exception as exc:
            failures.append(f"A: raised on raw={raw!r}: {exc}")
            continue
        final = str(cfg.get("execution", {}).get("risk_profile", "")).strip().lower()
        if normalized != "mid" or final != "mid":
            failures.append(f"A: raw={raw!r} normalized={normalized!r} final={final!r} expected 'mid'")

    # B) manual/abc -> mid without exception
    for raw in ("manual", "abc"):
        cfg = {"execution": {"risk_profile": raw}}
        try:
            normalized = engine._normalize_execution_risk_profile(source="unit_t02", config_obj=cfg)
        except Exception as exc:
            failures.append(f"B: raised on raw={raw!r}: {exc}")
            continue
        final = str(cfg.get("execution", {}).get("risk_profile", "")).strip().lower()
        if normalized != "mid" or final != "mid":
            failures.append(f"B: raw={raw!r} normalized={normalized!r} final={final!r} expected 'mid'")

    # C) trim + lowercase: " High " -> "high"
    cfg_c = {"execution": {"risk_profile": " High "}}
    try:
        normalized_c = engine._normalize_execution_risk_profile(source="unit_t02", config_obj=cfg_c)
    except Exception as exc:
        failures.append(f"C: raised on raw=' High ': {exc}")
        normalized_c = ""
    final_c = str(cfg_c.get("execution", {}).get("risk_profile", "")).strip().lower()
    if normalized_c != "high" or final_c != "high":
        failures.append(f"C: normalized={normalized_c!r} final={final_c!r} expected 'high'")

    # D) runtime requested strict legal set
    expected = {"low", "mid", "high", "ultra"}
    actual = set(RISK_PROFILE_CHOICES)
    if actual != expected:
        failures.append(f"D: RISK_PROFILE_CHOICES={tuple(RISK_PROFILE_CHOICES)!r} expected {sorted(expected)!r}")

    valid_cases = ["low", "mid", "high", "ultra", " High "]
    invalid_cases = ["manual", "abc", "", None, "  "]
    for v in valid_cases:
        if not _is_valid_runtime_requested(v):
            failures.append(f"D: valid runtime request flagged invalid: {v!r}")
    for v in invalid_cases:
        if _is_valid_runtime_requested(v):
            failures.append(f"D: invalid runtime request flagged valid: {v!r}")

    if str(RISK_PROFILE_DEFAULT).strip().lower() != "mid":
        failures.append(f"default profile is {RISK_PROFILE_DEFAULT!r}, expected 'mid'")

    if failures:
        print("[FAIL] risk_profile_normalize_default_mid")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[PASS] risk_profile_normalize_default_mid")
    print(f"[INFO] allowed_profiles={sorted(set(RISK_PROFILE_CHOICES))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

