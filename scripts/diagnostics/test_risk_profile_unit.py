#!/usr/bin/env python3
"""T01: unit checks for risk profile merge/whitelist/hash."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trading import (
    PaperTradingEngine,
    RISK_PROFILE_ALLOWED_KEYS,
    RISK_PROFILE_CHOICES,
    RISK_PROFILE_TEMPLATE_VERSION,
)


def _flatten_whitelist_paths() -> set[str]:
    paths: set[str] = set()
    for section, keys in (RISK_PROFILE_ALLOWED_KEYS or {}).items():
        for key in (keys or []):
            paths.add(f"{section}.{key}")
    return paths


def _build_engine_stub() -> PaperTradingEngine:
    engine = PaperTradingEngine.__new__(PaperTradingEngine)
    engine._risk_profile_default_events = []
    engine._risk_profile_default_event_keys = set()
    engine.config = {}
    return engine


def main() -> int:
    config_path = ROOT / "paper_config.json"
    if not config_path.exists():
        print(f"[FAIL] missing config: {config_path}")
        return 1

    engine = _build_engine_stub()
    base_config = engine.load_config(str(config_path))
    allowed_paths = _flatten_whitelist_paths()
    failures: list[str] = []

    # 1) per-profile checks
    for profile in ("low", "mid", "high", "ultra"):
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("execution", {})["risk_profile"] = profile
        merged, diag = engine.apply_risk_profile(cfg)
        if not isinstance(diag, dict):
            failures.append(f"{profile}: diag is not dict")
            continue

        applied_profile = str(diag.get("profile", "") or "").strip().lower()
        status = str(diag.get("status", "") or "").strip().lower()
        if status != "applied":
            failures.append(f"{profile}: status={status!r}, expected 'applied'")
        if applied_profile != profile:
            failures.append(f"{profile}: diag.profile={applied_profile!r}, expected {profile!r}")

        overrides = list(diag.get("overrides_applied", []) or [])
        unknown_paths = [p for p in overrides if p not in allowed_paths]
        if unknown_paths:
            failures.append(f"{profile}: overrides outside whitelist: {unknown_paths}")

        important = engine._build_risk_profile_important_values(merged)
        required_numeric = [
            "target_vol_annual",
            "min_cash_floor",
            "portfolio_exposure_cap",
            "max_single_weight",
            "rc_limit",
            "overlay_alpha",
            "overlay_max_abs_delta",
            "overlay_min_confidence",
        ]
        for key in required_numeric:
            val = important.get(key)
            if not isinstance(val, (int, float)):
                failures.append(f"{profile}: important_values[{key}] invalid type={type(val).__name__}")

        # bool sanity from merged config
        use_cov = merged.get("risk_model", {}).get("use_cov_vol_for_gate")
        enable_target_cov = merged.get("execution", {}).get("enable_target_cov_gate")
        if not isinstance(use_cov, bool):
            failures.append(f"{profile}: risk_model.use_cov_vol_for_gate not bool")
        if not isinstance(enable_target_cov, bool):
            failures.append(f"{profile}: execution.enable_target_cov_gate not bool")

        # 2) hash stability
        h1 = engine._compute_risk_profile_overrides_hash(diag, important)
        h2 = engine._compute_risk_profile_overrides_hash(diag, important)
        if h1 != h2:
            failures.append(f"{profile}: hash unstable ({h1} vs {h2})")

        # 3) template version checks
        tv = diag.get("template_version")
        if not isinstance(tv, int):
            failures.append(f"{profile}: template_version type={type(tv).__name__}, expected int")
        elif tv != 1:
            failures.append(f"{profile}: template_version={tv}, expected 1")

    if int(RISK_PROFILE_TEMPLATE_VERSION) != 1:
        failures.append(f"RISK_PROFILE_TEMPLATE_VERSION={RISK_PROFILE_TEMPLATE_VERSION}, expected 1")
    if set(RISK_PROFILE_CHOICES) != {"low", "mid", "high", "ultra"}:
        failures.append(f"RISK_PROFILE_CHOICES={tuple(RISK_PROFILE_CHOICES)} unexpected")

    if failures:
        print("[FAIL] risk_profile_unit")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[PASS] risk_profile_unit")
    print(f"[INFO] whitelist_paths={len(allowed_paths)} profiles_tested=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
