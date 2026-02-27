import argparse
import json
import sys
from pathlib import Path


ALLOWED_PROFILES = {"low", "mid", "high", "ultra"}
DEFAULT_EXECUTION_WHITELIST = {"max_turnover_pct_per_rebalance"}


def _load_execution_whitelist():
    try:
        import paper_trading  # type: ignore

        keys = getattr(paper_trading, "RISK_PROFILE_ALLOWED_KEYS", {}).get("execution")
        if isinstance(keys, (set, list, tuple)):
            return set(keys)
    except Exception:
        pass
    return set(DEFAULT_EXECUTION_WHITELIST)


def _normalize_profile(raw):
    profile = str(raw or "").strip().lower()
    return profile if profile in ALLOWED_PROFILES else "mid"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_effective_execution(cfg):
    execution = cfg.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    merged = dict(execution)
    profile = _normalize_profile(execution.get("risk_profile"))

    risk_profiles = cfg.get("risk_profiles", {})
    if not isinstance(risk_profiles, dict):
        risk_profiles = {}
    rp = risk_profiles.get(profile, {})
    rp_exec = rp.get("execution", {}) if isinstance(rp, dict) else {}
    if not isinstance(rp_exec, dict):
        rp_exec = {}

    execution_whitelist = _load_execution_whitelist()
    for key in execution_whitelist:
        if key in rp_exec:
            merged[key] = rp_exec[key]

    return profile, merged


def _resolve_effective_values(cfg):
    profile, execution = _resolve_effective_execution(cfg)

    turnover = execution.get("max_turnover_pct_per_rebalance")
    cooldown_policy = execution.get("cooldown_policy", {})
    if not isinstance(cooldown_policy, dict):
        cooldown_policy = {}
    success_cd = cooldown_policy.get(
        "success_cooldown_min",
        execution.get("rebalance_cooldown_minutes", 90),
    )
    return profile, turnover, success_cd


def main():
    parser = argparse.ArgumentParser(description="Verify effective turnover/cooldown config.")
    parser.add_argument(
        "--config",
        default="paper_config.json",
        help="Path to config JSON (default: paper_config.json)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"FAIL: config not found: {cfg_path}")
        return 1

    cfg = _load_json(cfg_path)
    profile, turnover, success_cd = _resolve_effective_values(cfg)

    print(f"CONFIG_PATH={cfg_path.resolve()}")
    print(f"EFFECTIVE_PROFILE={profile}")
    print(f"EFFECTIVE_TURNOVER_CAP={turnover}")
    print(f"EFFECTIVE_SUCCESS_COOLDOWN_MIN={success_cd}")

    try:
        ok_turnover = abs(float(turnover) - 0.40) < 1e-12
        ok_success = abs(float(success_cd) - 60.0) < 1e-12
    except Exception:
        ok_turnover = False
        ok_success = False

    if ok_turnover and ok_success:
        print("PASS: effective limits are turnover=0.40 and success_cooldown_min=60")
        return 0

    print("FAIL: effective limits mismatch, expected turnover=0.40 and success_cooldown_min=60")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
