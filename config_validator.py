"""Lightweight configuration validator for paper_config.json.

No external dependencies — uses only the Python standard library.

Usage:
    from config_validator import validate_config, ConfigValidationError

    cfg = json.load(open("paper_config.json"))
    try:
        validate_config(cfg)
    except ConfigValidationError as e:
        sys.exit(f"Config error: {e}")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when paper_config.json fails validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        lines = "\n  ".join(errors)
        super().__init__(f"paper_config.json validation failed ({len(errors)} error(s)):\n  {lines}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path(parts: List[str]) -> str:
    return ".".join(str(p) for p in parts)


def _check_type(errors: List[str], value: Any, expected: type, path: str) -> bool:
    """Return True if value is an instance of expected; append error otherwise."""
    if not isinstance(value, expected):
        errors.append(
            f"{path}: expected {expected.__name__}, got {type(value).__name__} ({value!r})"
        )
        return False
    return True


def _check_range(
    errors: List[str],
    value: Any,
    path: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    exclusive_min: bool = False,
) -> None:
    if not isinstance(value, (int, float)):
        return
    if min_val is not None:
        if exclusive_min and value <= min_val:
            errors.append(f"{path}: must be > {min_val}, got {value}")
        elif not exclusive_min and value < min_val:
            errors.append(f"{path}: must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        errors.append(f"{path}: must be <= {max_val}, got {value}")


def _require_key(errors: List[str], obj: dict, key: str, parent_path: str) -> bool:
    if key not in obj:
        errors.append(f"{parent_path}.{key}: required field is missing")
        return False
    return True


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def _validate_root(cfg: dict, errors: List[str]) -> None:
    """Validate top-level scalar fields."""
    if "paper_mode" in cfg:
        _check_type(errors, cfg["paper_mode"], bool, "paper_mode")

    if _require_key(errors, cfg, "initial_cash_usd", "<root>"):
        if _check_type(errors, cfg["initial_cash_usd"], (int, float), "initial_cash_usd"):
            _check_range(errors, cfg["initial_cash_usd"], "initial_cash_usd", min_val=0, exclusive_min=True)

    if _require_key(errors, cfg, "rebalance_minutes", "<root>"):
        if _check_type(errors, cfg["rebalance_minutes"], (int, float), "rebalance_minutes"):
            _check_range(errors, cfg["rebalance_minutes"], "rebalance_minutes", min_val=1)

    if "duration_hours" in cfg:
        if _check_type(errors, cfg["duration_hours"], (int, float), "duration_hours"):
            _check_range(errors, cfg["duration_hours"], "duration_hours", min_val=0, exclusive_min=True)


def _validate_universe(cfg: dict, errors: List[str]) -> None:
    """Validate universe array: must be non-empty; each item needs ticker, name, asset_type."""
    if "universe" not in cfg:
        return
    uni = cfg["universe"]
    if not _check_type(errors, uni, list, "universe"):
        return
    if len(uni) == 0:
        errors.append("universe: must contain at least one asset")
        return
    for i, item in enumerate(uni):
        p = f"universe[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{p}: expected object, got {type(item).__name__}")
            continue
        for field in ("ticker", "name", "asset_type"):
            if field not in item:
                errors.append(f"{p}.{field}: required field is missing")
            elif not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"{p}.{field}: must be a non-empty string")


def _validate_execution(cfg: dict, errors: List[str]) -> None:
    """Validate execution section numeric constraints."""
    if "execution" not in cfg:
        return
    ex = cfg["execution"]
    if not _check_type(errors, ex, dict, "execution"):
        return

    _num_range = [
        ("weight_threshold",               0.0,  1.0,  False),
        ("max_turnover_pct_per_rebalance",  0.0,  1.0,  False),
        ("max_portfolio_volatility",        0.0,  5.0,  False),
        ("max_herfindahl_index",            0.0,  1.0,  False),
        ("portfolio_vol_min_coverage",      0.0,  1.0,  False),
        ("correlation_threshold",           0.0,  1.0,  False),
        ("stale_price_skip_minutes",        1.0,  None, False),
        ("signal_refresh_minutes",          1.0,  None, False),
        ("macro_refresh_minutes",           1.0,  None, False),
        ("min_trade_notional_usd",          0.0,  None, False),
    ]
    for field, lo, hi, excl in _num_range:
        if field not in ex:
            continue
        path = f"execution.{field}"
        if _check_type(errors, ex[field], (int, float), path):
            _check_range(errors, ex[field], path, min_val=lo, max_val=hi, exclusive_min=excl)


def _validate_risk_model(cfg: dict, errors: List[str]) -> None:
    """Validate risk_model section."""
    if "risk_model" not in cfg:
        return
    rm = cfg["risk_model"]
    if not _check_type(errors, rm, dict, "risk_model"):
        return

    for field, lo, hi in [
        ("rc_limit",             0.0, 1.0),
        ("returns_lookback_days", 5.0, None),
    ]:
        if field not in rm:
            continue
        path = f"risk_model.{field}"
        if _check_type(errors, rm[field], (int, float), path):
            _check_range(errors, rm[field], path, min_val=lo, max_val=hi)


def _validate_cost_model(cfg: dict, errors: List[str]) -> None:
    """Validate cost_model section."""
    if "cost_model" not in cfg:
        return
    cm = cfg["cost_model"]
    if not _check_type(errors, cm, dict, "cost_model"):
        return

    for field in ("slippage_bps", "fee_per_trade_usd", "fee_bps", "min_fee_usd"):
        if field not in cm:
            continue
        path = f"cost_model.{field}"
        if _check_type(errors, cm[field], (int, float), path):
            _check_range(errors, cm[field], path, min_val=0.0)


def _validate_reporting(cfg: dict, errors: List[str]) -> None:
    """Validate reporting section: required output paths must be strings."""
    if "reporting" not in cfg:
        return
    rp = cfg["reporting"]
    if not _check_type(errors, rp, dict, "reporting"):
        return

    # market_tz must be a non-empty string if present
    if "market_tz" in rp:
        if not isinstance(rp["market_tz"], str) or not rp["market_tz"].strip():
            errors.append("reporting.market_tz: must be a non-empty timezone string (e.g. 'America/New_York')")

    # time strings must match HH:MM format if present
    for field in ("market_open_time_et", "market_close_time_et"):
        if field not in rp:
            continue
        val = rp[field]
        if not isinstance(val, str):
            errors.append(f"reporting.{field}: must be a string in HH:MM format, got {type(val).__name__}")
            continue
        import re
        if not re.match(r"^\d{2}:\d{2}(:\d{2})?$", val.strip()):
            errors.append(f"reporting.{field}: must be in HH:MM or HH:MM:SS format, got {val!r}")


def _validate_risk_profiles(cfg: dict, errors: List[str]) -> None:
    """Validate risk_profiles section: each profile must have vol_target."""
    if "risk_profiles" not in cfg:
        return
    rps = cfg["risk_profiles"]
    if not _check_type(errors, rps, dict, "risk_profiles"):
        return

    for name, profile in rps.items():
        p = f"risk_profiles.{name}"
        if not isinstance(profile, dict):
            errors.append(f"{p}: expected object")
            continue
        if "vol_target" in profile:
            if _check_type(errors, profile["vol_target"], (int, float), f"{p}.vol_target"):
                _check_range(errors, profile["vol_target"], f"{p}.vol_target", min_val=0.0, max_val=5.0)
        if "max_turnover_pct" in profile:
            if _check_type(errors, profile["max_turnover_pct"], (int, float), f"{p}.max_turnover_pct"):
                _check_range(errors, profile["max_turnover_pct"], f"{p}.max_turnover_pct", min_val=0.0, max_val=1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_config(cfg: Any) -> None:
    """Validate a parsed paper_config.json dict.

    Args:
        cfg: Parsed JSON object (must be a dict).

    Raises:
        ConfigValidationError: If any validation rule fails. The exception
            message lists all errors found — not just the first one.
    """
    errors: List[str] = []

    if not isinstance(cfg, dict):
        raise ConfigValidationError([f"Config root must be a JSON object, got {type(cfg).__name__}"])

    _validate_root(cfg, errors)
    _validate_universe(cfg, errors)
    _validate_execution(cfg, errors)
    _validate_risk_model(cfg, errors)
    _validate_cost_model(cfg, errors)
    _validate_reporting(cfg, errors)
    _validate_risk_profiles(cfg, errors)

    if errors:
        raise ConfigValidationError(errors)

    logger.info("Config validation passed (%d universe assets)", len(cfg.get("universe", [])))


def validate_config_file(path: str = "paper_config.json") -> Dict[str, Any]:
    """Load and validate a config file.

    Returns:
        The validated config dict.

    Raises:
        ConfigValidationError: On validation failure.
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg
