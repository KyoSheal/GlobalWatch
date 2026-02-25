"""Covariance coverage diagnostics helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:  # NaN
            return float(default)
        return out
    except Exception:
        return float(default)


def _normalize_weights(target_weights: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(target_weights, dict):
        return out
    for raw_ticker, raw_weight in target_weights.items():
        ticker = str(raw_ticker).upper().strip()
        if not ticker or ticker == "CASH":
            continue
        w = _as_float(raw_weight, 0.0)
        if abs(w) <= 1e-12:
            continue
        out[ticker] = float(w)
    return out


def _normalize_ticker_set(tickers: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for raw in tickers or []:
        t = str(raw).upper().strip()
        if not t or t == "CASH":
            continue
        out.add(t)
    return out


def default_cov_coverage(
    *,
    basis: str = "target_weights",
    stage: str = "cov",
    known_weight: float = 0.0,
    missing_weight_total: float = 0.0,
    covered_count: int = 0,
    missing_count: int = 0,
    missing_tickers: Iterable[str] | None = None,
    top_missing: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "basis": str(basis or "target_weights"),
        "stage": str(stage or "cov"),
        "known_weight": float(known_weight),
        "missing_weight_total": float(missing_weight_total),
        "covered_count": int(covered_count),
        "missing_count": int(missing_count),
        "missing_tickers": list(missing_tickers or []),
        "top_missing": list(top_missing or []),
    }


def compute_cov_coverage(
    target_weights: Dict[str, float],
    covered_tickers: Set[str],
    basis: str,
    stage: str,
    top_n: int = 20,
    max_list: int = 200,
) -> Dict[str, Any]:
    """Build deterministic covariance coverage dump for diagnostics/output."""
    w_map = _normalize_weights(target_weights)
    covered = _normalize_ticker_set(covered_tickers or set())

    if top_n <= 0:
        top_n = 1
    if max_list <= 0:
        max_list = 1

    denom = sum(abs(v) for v in w_map.values())
    if denom <= 1e-12:
        return default_cov_coverage(
            basis=basis or "target_weights_abs",
            stage=stage or "cov",
            known_weight=0.0,
            missing_weight_total=0.0,
            covered_count=0,
            missing_count=0,
            missing_tickers=[],
            top_missing=[],
        )

    missing_rows = []
    covered_count = 0
    for ticker, w in w_map.items():
        if ticker in covered:
            covered_count += 1
            continue
        missing_rows.append({"ticker": ticker, "w": float(w)})

    missing_rows.sort(key=lambda x: abs(float(x.get("w", 0.0))), reverse=True)
    missing_weight_total_raw = sum(abs(float(item.get("w", 0.0))) for item in missing_rows)
    missing_weight_total = float(missing_weight_total_raw)
    known_weight = max(0.0, 1.0 - float(missing_weight_total_raw / denom))

    top_missing = [
        {"ticker": str(item.get("ticker", "")), "w": float(item.get("w", 0.0))}
        for item in missing_rows[: int(top_n)]
    ]
    missing_tickers = [str(item.get("ticker", "")) for item in missing_rows[: int(max_list)]]

    return default_cov_coverage(
        basis=basis or "target_weights_abs",
        stage=stage or "cov",
        known_weight=float(known_weight),
        missing_weight_total=float(missing_weight_total),
        covered_count=int(covered_count),
        missing_count=int(len(missing_rows)),
        missing_tickers=missing_tickers,
        top_missing=top_missing,
    )
