from __future__ import annotations

import pandas as pd

from returns_coverage_diag import (
    REASON_CALENDAR_MISMATCH,
    REASON_PRICE_MISSING,
    REASON_SOURCE_UNSUPPORTED,
    REASON_TOO_FEW_POINTS,
    diagnose_returns_coverage,
)


def _cfg():
    return {
        "lookback_days": 60,
        "expected_points": 60,
        "min_obs": 30,
        "drop_threshold": 0.5,
        "period": "6mo",
        "interval": "1d",
    }


def test_returns_coverage_diag_price_missing():
    def provider(**kwargs):
        return None

    out = diagnose_returns_coverage(
        ticker="XIU.TO",
        lookback_cfg=_cfg(),
        price_provider=provider,
        calendar_cfg={"detect_calendar_mismatch": True},
    )
    assert out["ticker"] == "XIU.TO"
    assert out["reason_code"] == REASON_PRICE_MISSING


def test_returns_coverage_diag_too_few_points():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    hist = pd.DataFrame({"Close": [10.0, 10.2, 10.1, 10.4, 10.3]}, index=idx)

    def provider(**kwargs):
        return hist

    out = diagnose_returns_coverage(
        ticker="FTS.TO",
        lookback_cfg=_cfg(),
        price_provider=provider,
        calendar_cfg={"detect_calendar_mismatch": True},
    )
    assert out["reason_code"] == REASON_TOO_FEW_POINTS


def test_returns_coverage_diag_calendar_mismatch_and_source_unsupported():
    weekend_idx = pd.to_datetime(["2026-02-07", "2026-02-08", "2026-02-09", "2026-02-10"])
    hist = pd.DataFrame({"Close": [20.0, 20.1, 20.2, 20.3]}, index=weekend_idx)

    def provider_calendar(**kwargs):
        return hist

    out_calendar = diagnose_returns_coverage(
        ticker="ABC",
        lookback_cfg={**_cfg(), "min_obs": 3, "expected_points": 4},
        price_provider=provider_calendar,
        calendar_cfg={"detect_calendar_mismatch": True},
    )
    assert out_calendar["reason_code"] == REASON_CALENDAR_MISMATCH

    def provider_unsupported(**kwargs):
        raise NotImplementedError("source not enabled")

    out_unsupported = diagnose_returns_coverage(
        ticker="DEF",
        lookback_cfg=_cfg(),
        price_provider=provider_unsupported,
        calendar_cfg={"detect_calendar_mismatch": True},
    )
    assert out_unsupported["reason_code"] == REASON_SOURCE_UNSUPPORTED

