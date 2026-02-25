from datetime import datetime, timezone

from cooldown_policy import cooldown_policy, next_market_open_time


def _cfg():
    return {
        "success_cooldown_min": 90,
        "enable_jitter": False,
        "jitter_pct": 0.10,
        "failure_backoff": {
            "default": {"base": 5, "cap": 30},
            "portfolio_cov_rc_limit": {"base": 5, "cap": 30},
            "broker_error": {"base": 2, "cap": 60},
            "hard_risk_limit": {"base": 10, "cap": 60},
        },
    }


def test_portfolio_cov_backoff_sequence_caps_at_30():
    cfg = _cfg()
    state = {}
    now = datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc)
    seq = []
    for _ in range(4):
        decision = cooldown_policy(now, "FAIL", "portfolio_cov_rc_limit", state, cfg)
        seq.append(int(round(float(decision["backoff_min"]))))
        state = decision["state"]
        now = decision["next_allowed_ts"]
    assert seq == [5, 10, 20, 30]


def test_success_trade_uses_success_cooldown_and_clears_fail_counts():
    cfg = _cfg()
    now = datetime(2026, 2, 10, 16, 0, tzinfo=timezone.utc)
    d1 = cooldown_policy(now, "FAIL", "portfolio_cov_rc_limit", {}, cfg)
    d2 = cooldown_policy(d1["next_allowed_ts"], "SUCCESS_TRADE", "success_trade", d1["state"], cfg)
    assert abs(float(d2["backoff_min"]) - 90.0) <= 1e-9
    assert d2["state"].get("consecutive_fail_by_reason", {}) == {}


def test_market_closed_uses_next_market_open_time():
    cfg = _cfg()
    now = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)  # 05:00 ET (pre-open)
    decision = cooldown_policy(
        now,
        "SKIP_MARKET_CLOSED",
        "market_closed_gate",
        {},
        cfg,
        tz_market="America/New_York",
        open_time_et="09:30",
    )
    expected = next_market_open_time(now, tz_market="America/New_York", open_time_et="09:30")
    assert decision["next_allowed_ts"] == expected


def test_failure_counts_are_reason_scoped():
    cfg = _cfg()
    state = {}
    now = datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc)

    d_cov_1 = cooldown_policy(now, "FAIL", "portfolio_cov_rc_limit", state, cfg)
    state = d_cov_1["state"]
    d_broker_1 = cooldown_policy(now, "FAIL", "broker_error", state, cfg)
    state = d_broker_1["state"]
    d_cov_2 = cooldown_policy(now, "FAIL", "portfolio_cov_rc_limit", state, cfg)

    assert int(d_broker_1["fail_count"]) == 1
    assert int(d_cov_2["fail_count"]) == 2
    fail_map = d_cov_2["state"].get("consecutive_fail_by_reason", {})
    assert int(fail_map.get("portfolio_cov_rc_limit", 0)) == 2
    assert int(fail_map.get("broker_error", 0)) == 1
