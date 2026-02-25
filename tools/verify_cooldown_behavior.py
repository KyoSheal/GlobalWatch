from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def main() -> int:
    cfg = _cfg()
    state = {}
    now = datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc)

    cov_seq = []
    for _ in range(4):
        decision = cooldown_policy(now, "FAIL", "portfolio_cov_rc_limit", state, cfg)
        cov_seq.append(int(round(float(decision["backoff_min"]))))
        state = decision["state"]
        now = decision["next_allowed_ts"]

    success_decision = cooldown_policy(now, "SUCCESS_TRADE", "success_trade", state, cfg)
    now_preopen = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    market_closed_decision = cooldown_policy(
        now_preopen,
        "SKIP_MARKET_CLOSED",
        "market_closed_gate",
        success_decision["state"],
        cfg,
        tz_market="America/New_York",
        open_time_et="09:30",
    )
    expected_open = next_market_open_time(now_preopen, tz_market="America/New_York", open_time_et="09:30")

    print(f"portfolio_cov_rc_limit: {','.join(str(x) for x in cov_seq)}")
    print(f"success cooldown: {int(round(float(success_decision['backoff_min'])))}")
    print(f"market_closed: {market_closed_decision['next_allowed_ts'].isoformat()}")

    ok = True
    if cov_seq != [5, 10, 20, 30]:
        print("FAIL: unexpected portfolio_cov_rc_limit backoff sequence")
        ok = False
    if abs(float(success_decision["backoff_min"]) - 90.0) > 1e-9:
        print("FAIL: unexpected success cooldown")
        ok = False
    if market_closed_decision["next_allowed_ts"] != expected_open:
        print("FAIL: market_closed next_allowed_ts mismatch")
        ok = False

    if ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
