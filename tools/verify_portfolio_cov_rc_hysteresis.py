from __future__ import annotations

import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_trading import resolve_portfolio_cov_rc_hysteresis_decision


def _check(name: str, condition: bool) -> bool:
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] {name}")
    return bool(condition)


def main() -> int:
    ok = True

    d_abort = resolve_portfolio_cov_rc_hysteresis_decision(0.31, 0.30, 0.03, "ABORT")
    d_allow = resolve_portfolio_cov_rc_hysteresis_decision(0.31, 0.30, 0.03, "ALLOW")
    d_fallback = resolve_portfolio_cov_rc_hysteresis_decision(0.31, 0.30, 0.03, None)

    ok &= _check("sticky keeps previous ABORT", d_abort.get("sticky_zone") and d_abort.get("final_gate_decision") == "ABORT")
    ok &= _check("sticky keeps previous ALLOW", d_allow.get("sticky_zone") and d_allow.get("final_gate_decision") == "ALLOW")
    ok &= _check(
        "sticky fallback without previous uses old logic",
        d_fallback.get("sticky_zone") and d_fallback.get("fallback_used") and d_fallback.get("final_gate_decision") == "ABORT",
    )

    print("[RC_HYST_CASE] " + json.dumps(d_allow, ensure_ascii=False))
    snapshot_fragment = {"risk_gate_decision": d_allow}
    print("[RC_HYST_SNAPSHOT] " + json.dumps(snapshot_fragment, ensure_ascii=False))

    if ok:
        print("VERIFY_SUMMARY pass=3 fail=0")
        return 0
    print("VERIFY_SUMMARY pass<3 fail>0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
