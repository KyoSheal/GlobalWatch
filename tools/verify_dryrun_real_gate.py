import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cov_coverage import compute_cov_coverage


def main() -> int:
    cov = compute_cov_coverage(
        target_weights={"XIU.TO": 0.1107, "FTS.TO": 0.0932, "SPY": 0.1, "XLP": 0.1},
        covered_tickers={"SPY", "XLP"},
        basis="target_weights",
        stage="cov",
        top_n=20,
        max_list=200,
    )
    missing_weight_total = float(cov.get("missing_weight_total", 0.0) or 0.0)
    missing_count = int(cov.get("missing_count", 0) or 0)
    ok = missing_weight_total > 0.0 and missing_count > 0
    print(
        "[VERIFY_DRYRUN_REAL_GATE] "
        f"known_weight={float(cov.get('known_weight', 0.0) or 0.0):.6f} "
        f"missing_weight_total={missing_weight_total:.6f} missing_count={missing_count}"
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
