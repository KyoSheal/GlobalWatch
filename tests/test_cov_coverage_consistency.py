from __future__ import annotations

from cov_coverage import compute_cov_coverage


def test_cov_coverage_consistency_with_masked_inputs():
    tickers_order = ["EWC", "XLI", "FTS", "SU.TO"]
    covered_mask = [True, True, True, False]
    target_weights = {
        "EWC": 0.1435,
        "XLI": 0.1113,
        "FTS": 0.1108,
        "SU.TO": 0.1104,
    }

    covered_tickers = {
        ticker for ticker, is_covered in zip(tickers_order, covered_mask) if bool(is_covered)
    }
    cov = compute_cov_coverage(
        target_weights=target_weights,
        covered_tickers=covered_tickers,
        basis="target_weights",
        stage="cov",
    )

    expected_known = 0.1435 + 0.1113 + 0.1108
    expected_missing = 0.1104
    abs_sum = sum(abs(v) for v in target_weights.values())

    assert int(cov.get("covered_count", -1)) == 3
    assert abs(float(cov.get("known_weight", 0.0)) - expected_known) < 1e-9
    assert abs(float(cov.get("missing_weight_total", 0.0)) - expected_missing) < 1e-9
    assert abs((float(cov.get("known_weight", 0.0)) + float(cov.get("missing_weight_total", 0.0))) - abs_sum) < 1e-9
