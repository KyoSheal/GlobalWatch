from cov_coverage import compute_cov_coverage


def test_cov_coverage_dump_contains_missing_and_top_sorted():
    target_weights = {
        "XIU.TO": 0.1107,
        "FTS.TO": 0.0932,
        "XOM": 0.20,
        "XLP": 0.15,
        "CASH": 0.4461,
    }
    covered = {"XOM", "XLP"}
    cov = compute_cov_coverage(
        target_weights=target_weights,
        covered_tickers=covered,
        basis="target_weights",
        stage="cov",
        top_n=20,
        max_list=200,
    )

    assert isinstance(cov, dict)
    assert int(cov.get("schema_version", 0)) == 1
    assert str(cov.get("basis", "")) == "target_weights"
    assert str(cov.get("stage", "")) == "cov"

    missing = set(str(x) for x in cov.get("missing_tickers", []))
    assert "XIU.TO" in missing
    assert "FTS.TO" in missing

    expected_missing_total = 0.1107 + 0.0932
    assert abs(float(cov.get("missing_weight_total", 0.0)) - expected_missing_total) <= 1e-9

    top_missing = cov.get("top_missing", [])
    assert isinstance(top_missing, list)
    assert len(top_missing) >= 2
    assert str(top_missing[0].get("ticker", "")) == "XIU.TO"
    assert float(top_missing[0].get("w", 0.0)) >= float(top_missing[1].get("w", 0.0))

