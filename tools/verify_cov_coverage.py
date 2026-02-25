from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cov_coverage import compute_cov_coverage


def _print_cov_logs(cov):
    print(
        "[COV_COVERAGE] "
        f"basis={cov.get('basis')} stage={cov.get('stage')} "
        f"known_weight={float(cov.get('known_weight', 0.0) or 0.0):.4f} "
        f"missing_weight_total={float(cov.get('missing_weight_total', 0.0) or 0.0):.4f} "
        f"missing_count={int(cov.get('missing_count', 0) or 0)} "
        f"covered_count={int(cov.get('covered_count', 0) or 0)}"
    )
    top = cov.get("top_missing", [])
    parts = []
    if isinstance(top, list):
        for idx, row in enumerate(top[:20], start=1):
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker", "")).upper().strip()
            if not t:
                continue
            w = float(row.get("w", 0.0) or 0.0)
            parts.append(f"{idx}) {t} {w:.4f}")
    if parts:
        print("[COV_MISSING_TOP] " + " ".join(parts))
    else:
        print("[COV_MISSING_TOP] none")


def main() -> int:
    target = {
        "XIU.TO": 0.1107,
        "FTS.TO": 0.0932,
        "XOM": 0.20,
        "XLP": 0.15,
        "CASH": 0.4461,
    }
    covered = {"XOM", "XLP"}
    cov = compute_cov_coverage(
        target_weights=target,
        covered_tickers=covered,
        basis="target_weights",
        stage="returns",
        top_n=20,
        max_list=200,
    )
    _print_cov_logs(cov)

    ok = True
    if int(cov.get("schema_version", 0) or 0) != 1:
        ok = False
    if str(cov.get("basis", "")) != "target_weights":
        ok = False
    if str(cov.get("stage", "")) != "returns":
        ok = False
    if abs(float(cov.get("missing_weight_total", 0.0) or 0.0) - (0.1107 + 0.0932)) > 1e-9:
        ok = False
    top = cov.get("top_missing", [])
    if not isinstance(top, list) or not top or str(top[0].get("ticker", "")) != "XIU.TO":
        ok = False

    if ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

