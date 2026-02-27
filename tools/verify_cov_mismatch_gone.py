from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


PAT_COV_INPUTS = re.compile(
    r"\[COV_INPUTS\].*abs_sum=(?P<abs_sum>[-+]?\d*\.?\d+).*covered_count=(?P<covered_count>\d+)"
)
PAT_COV_COVERAGE = re.compile(
    r"\[COV_COVERAGE\].*known_weight=(?P<known>[-+]?\d*\.?\d+).*missing_weight_total=(?P<missing>[-+]?\d*\.?\d+).*covered_count=(?P<covered_count>\d+)"
)


def _parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    has_gate_dump_mismatch = any("gate_dump_mismatch" in line for line in lines)
    cov_inputs = []
    cov_coverage = []
    for line in lines:
        m_inputs = PAT_COV_INPUTS.search(line)
        if m_inputs:
            cov_inputs.append(
                {
                    "line": line,
                    "abs_sum": float(m_inputs.group("abs_sum")),
                    "covered_count": int(m_inputs.group("covered_count")),
                }
            )
        m_cov = PAT_COV_COVERAGE.search(line)
        if m_cov:
            cov_coverage.append(
                {
                    "line": line,
                    "known": float(m_cov.group("known")),
                    "missing": float(m_cov.group("missing")),
                    "covered_count": int(m_cov.group("covered_count")),
                }
            )

    return has_gate_dump_mismatch, cov_inputs, cov_coverage


def main() -> int:
    outdir = Path("outputs/gw_cov_mismatch_verify")
    outdir.mkdir(parents=True, exist_ok=True)
    run_log = outdir / "run.log"

    env = os.environ.copy()
    env["GW_ASSET_POLICY_MODE"] = "ALLOW_ORIGINAL"
    env["GW_PROXY_SCOPE"] = "off"

    cmd = [
        sys.executable,
        "paper_trading.py",
        "--debug-system-s1-5",
        "--dryrun-real-risk-gate",
        "--debug-outdir",
        str(outdir),
    ]

    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)

    if proc.returncode != 0:
        print(f"FAIL: command failed rc={proc.returncode}")
        print(f"log={run_log}")
        return 1

    has_mismatch, cov_inputs, cov_coverage = _parse_log(run_log)
    if has_mismatch:
        print("FAIL: found gate_dump_mismatch in run log")
        print(f"log={run_log}")
        return 1

    if not cov_inputs or not cov_coverage:
        print("FAIL: missing COV_INPUTS or COV_COVERAGE lines")
        print(f"log={run_log}")
        return 1

    pair_count = min(len(cov_inputs), len(cov_coverage))
    if pair_count <= 0:
        print("FAIL: no comparable COV_INPUTS/COV_COVERAGE pairs")
        print(f"log={run_log}")
        return 1

    for idx in range(pair_count):
        inp = cov_inputs[idx]
        cov = cov_coverage[idx]
        if int(inp["covered_count"]) != int(cov["covered_count"]):
            print(
                f"FAIL: covered_count mismatch at pair {idx}: "
                f"inputs={inp['covered_count']} coverage={cov['covered_count']}"
            )
            return 1
        if abs(float(inp["abs_sum"]) - (float(cov["known"]) + float(cov["missing"]))) > 1e-6:
            print(
                f"FAIL: abs_sum mismatch at pair {idx}: "
                f"abs_sum={inp['abs_sum']:.6f} known+missing={(cov['known'] + cov['missing']):.6f}"
            )
            return 1

    print("PASS: cov mismatch gone")
    print(f"log={run_log}")
    print(f"pairs_checked={pair_count}")
    print(
        "sample="
        f"covered_count={cov_coverage[0]['covered_count']} "
        f"known_weight={cov_coverage[0]['known']:.6f} "
        f"missing_weight_total={cov_coverage[0]['missing']:.6f} "
        f"abs_sum={cov_inputs[0]['abs_sum']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
