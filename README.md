# GlobalWatch Paper Trading (V3.2.4)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and paper-execution system.
It combines cross-sectional alpha selection, macro/topic overlays, and execution/risk controls in one reproducible workflow.

From a quant angle, the stack supports momentum + volatility + correlation-aware allocation, regime-aware cash control, and optional industry news overlays.
From a systems angle, it provides checkpoint resume, structured snapshots, deterministic dry-runs, and a Streamlit dashboard for live monitoring and audit.

## Release Highlights (V3.2.4)
1. Engine fingerprint upgraded to `v3.2.4`.
2. Risk gate routing now supports target-cov basis with explicit enable switch and automatic fallback:
   - `execution.enable_target_cov_gate`
   - `execution.target_cov_gate_min_coverage`
   - `execution.target_cov_gate_require_ok`
3. Risk gate observability was expanded across runtime outputs:
   - snapshot fields: `risk_gate_basis`, `risk_gate_cov_coverage_used`
   - telemetry event: `RISK_GATE_DECISION`
   - cycle metrics fields for gate basis and coverage used
4. Portfolio exposure cap is now configurable instead of hardcoded:
   - `execution.portfolio_exposure_cap` (default `0.90`)
   - invalid values auto-fallback with warning and clamp to `[0.0, 1.0]`
5. Daily Quant Pack now includes cycle-level execution blocker attribution:
   - quant artifacts: `execution_blockers/exec_blockers.json`, `no_trade/no_trade.json`
   - daily JSON fields: `quant_pack.execution_blockers`, `quant_pack.no_trade`
   - index sync fields: `exec_blocker_top1_reason`, `exec_blocked_ratio`, `no_trade_flag`
6. Quant alerts now support execution-side operational warnings:
   - `exec_blocker_dominant_cyclelevel`
   - `no_trade_day`
7. CI now installs Python dependencies from `requirements.txt` before running diagnostics.

## Newly Added (Incremental Update)
1. Asset data policy matching is now finer-grained for `.TO` symbols:
   - `asset_data_policy.match_rules` now supports `include_tickers` and `exclude_tickers`.
   - Recommended setup for current production behavior:
     - keep `mode=FORCE_PROXY`
     - restrict proxying to specific symbols (for example `XIU.TO`, `FTS.TO`)
     - leave other Canadian `.TO` symbols as original tickers instead of `NO_PROXY_MAPPING -> DISABLE`
2. Replay has been extended toward L1 determinism:
   - replay bundle includes frozen risk/cov inputs for stronger drift checks.
   - replay/risk-gate diagnostics are more explicit in dry-run outputs.
3. Cost-model observability is integrated end-to-end:
   - `cost_model` config supports fee/slippage controls.
   - execution records and snapshots include cost summaries.
4. Daily report and UI observability were expanded:
   - no-trade/blocker summaries are easier to inspect.
   - risk/coverage diagnostics are surfaced for faster post-mortem.

## Latest Validation (Recent Corpus)
1. The replay comparison pipeline now distinguishes two comparability semantics:
   - baseline drift comparability (`config_metadata_compare`)
   - scenario-aware comparability (`scenario_metadata_compare`)
2. Walk-forward ranking and winner selection now use scenario-aware comparable days as the primary eligibility basis.
3. A focused OPEN-market corpus check was executed (not only `MARKET_CLOSED` snapshots):
   - samples included both tradable OPEN cases and risk-gated OPEN cases
   - scenario reason distribution showed real divergence (`portfolio_cov_rc_limit` vs `RISK_GATE` vs `traded`)
4. In the latest OPEN-focused rerun, global winner remained `baseline_mid`, and window winners also stayed `baseline_mid`.
5. A minimal export layer is available for one-page review output:
   - `walkforward_report.md`
   - `walkforward_report_summary.json`
6. Current limitation: available OPEN/high-signal dates are still limited, so stability conclusions should be re-checked as more replayable OPEN samples are collected.

## Documentation
- Detailed English documentation: `README.en.md`
- 详细中文文档: `README.zh.md`

## Quick Start
```bash
python -u paper_trading.py paper_config.json
```

```bash
streamlit run GlobalWatch_V2.py
```

## CI Dependencies
The repository now includes a root `requirements.txt` used by CI:
- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`

