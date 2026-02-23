# GlobalWatch Paper Trading (V3.2.4)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## Overview
GlobalWatch Paper Trading is a local simulation framework for quantitative strategy research, execution validation, and risk-control testing.
It does not connect to a real broker. The focus is transparent behavior, stable iteration, and auditable outputs.

The stack combines:
- cross-sectional ranking and portfolio construction
- market-regime and macro/topic overlays
- execution safeguards (stale policy, turnover budget, planner)
- structured runtime outputs consumed by Streamlit monitoring pages

## What's New in V3.2.4
This release updates the engine fingerprint to `v3.2.4` and adds the latest execution/risk-control improvements.

1. Target-cov risk gate routing is now configurable and fallback-safe:
   - `execution.enable_target_cov_gate`
   - `execution.target_cov_gate_min_coverage`
   - `execution.target_cov_gate_require_ok`
2. Risk gate observability now includes:
   - snapshot fields: `risk_gate_basis`, `risk_gate_cov_coverage_used`
   - telemetry event: `RISK_GATE_DECISION`
   - cycle-metrics fields for basis and coverage used
3. Portfolio exposure cap is configurable instead of fixed:
   - `execution.portfolio_exposure_cap` (default `0.90`)
   - invalid inputs auto-fallback and clamp to `[0.0, 1.0]`
4. Default behavior remains backward-compatible:
   - target-cov gate stays off unless enabled
   - exposure cap remains 90% unless changed
5. Daily Quant Pack now writes execution/no-trade diagnostics:
   - `quant_packs/<date>/execution_blockers/exec_blockers.json`
   - `quant_packs/<date>/no_trade/no_trade.json`
   - embedded into flat daily JSON as `quant_pack.execution_blockers` and `quant_pack.no_trade`
6. Daily index synchronization now includes execution blocker summaries:
   - `exec_blocker_top1_reason`, `exec_blocker_top1_ratio`, `exec_blocked_ratio`
   - `no_trade_flag`, `no_trade_primary_reason`, merged warnings count
7. Alert rules were expanded with:
   - `exec_blocker_dominant_cyclelevel`
   - `no_trade_day`
8. CI installs dependencies from root `requirements.txt` before diagnostics.

## Functional Capabilities
### Quant Layer
- cross-sectional score ranking
- volatility-aware allocation and clipping
- correlation filtering
- regime-aware cash target and max-weight controls
- optional macro/topic/industry overlays

### Execution and Risk Layer
- stale quote policy for BUY/SELL paths
- stale-ratio abort guard during tradable sessions
- turnover budgeting with planner diagnostics
- exit-signal hooks and forced de-risk path
- optional cost estimation, ADV checks, and planner scoring controls

### System Layer
- checkpoint resume and fresh-start handling
- deterministic config-driven runtime
- cycle snapshots and trade history persistence
- dry-run diagnostics and calibration utilities
- Streamlit monitoring integration

## Start
### Paper engine
```bash
python -u paper_trading.py paper_config.json
```

### GlobalWatch UI
```bash
streamlit run GlobalWatch_V2.py
```

### Windows launchers
- `Start_Paper_Trading.bat`
- `Start_GlobalWatch.bat`
- `Start_GlobalWatch_And_Paper.bat`

## Outputs
Main runtime artifacts are written under `outputs/`:
- `snapshot_live.json`: live state payload for UI cards/charts
- `trade_history.jsonl`: normalized trade rows for UI table
- `portfolio_snapshots.jsonl`: per-cycle snapshots for audit
- `paper_trades.csv`: execution-oriented trade log
- `paper_summary_live.txt`: rolling text summary
- `paper_summary.txt`: final report
- `scoreboard.jsonl`: performance diagnostics

## Web Monitor
The Streamlit app includes:
- `Global Macro Signals`: macro/topic interaction and signal diagnostics
- `Portfolio Monitor`: equity, cash, holdings composition, trade history, and summary text

Data mapping:
- cards/charts: `outputs/snapshot_live.json`
- trade table: `outputs/trade_history.jsonl` (fallback can read `outputs/paper_trades.csv`)
- summary block: `outputs/paper_summary_live.txt`

## Key CLI Diagnostics
### GlobalWatch
```bash
python GlobalWatch_V2.py --dump-industry-taxonomy --config paper_config.json
python GlobalWatch_V2.py --industry-sanity-check --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once-debug --config paper_config.json --output outputs/industry_runtime_debug_latest.json
python GlobalWatch_V2.py --run-industry-one-bucket-debug rates_and_gold --config paper_config.json --max_evidence 4 --llm_timeout_seconds 120 --output outputs/debug_industry_one_bucket.json
python GlobalWatch_V2.py --debug-industry-news --config paper_config.json --debug-outdir outputs/gw_industry_dryrun
```

### Paper engine
```bash
python paper_trading.py --debug-news-overlay-once --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --calibrate-news-overlay --lookback-hours 72 --target-cash-delta 0.02 --config paper_config.json --out outputs/news_overlay_calibration.json
python paper_trading.py --debug-news-overlay-phase2 --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --debug-system-s1-5 --debug-outdir outputs/gw_dryrun paper_config.json
```

## Key Config Quick Reference (`paper_config.json`)
Only high-level meaning is listed below.

### `execution`
- `signal_refresh_minutes`: signal refresh cadence
- `macro_refresh_minutes`: macro refresh cadence
- `weight_threshold`: rebalance trigger threshold
- `min_trade_notional_usd`: minimum trade size
- `max_turnover_pct_per_rebalance`: turnover cap per cycle
- `max_stale_ratio`: stale-ratio abort guard
- `price_stale_policy.allow_buy`: BUY quote statuses allowed
- `price_stale_policy.allow_sell`: SELL quote statuses allowed
- `rebalance_cooldown_minutes`: cooldown after successful rebalance
- `rebalance_attempt_cooldown_minutes`: cooldown after blocked attempts

### `trade_planner`
- `enable_trade_planner`: planner switch
- `allow_partial_fill`: partial fill on budget pressure
- `min_trade_notional`: planner min notional filter
- `enable_adv_limit`: ADV hard limit switch
- `adv_limit_frac`: max participation per trade
- `enable_cost_sensitive_ranking`: score-based ranking switch
- `lambda_cost`: cost penalty strength

### `news_overlay`
- `enabled`: industry overlay switch
- `mode`: default `risk_only`
- `alpha`: overlay sensitivity
- `min_confidence`: confidence floor
- `max_abs_delta`: per-bucket clip
- `enable_confidence_scaling`: confidence scaling switch
- `max_age_hours`: freshness filter
- `macro_risk_off_prior.enabled`: risk-off prior switch
- `macro_risk_off_prior.strength`: prior strength
- `macro_risk_off_prior.min_score`: prior gating score
- `macro_risk_off_prior.cooldown_minutes`: prior cooldown

### `risk_model`
- covariance diagnostics and vol-targeting options (all optional)

### `cost_model`
- estimated fee/slippage/impact controls for audit/planner scoring

## Validation
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

## New Quant CLIs (Execution/No-Trade Attribution)
```bash
python scripts/quant/a19_compute_exec_blockers.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD
python scripts/quant/a20_attach_exec_blockers_to_daily.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD
```

These commands generate and embed:
- cycle-level blocker distribution (`market_closed`, `attempt_cooldown`, `risk/cov gate`, etc.)
- day-level no-trade reason inference

## Notes
- Paper trading only.
- No real broker connection.
- Not investment advice.
