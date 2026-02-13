# GlobalWatch Paper Trading (V3.2.2)

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

## What's New in V3.2.2
This release finalizes the recent strategy and system upgrades and moves the engine fingerprint to `v3.2.2`.

### Quant and Signal Engine
- L2/L3 multi-tag taxonomy support with `industry_taxonomy` + `ticker_tags`
- industry runtime no longer starves buckets before bucketing; dedup happens before per-bucket limits
- seed-aware industry mapping and bucket contamination controls
- deterministic industry scoring path: evidence labels -> deterministic bucket score -> bounded signal
- macro risk-off prior (gated + cooldown) for risk-sensitive buckets
- confidence and schema guards for LLM output parsing and fallback handling

### Paper Engine and Overlay
- industry signal consumption is integrated into cash-target overlay with explicit gating
- `min_confidence`, `risk_only`, `max_abs_delta`, and confidence-scaling are enforceable and debuggable
- calibration and replay tools added to tune overlay strength against historical signal distribution
- covariance risk diagnostics are retained for risk visibility and audit outputs

### Execution and Risk Control
- trade planner path remains configurable and auditable
- stale quote policy and session-aware rebalance gates remain active
- post-rebalance snapshot refresh ensures UI holdings state is up-to-date after fills
- debug fields for planner/cost/overlay are written for cycle-level traceability

### UI and UX
- interactive macro analysis can run in scoped mode (selected targets only)
- optional `Request <think>` checkbox added; default is concise JSON path
- when no `<think>` block is returned, UI now shows `Reasoning Summary` instead of misleading placeholder text
- equity chart supports trading-hour focused visualization and manual axis controls
- portfolio monitor reads latest snapshot/trades and shows newest trades first in UI

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

## Notes
- Paper trading only.
- No real broker connection.
- Not investment advice.
