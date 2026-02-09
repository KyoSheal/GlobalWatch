# GlobalWatch Paper Trading (V2.11.3)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## Overview
GlobalWatch Paper Trading is a local simulation framework for quantitative strategy research, execution validation, and risk-control testing.
It does not connect to a real broker. The goal is to make strategy behavior observable, auditable, and easy to iterate.

The system combines:
- market data and cross-sectional ranking logic
- regime and macro/topic overlays
- execution safeguards (stale price policy, turnover control, forced de-risking)
- live monitoring outputs for the Streamlit interface

## What Is New in V2.11.3
This release focuses on planner quality and audit clarity while preserving the main strategy flow.

Main improvements:
- planner score normalization now uses comparable units (`benefit` vs `cost_weight`)
- planner audit records are cleaned so one trade cannot appear as both scaled and dropped
- richer planner diagnostics in snapshot payloads (including score distribution summary)
- stronger snapshot fields for UI and troubleshooting
- engine fingerprint version updated to `v2.11.3-2026-02-09`

## Functional Capabilities
### Quant layer
- cross-sectional selection and ranking
- volatility-aware weighting
- correlation filtering / diversification control
- regime-aware cash and exposure controls
- macro signal integration and topic tilts
- optional covariance diagnostics and vol-targeting hooks

### Execution and risk layer
- stale quote policy for BUY/SELL
- stale-ratio abort protection
- turnover cap handling
- circuit-breaker / risk-off forced de-risk path
- trade planner with forced-trade priority
- planner diagnostics, cost estimates, and structured decision traces

### System layer
- checkpoint resume / fresh start control
- deterministic config-driven runtime
- cycle-by-cycle snapshot persistence
- live summary and report generation
- Streamlit monitoring integration

## Quick Start
### Run paper engine
```bash
python -u paper_trading.py paper_config.json
```

### Run GlobalWatch UI
```bash
streamlit run GlobalWatch_V2.py
```

### Windows launchers
- `Start_Paper_Trading.bat`
- `Start_GlobalWatch.bat`
- `Start_GlobalWatch_And_Paper.bat`

## Outputs
Core runtime outputs are written under `outputs/`:

- `snapshot_live.json`: live state payload for UI
- `trade_history.jsonl`: normalized trade rows for UI table
- `portfolio_snapshots.jsonl`: cycle snapshots for audit
- `paper_trades.csv`: execution-focused trade log
- `paper_summary_live.txt`: rolling text summary
- `paper_summary.txt`: final summary
- `scoreboard.jsonl`: rolling performance diagnostics
- `equity_curve.png`: end-of-run equity chart

## Web Monitoring (Streamlit)
The UI provides two main pages:

- `Global Macro Signals`: macro/topic observation and signal diagnostics
- `Portfolio Monitor`: live equity, cash, holdings composition, trade history, summary text

Data source mapping:
- portfolio cards/charts: `outputs/snapshot_live.json`
- trade table: `outputs/trade_history.jsonl` (fallback may use `outputs/paper_trades.csv`)
- text summary: `outputs/paper_summary_live.txt`

If UI content looks stale, verify that the paper engine is still writing `snapshot_live.json`.

## Key Config Quick Reference (`paper_config.json`)
Only purpose labels are listed here (no proprietary thresholds).

### execution
- `signal_refresh_minutes`: signal refresh cadence
- `macro_refresh_minutes`: macro refresh cadence
- `weight_threshold`: rebalance trigger threshold
- `min_trade_notional_usd`: minimum tradable notional
- `max_turnover_pct_per_rebalance`: turnover cap per cycle
- `max_stale_ratio`: stale-ratio abort guard
- `price_stale_policy.allow_buy`: BUY quote policy
- `price_stale_policy.allow_sell`: SELL quote policy
- `circuit_breaker_forced_days`: forced risk-off duration
- `fill_gap_max`: small-gap soft fill cap
- `fill_gap_max_iters`: soft fill max iterations
- `allow_buy_benchmarks`: benchmark buy switch
- `cross_section_top_n`: cross-sectional selection size
- `correlation_lookback_days`: correlation lookback window
- `correlation_threshold`: correlation filter threshold
- `volatility_floor`: volatility denominator floor
- `min_holding_cycles`: minimum holding lock
- `enable_exit_signals`: exit-signal switch
- `exit_signal_lookback_days`: exit-signal lookback
- `exit_on_gap_volume`: gap/volume crash-exit switch
- `max_weight_boost_for_hot`: hot-asset cap boost
- `hot_zscore_threshold`: hot-asset z-score gate
- `hot_momentum_top_k`: hot-asset momentum rank gate
- `hot_persistence_cycles`: hot-asset persistence gate

### macro_integration
- `enable_llm_topic_signals`: topic signal switch
- `topic_memory_window`: topic memory window
- `llm_topic_confidence_threshold`: confidence gate
- `llm_topic_score_threshold`: score gate
- `llm_topic_tilt_scale`: topic tilt scaling
- `macro_cash_slope`: macro-to-cash sensitivity
- `tilt_max_delta`: per-asset tilt clamp
- `macro_allow_new_positions`: risk-off allowlist for new positions

### risk_model
- `enable_cov_diagnostics`: covariance diagnostics switch
- `shrinkage_alpha`: covariance shrinkage strength
- `annualization_factor`: annualization factor
- `max_pair_corr_pairs`: top correlation pairs to report
- `fallback_to_diag_on_error`: covariance fallback behavior
- `enable_vol_targeting`: vol-targeting switch
- `vol_target`: target annualized volatility
- `vol_target_min_coverage`: minimum covariance coverage
- `vol_target_min_scale`: lower scale clamp
- `vol_target_max_scale`: upper scale clamp
- `vol_target_use_cov_only`: cov-only scaling policy

### trade_planner
- `enable_trade_planner`: planner switch
- `allow_partial_fill`: partial-fill switch
- `min_trade_notional`: planner min notional
- `enable_adv_limit`: ADV limit switch
- `adv_limit_frac`: max participation per trade
- `adv_lookback_days`: ADV lookback window
- `adv_apply_to_forced`: apply ADV clamp to forced trades
- `enable_cost_sensitive_ranking`: score ranking switch
- `lambda_cost`: cost penalty strength
- `benefit_mode`: benefit proxy mode
- `max_audit_items`: audit list truncation length

### cost_model
- `enabled`: cost estimation switch
- `fee_bps`: fee basis points
- `slippage_bps`: slippage basis points
- `impact_enabled`: impact model switch
- `impact_k`: impact coefficient
- `adv_lookback_days`: ADV lookback for participation

### reporting
- `trades_log_path`: trades csv path
- `portfolio_snapshots_path`: snapshots path
- `summary_report_path`: summary path
- `scoreboard_path`: scoreboard path
- `snapshot_live_path`: live snapshot path
- `trade_history_path`: UI trade history path

## Validation Commands
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

Optional quick smoke:
```bash
python -u -c "from paper_trading import PaperTradingEngine; e=PaperTradingEngine('paper_config.json'); print('SMOKE_OK')"
```

## Notes
- Paper trading only.
- No real broker connection.
- Not investment advice.
