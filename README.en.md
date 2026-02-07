# GlobalWatch Paper Trading (V2.10.1)

[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)
[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)

## 1. Overview
- Local automated paper-trading engine with no real broker connection.
- Built for strategy iteration, execution validation, and risk-control testing.
- This document exposes operation and validation flow, but not proprietary thresholds.

## 2. Version Delta (vs previous release)
### 2.1 Key additions from `v2.9.1` to `v2.10.1`
- Step 1: cash-efficiency update with optional high-conviction cash override (still respecting cash floor).
- Step 2: momentum upgraded from single-timescale to blended multi-timescale signals.
- Step 3: high-conviction single-asset weight boost under portfolio risk constraints.
- Step 4: exit-signal module using simple price/volume patterns for reduce/exit actions.
- Step 5: GlobalWatch structured topic signals with optional LLM topic injection.
- Step 6: score smoothing/normalization/clipping plus portfolio-level risk gates.

### 2.2 Safety architecture retained
- decoupled signal/macro refresh
- stale-price strict policy and stale-ratio hard abort
- turnover cap applied to final executable trades
- circuit breaker unified into `risk_off_forced`
- benchmark universe decoupling
- rolling `scoreboard.jsonl` diagnostics

## 3. Quick Start
### 3.1 Requirements
- Python 3.10+
- Main dependencies: `pandas`, `numpy`, `yfinance`, `matplotlib`

### 3.2 Run
```bash
python -u paper_trading.py paper_config.json
```

Windows shortcut:
```bash
Start_Paper_Trading.bat
```

### 3.3 Outputs
- `outputs/paper_trades.csv` # execution log
- `outputs/portfolio_snapshots.jsonl` # cycle snapshots
- `outputs/scoreboard.jsonl` # rolling performance windows
- `outputs/paper_summary_live.txt` # live status
- `outputs/paper_summary.txt` # final report

## 4. Core Engine Behavior
### 4.1 Refresh decoupling
- Snapshot is recorded every cycle.
- Macro signals refresh on macro cadence.
- Target weights refresh on signal cadence.
- Reuse state and last refresh timestamps are persisted in snapshots.

### 4.2 Execution safeguards
- BUY does not accept STALE quotes (policy-driven).
- Rebalance aborts when stale ratio of candidates exceeds threshold.
- Turnover cap is enforced on final tradable notionals.

### 4.3 Portfolio-level safeguards (v2.10.1)
- score stabilization: optional smoothing, normalization, clipping
- volatility gate: abort if weighted portfolio volatility is confidently above limit
- diversity gate: abort if concentration (HHI) exceeds limit
- risk-gate diagnostics are written into snapshots

## 5. Config Quick Reference (`paper_config.json`)
Note: short purpose labels only; no proprietary threshold logic disclosed.

### 5.1 `execution`
- `signal_refresh_minutes` # signal cadence
- `macro_refresh_minutes` # macro cadence
- `weight_threshold` # rebalance trigger
- `min_trade_notional_usd` # min trade size
- `max_turnover_pct_per_rebalance` # turnover cap
- `max_stale_ratio` # stale abort ratio
- `price_stale_policy.allow_buy` # buy quote policy
- `price_stale_policy.allow_sell` # sell quote policy
- `circuit_breaker_forced_days` # forced risk-off duration
- `fill_gap_max` # soft fill ceiling
- `fill_gap_max_iters` # fill iterations
- `allow_buy_benchmarks` # benchmark buy switch
- `cross_section_top_n` # top-N selection
- `correlation_lookback_days` # corr lookback
- `correlation_threshold` # corr threshold
- `volatility_floor` # vol floor
- `min_holding_cycles` # holding lock
- `allow_high_conviction_override` # cash override switch
- `enable_high_conviction_weighting` # weight-boost switch
- `max_high_conviction_weight` # boosted max cap
- `enable_short_term_momentum` # short momentum switch
- `short_momentum_lookback_days` # short momentum window
- `enable_exit_signals` # exit signal switch
- `exit_signal_lookback_days` # exit signal window
- `enable_score_smoothing` # score smoothing switch
- `score_smoothing_window` # smoothing window
- `max_portfolio_volatility` # vol gate limit
- `enable_diversity_check` # diversity gate switch
- `max_herfindahl_index` # concentration limit
- `portfolio_vol_min_coverage` # min vol coverage

### 5.2 `macro_integration`
- `macro_cash_slope` # cash sensitivity
- `tilt_max_delta` # tilt cap
- `macro_allow_new_positions` # risk-off allowlist
- `enable_llm_topic_signals` # LLM topic switch
- `llm_topic_confidence_threshold` # confidence gate
- `llm_topic_score_threshold` # topic gate
- `llm_topic_tilt_scale` # tilt scale

### 5.3 `reporting`
- `trades_log_path` # trades output
- `portfolio_snapshots_path` # snapshots output
- `summary_report_path` # report output
- `scoreboard_path` # scoreboard output

## 6. Validation Checklist
### 6.1 Compile
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

### 6.2 Runtime artifacts
```powershell
Get-Content outputs\portfolio_snapshots.jsonl -Tail 3
Get-Content outputs\paper_trades.csv -Tail 5
Get-Content outputs\scoreboard.jsonl -Tail 5
```

### 6.3 Acceptance targets
1. Refresh reuse flags work under short intervals.
2. STALE scenarios trigger expected skip/abort behavior.
3. Turnover cap constrains final executable trades.
4. Risk gates block high-vol or high-concentration cycles.
5. Circuit-breaker flow enters `risk_off_forced` instead of permanent pause.

## 7. Chinese Text Garbling FAQ
- Chinese garbling is usually an encoding/code-page issue, not a timezone issue.
- If GitHub renders correctly but terminal does not, adjust terminal encoding.
- PowerShell recommendation:
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 8. Safety Notice
- Paper trading only.
- No real broker connection.
- Not investment advice.
