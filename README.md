# GlobalWatch Paper Trading (V3.4.1)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and paper-execution system.
It combines cross-sectional alpha selection, macro/topic overlays, and execution/risk controls in one reproducible workflow.

From a quant angle, the stack supports momentum + volatility + correlation-aware allocation, regime-aware cash control, and AI-driven industry news overlays (bidirectional: buy and sell signals).
From a systems angle, it provides checkpoint resume, structured snapshots, deterministic dry-runs, permanent daily summary logs, and a Streamlit dashboard for live monitoring and audit.

## Release Highlights (V3.4.1)

1. **Semantic vector retrieval** — industry signal lookup upgraded from full `collection.get()` to `collection.query()`:
   - `_read_recent_industry_signals(tickers)` now receives current holdings and queries ChromaDB for the top-N most relevant sector signals
   - Similarity weighting: `effective_weight = decay_weight × (0.5 + 0.5 × similarity)` — distant sectors auto-downweighted
   - Fallback to `get()` when tickers list is empty; `distance` field exposed in every row for diagnostics

2. **Closed-loop IC feedback** — AI overlay now learns which sector signals are accurate:
   - `_append_prediction_log()` writes `{cycle_ts, worst_l2, predicted_delta, cash_adj, snapshot_equity}` to ephemeral `prediction_log.jsonl` after each applied overlay
   - `_settle_previous_predictions()` runs at the start of each cycle: compares prior snapshot equity to current equity, computes direction correctness, and updates `signal_ic_state.json` via EMA
   - IC EMA formula: `ic_new = 0.1 × contribution + 0.9 × ic_old` — recent market regimes automatically outweigh stale history

3. **Three-layer state separation** — prediction log and IC state are fully decoupled:
   - `prediction_log.jsonl` — ephemeral, safe to delete for debugging without losing learned IC
   - `signal_ic_state.json` — durable, persists across restarts and log deletions; stores `{L2: {ic, n_settled}}`
   - Deleting logs never resets the IC state the system has learned

4. **Adaptive alpha (gradual ramp-up)** — per-L2 alpha dynamically scales with IC confidence:
   - `effective_alpha = base_alpha × ((1−conf) × 1.0 + conf × clamp(1 + ic × 3, 0.5, 2.0))`
   - `conf = min(1.0, n_settled / min_cycles_before_adaptive)` — confidence grows linearly from 0 to 1
   - No hard cutoff: system is immediately useful from cycle 1, reaches full adaptation at `min_cycles_before_adaptive` (default 20, configurable to 5 for dev)
   - IC diagnostics (`ic`, `ic_conf`, `n_settled`, `effective_alpha`) visible in `l2_delta_map_sample` of every cycle snapshot

5. **Industry signal staleness detection** — prevents silent `no_data` failures:
   - `_read_recent_industry_signals()` emits `[INDUSTRY_SIGNAL_STALE]` warning with last-record age and remediation command when no fresh signals found
   - `run_news_pipeline.py --include-industry` adds step 4 that checks `industry_signals` collection freshness and prints `[INDUSTRY_HEALTH] OK/STALE/WARN`

## Release Highlights (V3.3.0)

1. **Full CAD/USD FX overhaul** — all portfolio accounting is now in USD throughout:
   - `_build_post_rebalance_snapshot`, `_evaluate_portfolio_risk_gate`, `_run_circuit_breaker_derisk` all apply FX
   - BUY/SELL execution paths use `price_usd = price × fx_rate` for qty, cash deduction, and proceeds
   - Canadian `.TO` stocks display with `C$` prefix to distinguish from USD prices
   - Real-time CAD/USD rate via yfinance (`CADUSD=X`), 60-min cache, fallback `0.73`

2. **New industry baskets: Chip + Semiconductor** — 13 new tickers added to universe (65 → 78):
   - Chip: `AMD, QCOM, AVGO, TXN, MRVL, ARM`
   - Semiconductor: `TSM, ASML, LRCX, KLAC, AMAT, MU, ON`
   - Full `industry_map`, `ticker_tags`, `industry_taxonomy` (L2/L3), `topic_sector_ticker_map`, `industry_topic_queries`, `industry_keyword_map` entries added

3. **AI news overlay upgraded to alpha mode**:
   - `news_overlay.mode`: `risk_only` → `symmetric` — AI signals can now drive both BUY and SELL
   - `news_overlay.max_abs_delta`: `0.03` → `0.05` — wider weight shift range per signal
   - Previously, Ollama/LLM could only reduce positions; now it drives the full allocation

4. **Risk gate tuned for mid profile**:
   - `rc_limit`: `0.45` → `0.65` — prevents portfolio from being permanently locked
   - `pass_threshold` = `0.63`, `fail_threshold` = `0.67`

5. **Execution parameters relaxed**:
   - `ramp_in_cycles`: `3` → `2` — new positions reach full weight in 40 min instead of 60 min
   - `macro_allow_new_positions`: expanded from 2 tickers (TLT, GLD) to 13 (major ETFs + sector ETFs)

6. **Permanent daily summary log**:
   - Each cycle appends a one-line summary to `outputs/daily/YYYY-MM-DD.log`
   - This file is never deleted by cleanup routines — provides a permanent compact audit trail
   - Full `app.log` remains a rotating temp file

## Documentation
- Detailed English documentation: `README.en.md`
- 详细中文文档: `README.zh.md`

## Quick Start
```bash
python -u paper_trading.py paper_config.json
```

```bash
python run_news_pipeline.py --interval 30
```

```bash
streamlit run GlobalWatch_V2.py
```

## CI Dependencies
```
pandas  numpy  matplotlib  yfinance
```


# 启动
./start_trading_day.sh

# 监控（看是否在跑）
ps aux | grep -E "paper_trading|news_pipeline" | grep -v grep

# 停止
pkill -f paper_trading.py; pkill -f run_news_pipeline.py