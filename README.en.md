# GlobalWatch Paper Trading (V3.4.1)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## Overview
GlobalWatch Paper Trading is a local simulation framework for quantitative strategy research, execution validation, and risk-control testing.
It does not connect to a real broker. The focus is transparent behavior, stable iteration, and auditable outputs.

The stack combines:
- cross-sectional ranking and portfolio construction
- market-regime and macro/topic overlays
- AI-driven industry news signals (bidirectional buy + sell via local Ollama LLM)
- execution safeguards (stale policy, turnover budget, planner, FX-correct accounting)
- structured runtime outputs consumed by Streamlit monitoring pages

## What's New in V3.4.1

### 1. Semantic Vector Retrieval
Industry signal lookup upgraded from full `collection.get()` to `collection.query()`:
- `_read_recent_industry_signals(tickers)` receives current holdings and queries ChromaDB for the top-N most relevant sector signals
- Similarity weighting: `effective_weight = decay_weight × (0.5 + 0.5 × similarity)` — distant sectors auto-downweighted
- Falls back to `get()` when no tickers provided; `distance` field exposed per row for diagnostics

### 2. Closed-Loop IC Feedback
The AI overlay now learns which sector signals are accurate over time:
- `_append_prediction_log()` writes `{cycle_ts, worst_l2, predicted_delta, cash_adj, snapshot_equity}` to ephemeral `prediction_log.jsonl` after each applied overlay
- `_settle_previous_predictions()` runs at cycle start: compares prior snapshot equity to current equity, computes direction correctness, and updates `signal_ic_state.json` via EMA
- IC EMA: `ic_new = 0.1 × contribution + 0.9 × ic_old` — recent regime automatically outweighs stale history

### 3. Three-Layer State Separation
Prediction log and IC state are fully decoupled:
- `prediction_log.jsonl` — ephemeral, safe to delete for debugging
- `signal_ic_state.json` — durable, persists across restarts; stores `{L2: {ic, n_settled}}`
- Deleting logs never resets learned IC state

### 4. Adaptive Alpha (Gradual Ramp-Up)
Per-L2 alpha dynamically scales with IC confidence:
- `effective_alpha = base_alpha × ((1−conf) × 1.0 + conf × clamp(1 + ic × 3, 0.5, 2.0))`
- `conf = min(1.0, n_settled / min_cycles_before_adaptive)` — useful from cycle 1, fully adaptive at `min_cycles_before_adaptive` (default 20, dev: 5)
- IC diagnostics (`ic`, `ic_conf`, `n_settled`, `effective_alpha`) visible in `l2_delta_map_sample` of every cycle snapshot

### 5. Industry Signal Staleness Detection
Prevents silent `no_data` failures in the news overlay:
- `_read_recent_industry_signals()` emits `[INDUSTRY_SIGNAL_STALE]` warning with last-record age and remediation command when no fresh signals are found
- `run_news_pipeline.py --include-industry` adds a step 4 that checks `industry_signals` collection freshness and prints `[INDUSTRY_HEALTH] OK/STALE/WARN`

## What's New in V3.3.0

### 1. Full CAD/USD FX Overhaul
All portfolio accounting is now consistently in USD throughout the engine.

**Root cause fixed:** Canadian `.TO` stocks (e.g. `MFC.TO`, `SU.TO`) are priced in CAD. Previously several code paths used the native CAD price directly in USD cash arithmetic, causing phantom P&L losses on every buy cycle.

**Fixed locations:**
| Function | Fix |
|---|---|
| `_build_post_rebalance_snapshot` | `value = qty × price × fx` |
| `_evaluate_portfolio_risk_gate` | fallback value uses `fx` |
| SELL execution path | `price_usd = price × fx_sell`, `proceeds = sell_qty × price_usd` |
| BUY execution path | `price_usd = price × fx_buy`, `required_cash = buy_qty × price_usd` |
| `_run_circuit_breaker_derisk` | holdings dict stores `fx`, sell qty and proceeds use `price × fx` |
| SELL/BUY weight fallbacks | use `price × fx` in `old_position_value` |
| Display | `.TO` stocks print `C$` prefix; trade logs show USD price |

**FX rate mechanism:**
- Source: yfinance `CADUSD=X` (real-time)
- Cache: 60 minutes in-memory
- Fallback: `0.73` from `paper_config.json → fx_rates.CAD_USD` if fetch fails
- Non-`.TO` tickers always use `fx = 1.0`

### 2. New Industry Baskets: Chip + Semiconductor
Universe expanded from **65 → 78 tickers**.

**Chip basket** (`industry_map.chip`):
| Ticker | Company |
|---|---|
| AMD | Advanced Micro Devices |
| QCOM | Qualcomm |
| AVGO | Broadcom |
| TXN | Texas Instruments |
| MRVL | Marvell Technology |
| ARM | Arm Holdings |

**Semiconductor basket** (`industry_map.semiconductor`):
| Ticker | Company |
|---|---|
| TSM | Taiwan Semiconductor |
| ASML | ASML Holding |
| LRCX | Lam Research |
| KLAC | KLA Corp |
| AMAT | Applied Materials |
| MU | Micron Technology |
| ON | ON Semiconductor |

All 13 tickers have full entries in:
- `universe` (tradable)
- `industry_map` (sector concentration caps)
- `ticker_tags` (L2/L3 labels + keywords for news routing)
- `industry_taxonomy` (L2: chip, semiconductor; L3: fabless_cpu_gpu, foundry, semi_equipment, memory, etc.)
- `topic_sector_ticker_map`, `industry_topic_queries`, `industry_keyword_map`

### 3. AI News Overlay — Alpha Mode (Bidirectional)
Previously the Ollama/LLM news signal could **only reduce positions** (`risk_only` mode).
This meant the AI could never act on positive news to increase or initiate positions — a fundamental limitation.

**Changes:**
| Parameter | Before | After |
|---|---|---|
| `news_overlay.mode`: `risk_only` → `symmetric`` |
| `news_overlay.max_abs_delta` | `0.03` (3%) | `0.05` (5%) |

In `alpha` mode, the LLM signal (qwen2.5:32b / gemma3:12b via Ollama) can:
- **Increase** position weights on positive sector news
- **Decrease** position weights on negative sector news
- Open new positions when signal confidence ≥ `min_confidence` (0.45)

### 4. Risk Gate Tuning (mid profile)
| Parameter | Before | After |
|---|---|---|
| `rc_limit` | `0.45` | `0.65` |
| `pass_threshold` | `0.43` | `0.63` |
| `fail_threshold` | `0.47` | `0.67` |
| `hysteresis_band` | `0.02` | `0.02` (unchanged) |

The previous limit of 0.45 caused the gate to permanently lock at ~60% rc_fraction (energy/Canada concentration), blocking all trades for days.

### 5. Execution Parameter Updates (mid profile)
| Parameter | Before | After | Effect |
|---|---|---|---|
| `ramp_in_cycles` | `3` (60 min) | `2` (40 min) | New positions reach full weight faster |
| `macro_allow_new_positions` | `[TLT, GLD]` | 13 tickers | Macro signals can open positions in major ETFs |

`macro_allow_new_positions` now includes:
`TLT, GLD, SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLP, XLI, VOO, XIU.TO`

### 6. Permanent Daily Summary Log
A new `_write_daily_summary()` method appends a compact one-line record per cycle to:
```
outputs/daily/YYYY-MM-DD.log
```

**Format:**
```
[2026-05-12 09:42:02] Cycle=223 Cash=$52,047 Pos=$28,618 Equity=$80,665 Return=+0.83% RC=60.6% Gate=ABORT Regime=RISK_ON Trades=0 Holdings=[BP x42 | MFC.TO x304 ...]
```

- **Permanent** — never deleted by log cleanup routines
- **Full `app.log`** remains a rotating temp file (deleted on restart)
- Enables fast multi-day performance review without loading large logs

## Functional Capabilities

### Quant Layer
- cross-sectional score ranking
- volatility-aware allocation and clipping
- correlation filtering
- regime-aware cash target and max-weight controls
- macro/topic/industry overlays (bidirectional alpha mode)

### Execution and Risk Layer
- FX-correct accounting for CAD/USD mixed portfolios
- stale quote policy for BUY/SELL paths
- stale-ratio abort guard during tradable sessions
- turnover budgeting with planner diagnostics
- exit-signal hooks and forced de-risk path
- rolling drawdown circuit breaker
- stop-loss override (-8% from cost basis)
- ramp-in gradual position entry
- optional cost estimation, ADV checks, and planner scoring controls

### AI Signal Layer (Ollama / Local LLM)
- Models: `qwen2.5:32b`, `gemma3:12b` (always in RAM, ~22 GB)
- News pipeline: `run_news_pipeline.py --interval 30` (separate persistent process)
- Signal refresh: every 15 min minimum (`runtime_min_interval_seconds: 900`)
- Coverage: 78-ticker universe across 10+ industry baskets including Chip + Semiconductor
- Mode: `alpha` — drives both buy and sell decisions

### System Layer
- checkpoint resume and fresh-start handling
- deterministic config-driven runtime
- cycle snapshots and trade history persistence
- permanent daily summary logs (`outputs/daily/`)
- dry-run diagnostics and calibration utilities
- Streamlit monitoring integration

## Start
### Paper engine
```bash
python -u paper_trading.py paper_config.json
```

### News pipeline
```bash
python run_news_pipeline.py --interval 30
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

| File | Description |
|---|---|
| `snapshot_live.json` | live state payload for UI cards/charts |
| `trade_history.jsonl` | normalized trade rows for UI table |
| `portfolio_snapshots.jsonl` | per-cycle snapshots for audit |
| `paper_trades.csv` | execution-oriented trade log |
| `paper_summary_live.txt` | rolling text summary |
| `paper_summary.txt` | final report |
| `scoreboard.jsonl` | performance diagnostics |
| `daily/YYYY-MM-DD.log` | **permanent** one-line-per-cycle daily summary |

## Web Monitor
The Streamlit app includes:
- `Global Macro Signals`: macro/topic interaction and signal diagnostics
- `Portfolio Monitor`: equity, cash, holdings composition, trade history, and summary text

Data mapping:
- cards/charts: `outputs/snapshot_live.json`
- trade table: `outputs/trade_history.jsonl`
- summary block: `outputs/paper_summary_live.txt`

## Key CLI Diagnostics
### GlobalWatch
```bash
python GlobalWatch_V2.py --dump-industry-taxonomy --config paper_config.json
python GlobalWatch_V2.py --industry-sanity-check --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once-debug --config paper_config.json --output outputs/industry_runtime_debug_latest.json
python GlobalWatch_V2.py --run-industry-one-bucket-debug chip --config paper_config.json --max_evidence 4 --llm_timeout_seconds 120 --output outputs/debug_industry_one_bucket.json
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

### `fx_rates`
- `CAD_USD`: fallback rate when live fetch fails (default `0.73`)
- `auto_fetch`: enable real-time yfinance fetch (default `true`)
- `auto_fetch_symbol`: Yahoo Finance symbol (default `CADUSD=X`)
- `cache_minutes`: rate cache TTL (default `60`)

### `execution`
- `signal_refresh_minutes`: signal refresh cadence
- `macro_refresh_minutes`: macro refresh cadence
- `weight_threshold`: rebalance trigger threshold (mid: `0.015`)
- `min_trade_notional_usd`: minimum trade size (default `$400`)
- `max_turnover_pct_per_rebalance`: turnover cap per cycle (mid: `0.40`)
- `ramp_in_cycles`: gradual entry cycles for new positions (mid: `2`)
- `min_holding_cycles`: minimum hold before selling (mid: `4`)
- `rebalance_cooldown_minutes`: cooldown after successful rebalance
- `stop_loss_pct`: stop-loss threshold (default `-0.08`)
- `portfolio_exposure_cap`: max invested fraction (default `0.90`)
- `enable_target_cov_gate`: use target-cov as risk gate basis (default `false`)

### `risk_model` (mid profile)
- `rc_limit`: max portfolio risk concentration fraction (`0.65`)
- `portfolio_cov_rc_hysteresis_band`: hysteresis band (`0.02`)
- `portfolio_cov_rc_abort_buffer_enabled`: consecutive-abort relaxation (`true`)

### `trade_planner`
- `enable_trade_planner`: planner switch
- `allow_partial_fill`: partial fill on budget pressure
- `enable_adv_limit`: ADV hard limit switch (default `false`)
- `lambda_cost`: cost penalty strength

### `news_overlay`
- `enabled`: industry overlay switch
- `mode`: `symmetric` (bidirectional) — buy and sell) or `risk_only` (sell only)
- `alpha`: overlay sensitivity (`0.6`)
- `min_confidence`: confidence floor (`0.45`)
- `max_abs_delta`: per-bucket weight clip (`0.05`)
- `max_age_hours`: news freshness window

### `macro_integration`
- `macro_allow_new_positions`: tickers where macro can open new positions
- `cooldown_cycles`: macro signal cooldown in cycles

### `risk_profiles`
Each profile (`low`, `mid`, `high`, `ultra`) can override:
- `rc_limit`, `max_weight_per_asset`, `min_cash_pct`, `max_turnover_pct_per_rebalance`, `weight_threshold`

### `universe`
- 78 tradable assets: US stocks, Canadian `.TO` stocks, ETFs, sector ETFs, CASH
- Chip basket: `AMD QCOM AVGO TXN MRVL ARM`
- Semiconductor basket: `TSM ASML LRCX KLAC AMAT MU ON`

## Validation
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

## Notes
- Paper trading only — no real broker connection.
- Not investment advice.
- Canadian `.TO` stocks are priced in CAD; all portfolio values are converted to USD for accounting.
