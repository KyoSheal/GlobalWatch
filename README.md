# GlobalWatch Paper Trading (V3.3.0)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and paper-execution system.
It combines cross-sectional alpha selection, macro/topic overlays, and execution/risk controls in one reproducible workflow.

From a quant angle, the stack supports momentum + volatility + correlation-aware allocation, regime-aware cash control, and AI-driven industry news overlays (bidirectional: buy and sell signals).
From a systems angle, it provides checkpoint resume, structured snapshots, deterministic dry-runs, permanent daily summary logs, and a Streamlit dashboard for live monitoring and audit.

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
   - `news_overlay.mode`: `risk_only` → `symmetric`` — AI signals can now drive both BUY and SELL
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
