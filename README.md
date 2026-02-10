# GlobalWatch Paper Trading (V3.1.2)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and execution sandbox.  
It combines market data, macro/topic signals, and portfolio risk controls into one paper-trading workflow, with no real broker connection.

From a quant perspective, the engine includes cross-sectional ranking, regime/macro overlays, and execution-aware risk controls.  
From a systems perspective, it provides checkpoint/resume, config-driven reproducibility, and UI-ready structured outputs for monitoring and audit.

## What's New in V3.1.2
This release consolidates S1-S5 updates and promotes the engine fingerprint to `v3.1.2`.

Quant updates:
- stale-ratio logic now uses policy-pass candidates only; stale-abort is only valid during tradable OPEN session
- turnover planner and post-planner execution filtering are better aligned for cleaner trade quality and auditability
- `price_debug` now exposes per-ticker source, timestamp, timezone quality, and age-threshold reasoning

System updates:
- market-session aware gate blocks rebalance when market is closed or pre-open grace is not passed
- rebalance attempt cooldown prevents busy retry loops after abort/skip conditions
- atomic output writes for `snapshot_live.json` and `trade_history.jsonl` reduce partial-read UI errors
- run identity is now explicit via `session_id` and `config_hash` in snapshot/trade outputs
- built-in offline deterministic dry-run entry and PASS/FAIL summary

## Documentation
- Full English docs (features, start, outputs, web monitor, config quick reference): `README.en.md`
- 完整的中文文档（功能、入门、输出、Web 监控、配置快速参考） `README.zh.md`

## Quick Start
```bash
python -u paper_trading.py paper_config.json
```

```bash
streamlit run GlobalWatch_V2.py
```

