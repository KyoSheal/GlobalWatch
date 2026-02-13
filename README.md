# GlobalWatch Paper Trading (V3.2.2)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and paper-execution system.
It combines cross-sectional alpha selection, macro/topic overlays, and execution/risk controls in one reproducible workflow.

From a quant angle, the stack supports momentum + volatility + correlation-aware allocation, regime-aware cash control, and optional industry news overlays.
From a systems angle, it provides checkpoint resume, structured snapshots, deterministic dry-runs, and a Streamlit dashboard for live monitoring and audit.

## Release Highlights (V3.2.2)
- engine fingerprint upgraded to `v3.2.2`
- industry pipeline upgraded with L2/L3 multi-tag taxonomy, seed-aware bucketing, and stronger runtime diagnostics
- deterministic industry scoring path (label -> score) with macro risk-off prior gate and cooldown controls
- paper engine news overlay calibration tooling (one-shot debug + replay calibrator)
- interactive UI improved with scoped analysis mode, optional `<think>` request, and fallback `Reasoning Summary`
- portfolio monitor UX improvements, including improved trade-history ordering and equity chart controls

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
