# GlobalWatch Paper Trading (V2.11.3)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and execution sandbox.  
It combines market data, macro/topic signals, and portfolio risk controls into one paper-trading workflow, with no real broker connection.

From a quant perspective, the engine includes cross-sectional ranking, regime/macro overlays, and execution-aware risk controls.  
From a systems perspective, it provides checkpoint/resume, config-driven reproducibility, and UI-ready structured outputs for monitoring and audit.

## Documentation
- Full English docs (features, start, outputs, web monitor, config quick reference): `README.en.md`
- 完整中文文档（功能、启动、输出、Web 查看、关键配置速查）: `README.zh.md`

## Quick Start
```bash
python -u paper_trading.py paper_config.json
```

```bash
streamlit run GlobalWatch_V2.py
```
