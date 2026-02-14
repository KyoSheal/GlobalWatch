# GlobalWatch Paper Trading (V3.2.4)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

GlobalWatch Paper Trading is a local-first quantitative research and paper-execution system.
It combines cross-sectional alpha selection, macro/topic overlays, and execution/risk controls in one reproducible workflow.

From a quant angle, the stack supports momentum + volatility + correlation-aware allocation, regime-aware cash control, and optional industry news overlays.
From a systems angle, it provides checkpoint resume, structured snapshots, deterministic dry-runs, and a Streamlit dashboard for live monitoring and audit.

## Release Highlights (V3.2.4)
1. Engine fingerprint upgraded to `v3.2.4`.
2. Risk gate routing now supports target-cov basis with explicit enable switch and automatic fallback:
   - `execution.enable_target_cov_gate`
   - `execution.target_cov_gate_min_coverage`
   - `execution.target_cov_gate_require_ok`
3. Risk gate observability was expanded across runtime outputs:
   - snapshot fields: `risk_gate_basis`, `risk_gate_cov_coverage_used`
   - telemetry event: `RISK_GATE_DECISION`
   - cycle metrics fields for gate basis and coverage used
4. Portfolio exposure cap is now configurable instead of hardcoded:
   - `execution.portfolio_exposure_cap` (default `0.90`)
   - invalid values auto-fallback with warning and clamp to `[0.0, 1.0]`

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

