# GlobalWatch Paper Trading (V3.3.0)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## 项目概览
GlobalWatch Paper Trading 是一个本地优先的量化研究与纸面交易系统。
它不连接真实券商，目标是让策略行为可观察、可审计、可复现、可迭代。

系统由四层组成：
- **量化层**：横截面打分、组合构建、相关性与波动约束
- **AI 信号层**：本地 Ollama LLM（qwen2.5:32b / gemma3:12b）分析行业新闻，双向驱动买卖
- **执行层**：FX 正确的 CAD/USD 记账、stale 报价策略、换手预算、风控退出
- **系统层**：checkpoint 恢复、快照落盘、永久日度摘要日志、Streamlit 可视化监控

## V3.3.0 更新摘要

### 1. CAD/USD 汇率全面修复
所有组合估值与现金记账现在**全程以 USD 为单位**，彻底修复加拿大股票（`.TO`）的虚假亏损问题。

**根本原因：** `MFC.TO`、`SU.TO` 等加拿大股票以 CAD 报价，此前多个代码路径直接用 CAD 价格做 USD 现金运算，导致每次买入周期产生虚假 P&L 损失（约 -2% ~ -3%）。

**修复位置：**
| 函数 | 修复内容 |
|---|---|
| `_build_post_rebalance_snapshot` | `value = qty × price × fx` |
| `_evaluate_portfolio_risk_gate` | 回退估值加入 `fx` |
| SELL 执行路径 | `price_usd = price × fx_sell`，`proceeds = sell_qty × price_usd` |
| BUY 执行路径 | `price_usd = price × fx_buy`，`required_cash = buy_qty × price_usd` |
| `_run_circuit_breaker_derisk` | holdings 存 `fx`，卖出数量与收益均用 `price × fx` |
| SELL/BUY 权重回退 | `old_position_value` 改用 `price × fx` |
| 打印显示 | `.TO` 股票显示 `C$` 前缀，交易日志显示 USD 价格 |

**实时汇率机制：**
- 数据源：yfinance `CADUSD=X`（实时）
- 缓存：内存 60 分钟
- 兜底：`paper_config.json → fx_rates.CAD_USD = 0.73`（拉取失败时使用）
- 非 `.TO` 股票始终使用 `fx = 1.0`

### 2. 新增行业篮子：芯片 + 半导体
交易 universe 从 **65 → 78 个标的**。

**芯片篮子**（`industry_map.chip`）：
| Ticker | 公司 |
|---|---|
| AMD | 超微半导体 |
| QCOM | 高通 |
| AVGO | 博通 |
| TXN | 德州仪器 |
| MRVL | 美满电子 |
| ARM | Arm Holdings |

**半导体篮子**（`industry_map.semiconductor`）：
| Ticker | 公司 |
|---|---|
| TSM | 台积电 |
| ASML | ASML 光刻机 |
| LRCX | 泛林半导体 |
| KLAC | KLA 检测设备 |
| AMAT | 应用材料 |
| MU | 美光科技 |
| ON | 安森美 |

13 个新标的已完整配置至：
- `universe`（可交易）
- `industry_map`（行业集中度控制）
- `ticker_tags`（L2/L3 标签 + 关键词，用于新闻路由）
- `industry_taxonomy`（L2: chip、semiconductor；L3: fabless_cpu_gpu、foundry、semi_equipment、memory 等）
- `topic_sector_ticker_map`、`industry_topic_queries`、`industry_keyword_map`

### 3. AI 新闻 Overlay 升级为双向 Alpha 模式
此前 Ollama/LLM 新闻信号**只能减仓**（`risk_only` 模式），AI 看好的正面新闻无法驱动买入——这是系统最大的功能缺陷。

**变更：**
| 参数 | 之前 | 现在 |
|---|---|---|
| ``news_overlay.mode`: `risk_only` → `symmetric`
| `news_overlay.max_abs_delta` | `0.03`（3%）| `0.05`（5%）|

`symmetric` 模式下，本地 LLM 信号可以：
- 看好行业新闻 → **增加**对应持仓权重（买入）
- 看空行业新闻 → **减少**对应持仓权重（卖出）
- 置信度 ≥ `min_confidence`（0.45）时可开新仓

### 4. 风险门控调整（mid profile）
| 参数 | 之前 | 现在 |
|---|---|---|
| `rc_limit` | `0.45` | `0.65` |
| `pass_threshold` | `0.43` | `0.63` |
| `fail_threshold` | `0.47` | `0.67` |
| `hysteresis_band` | `0.02` | `0.02`（不变）|

旧阈值 0.45 导致 rc_fraction 持续卡在 ~60%（能源/加拿大板块集中），风险门控连续锁死数天，所有交易被阻断。

### 5. 执行参数优化（mid profile）
| 参数 | 之前 | 现在 | 效果 |
|---|---|---|---|
| `ramp_in_cycles` | `3`（60 分钟）| `2`（40 分钟）| 新仓位更快建满 |
| `macro_allow_new_positions` | `[TLT, GLD]` | 13 个标的 | 宏观信号可在主要 ETF 开新仓 |

`macro_allow_new_positions` 现包含：
`TLT, GLD, SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLP, XLI, VOO, XIU.TO`

### 6. 永久日度摘要日志
新增 `_write_daily_summary()` 方法，每个周期结束后追加一行摘要至：
```
outputs/daily/YYYY-MM-DD.log
```

**格式示例：**
```
[2026-05-12 09:42:02] Cycle=223 Cash=$52,047 Pos=$28,618 Equity=$80,665 Return=+0.83% RC=60.6% Gate=ABORT Regime=RISK_ON Trades=0 Holdings=[BP x42 | MFC.TO x304 | SU.TO x156 ...]
```

- **永久保存** — 不会被日志清理例程删除
- `app.log` 保持轮转临时文件（重启时删除）
- 可快速回顾多天表现，无需加载大型日志

## 交易限制体系（mid profile 关键参数）

| 类型 | 参数 | 当前值 |
|---|---|---|
| 市场时间 | 09:30–16:00 ET | 非交易时段完全阻断 |
| RC 集中度门控 | `rc_limit` | 0.65（pass=0.63, fail=0.67）|
| 单仓上限 | `max_weight_per_asset` | 15% |
| 最小现金 | `min_cash_pct` | 12% |
| 单次换手上限 | `max_turnover_pct_per_rebalance` | 40% |
| 最小交易额 | `min_trade_notional_usd` | $400 |
| 止损 | `stop_loss_pct` | -8% |
| 熔断器 | `circuit_breaker_rolling_drawdown_pct` | 12%（10 周期内）|
| 行业集中度 | `max_sector_weight` | 45% |
| 建仓周期 | `ramp_in_cycles` | 2 周期（40 分钟）|
| 最短持仓 | `min_holding_cycles` | 4 周期（80 分钟）|

## 快速启动

### 启动纸面交易引擎
```bash
python -u paper_trading.py paper_config.json
```

### 启动新闻信号管线
```bash
python run_news_pipeline.py --interval 30
```

### 启动 GlobalWatch 前端
```bash
streamlit run GlobalWatch_V2.py
```

### Windows 启动脚本
- `Start_Paper_Trading.bat`
- `Start_GlobalWatch.bat`
- `Start_GlobalWatch_And_Paper.bat`

## 输出文件说明

| 文件 | 说明 |
|---|---|
| `snapshot_live.json` | 前端卡片/图表主数据 |
| `trade_history.jsonl` | 前端交易历史表 |
| `portfolio_snapshots.jsonl` | 逐轮快照审计 |
| `paper_trades.csv` | 成交导向交易流水 |
| `paper_summary_live.txt` | 滚动文本摘要 |
| `paper_summary.txt` | 最终汇总 |
| `scoreboard.jsonl` | 绩效与诊断 |
| `daily/YYYY-MM-DD.log` | **永久**日度摘要（每周期一行）|

## Web 页面如何查看
Streamlit 主要包含：
- `Global Macro Signals`：宏观/主题/行业信号诊断
- `Portfolio Monitor`：净值、现金、持仓构成、交易历史、摘要

数据映射：
- 顶部指标与图表：`outputs/snapshot_live.json`
- 交易表：`outputs/trade_history.jsonl`
- 文本摘要：`outputs/paper_summary_live.txt`

## 关键调试命令

### GlobalWatch 侧
```bash
python GlobalWatch_V2.py --dump-industry-taxonomy --config paper_config.json
python GlobalWatch_V2.py --industry-sanity-check --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json
python GlobalWatch_V2.py --run-industry-one-bucket-debug chip --config paper_config.json --max_evidence 4 --llm_timeout_seconds 120 --output outputs/debug_chip_bucket.json
python GlobalWatch_V2.py --debug-industry-news --config paper_config.json --debug-outdir outputs/gw_industry_dryrun
```

### PaperTrading 侧
```bash
python paper_trading.py --debug-news-overlay-once --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --calibrate-news-overlay --lookback-hours 72 --target-cash-delta 0.02 --config paper_config.json --out outputs/news_overlay_calibration.json
python paper_trading.py --debug-news-overlay-phase2 --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --debug-system-s1-5 --debug-outdir outputs/gw_dryrun paper_config.json
```

### Quant Pack 执行阻塞归因
```bash
python scripts/quant/a19_compute_exec_blockers.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD
python scripts/quant/a20_attach_exec_blockers_to_daily.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD
```

## 关键配置速查（paper_config.json）

### `fx_rates`
- `CAD_USD`：CAD/USD 兜底汇率（默认 `0.73`）
- `auto_fetch`：是否实时拉取 yfinance 汇率（默认 `true`）
- `auto_fetch_symbol`：Yahoo Finance 代码（默认 `CADUSD=X`）
- `cache_minutes`：汇率缓存时间（默认 `60` 分钟）

### `execution`
- `signal_refresh_minutes`：信号刷新周期
- `macro_refresh_minutes`：宏观刷新周期
- `weight_threshold`：触发调仓阈值（mid: `0.015`）
- `min_trade_notional_usd`：最小交易金额（默认 `$400`）
- `max_turnover_pct_per_rebalance`：单轮换手上限（mid: `0.40`）
- `ramp_in_cycles`：新仓建仓周期数（mid: `2`）
- `min_holding_cycles`：最短持仓周期（mid: `4`）
- `rebalance_cooldown_minutes`：成功调仓后冷却时间
- `stop_loss_pct`：止损阈值（默认 `-0.08`）
- `portfolio_exposure_cap`：最大可投入比例（默认 `0.90`）

### `risk_model`（mid profile）
- `rc_limit`：组合风险集中度上限（`0.65`）
- `portfolio_cov_rc_hysteresis_band`：迟滞带宽（`0.02`）
- `portfolio_cov_rc_abort_buffer_enabled`：连续 abort 自动放宽（`true`）

### `news_overlay`
- `enabled`：行业新闻 overlay 开关
- `mode`: `symmetric`（双向））或 `risk_only`（仅减仓）
- `alpha`：overlay 敏感度（`0.6`）
- `min_confidence`：置信度下限（`0.45`）
- `max_abs_delta`：单桶权重最大变化（`0.05`）
- `max_age_hours`：新闻新鲜度窗口

### `macro_integration`
- `macro_allow_new_positions`：允许宏观信号开新仓的标的列表
- `cooldown_cycles`：宏观信号冷却周期数

### `risk_profiles`
每个 profile（`low`、`mid`、`high`、`ultra`）可覆盖：
- `rc_limit`、`max_weight_per_asset`、`min_cash_pct`、`max_turnover_pct_per_rebalance`、`weight_threshold`

### `universe`
- 78 个可交易标的：美股、加拿大 `.TO` 股票、ETF、板块 ETF、CASH
- 芯片篮子：`AMD QCOM AVGO TXN MRVL ARM`
- 半导体篮子：`TSM ASML LRCX KLAC AMAT MU ON`

## 验证命令
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

## 说明
- 仅用于模拟交易，不连接真实券商。
- 不构成投资建议。
- 加拿大 `.TO` 股票以 CAD 报价，所有组合价值均转换为 USD 进行记账。
