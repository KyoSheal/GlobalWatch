# GlobalWatch Paper Trading (V2.8+)

本项目是一个本地运行的自动化 Paper Trading 引擎（不连接真实券商）。
你最近新增的功能已经较多，本 README 重点覆盖：
- 如何运行
- 如何验收关键行为
- 关键配置项说明

## 1. 快速开始

### 1.1 环境
- Python 3.10+
- 依赖：`pandas` `numpy` `yfinance` `matplotlib`（以及你项目内其余依赖）

### 1.2 启动
```bash
python -u paper_trading.py paper_config.json
```

Windows 可直接用：
```bash
Start_Paper_Trading.bat
```

### 1.3 主要输出
- `outputs/paper_trades.csv`：成交记录（含 decision_trace）
- `outputs/portfolio_snapshots.jsonl`：每轮快照
- `outputs/scoreboard.jsonl`：2周窗口表现看板（新）
- `outputs/paper_summary_live.txt`：实时摘要
- `outputs/paper_summary.txt`：结束后总结

---

## 2. 当前引擎核心行为（已实现）

### 2.1 信号刷新解耦
- `execution.signal_refresh_minutes`（默认 `1440`）
- `execution.macro_refresh_minutes`（默认 `60`）

行为：
- 每个 cycle 都会 `record_snapshot()`
- 仅当到达宏观刷新间隔时，才拉取宏观信号
- 仅当到达信号刷新间隔时，才重算 target_weights
- 快照字段：
  - `weights_reused`
  - `macro_reused`
  - `last_signal_time`
  - `last_macro_time`

### 2.2 STALE 强风控
相关配置：
- `execution.max_stale_ratio`（默认 `0.3`）
- `execution.price_stale_policy.allow_buy`（默认 `LIVE/RECENT`）
- `execution.price_stale_policy.allow_sell`（默认 `LIVE/RECENT/STALE`）

行为：
- BUY 不允许 STALE（按 allow_buy）
- SELL 可允许 STALE（按 allow_sell）
- 候选交易 stale 比例超阈值时整轮 abort（`price_stale_abort=true`）

### 2.3 单轮换手上限（最终成交级别）
相关配置：
- `execution.max_turnover_pct_per_rebalance`

行为：
- 先算候选 `desired_trade_value`
- 超上限则整体按比例缩放
- 再换算整股并执行
- 快照/成交字段：
  - `turnover_notional_pre`
  - `turnover_notional_post`
  - `turnover_capped`
  - `turnover_scale`

### 2.4 Circuit Breaker 与 Regime 统一
相关配置：
- `execution.circuit_breaker_forced_days`（默认 `1`）

行为：
- 不再使用永久 `PAUSED`
- 改为 `risk_off_forced`，并设置 `forced_until_time`
- 强制风险状态下进行结构化减仓（按最差评分优先）
- Circuit breaker 交易同样写入完整上下文字段（含 `decision_trace` 中 `circuit_breaker`）

### 2.5 宏观双通路（5.12）

#### 通路1：风险雷达（影响现金）
- `cash_target = base_cash_from_regime + macro_cash_slope * macro_risk_score_smoothed + topic cash_add`
- clip 到 `[min_cash_from_regime, 0.60]`

#### 通路2：趋势放大器（影响权重上限/倾斜）
- 生成 `max_weight_per_asset_effective`
- `risk_off/risk_off_forced` 时，仅允许防守类倾斜（TLT/GLD/CASH）
- `macro_allow_new_positions` 控制风险状态下可新开仓资产（默认 `TLT/GLD`）

#### 小缺口补足（B 方案）
相关配置：
- `execution.fill_gap_max`（默认 `0.03`）
- `execution.fill_gap_max_iters`（默认 `2`）

行为：
- 先 cap
- 仅 overweight 时 downscale
- 若剩余缺口 `<= fill_gap_max`，在 headroom 内温和补足
- 若缺口更大，留现金，不强行买满

新增快照/trace指标：
- `invested_budget`
- `total_before_caps`
- `total_after_caps`
- `downscaled`
- `downscale_factor`
- `remaining_gap`
- `fill_gap_max`
- `fill_applied`
- `fill_amount`
- `fill_reason`
- `capped_assets`

### 2.6 Universe 与 Benchmarks 解耦
相关配置：
- `execution.allow_buy_benchmarks`（默认 `false`）

行为：
- `benchmarks.tickers` 只用于：
  - `compute_regime_state()`
  - `compute_benchmark_returns()`
- 在 `allow_buy_benchmarks=false` 下，打分/选权重会排除 benchmark ticker
- macro_tilt 也只对最终 trade_universe 生效；被过滤的 tilt 会记录日志

### 2.7 Scoreboard（新）
相关配置：
- `reporting.scoreboard_path`（默认 `outputs/scoreboard.jsonl`）

每次 `record_snapshot()` 后追加一行：
- `timestamp`
- `strategy_return_2w`（10交易日窗口）
- `bench_avg_return_2w`
- `excess_return_2w`
- `win_flag_2w`
- `turnover_sum_2w`
- `avg_cash_2w`
- `macro_active_ratio_2w`
- `diagnostic_hint`

自动诊断：
- 连续3个窗口 `win_flag_2w=false` 时触发提示，例如：
  - `turnover_too_high`
  - `too_defensive`
  - `macro_too_noisy`
  - `regime_filter_too_strict`
- 最新 `diagnostic_hint` 会同步写入 snapshot

---

## 3. 关键配置速查（paper_config.json）

### 3.1 execution
- `signal_refresh_minutes: 1440`
- `macro_refresh_minutes: 60`
- `weight_threshold: 0.025`
- `min_trade_notional_usd: 400`
- `max_turnover_pct_per_rebalance: 0.20`
- `max_stale_ratio: 0.3`
- `price_stale_policy.allow_buy: ["LIVE","RECENT"]`
- `price_stale_policy.allow_sell: ["LIVE","RECENT","STALE"]`
- `circuit_breaker_forced_days: 1`
- `fill_gap_max: 0.03`
- `fill_gap_max_iters: 2`
- `allow_buy_benchmarks: false`

### 3.2 macro_integration
- `macro_cash_slope: 0.02`
- `tilt_max_delta: 0.02`
- `macro_allow_new_positions: ["TLT", "GLD"]`

### 3.3 reporting
- `trades_log_path`
- `portfolio_snapshots_path`
- `summary_report_path`
- `scoreboard_path`

---

## 4. 建议测试清单

### 4.1 启动前快速检查
```bash
python -m py_compile paper_trading.py
```

### 4.2 运行后检查文件是否增长
```bash
# PowerShell
Get-Content outputs\portfolio_snapshots.jsonl -Tail 3
Get-Content outputs\paper_trades.csv -Tail 5
Get-Content outputs\scoreboard.jsonl -Tail 5
```

### 4.3 重点验收项
1. 连续两轮间隔小于 `signal_refresh_minutes`：
- `weights_reused=true`
- `target_weights` 不变

2. 强制 STALE 场景：
- 出现 `price_stale_abort=true` 或大量 `stale_price_skip`

3. 超换手场景：
- `turnover_notional_post <= turnover_limit`

4. 触发回撤熔断：
- 不进入 `PAUSED`
- 进入 `risk_off_forced`
- 有结构化减仓交易与日志

5. `allow_buy_benchmarks=false`：
- `target_weights` 不应自动出现 `QQQ/SPY/VTI/DIA`

6. 连续落后窗口：
- `scoreboard.jsonl` 出现 `diagnostic_hint`
- snapshot 同步该字段

---

## 5. 常见问题

### Q1: 启动后提示 checkpoint，是否继续？
- 想续跑历史会话：输入 `y`
- 想重新开始：输入 `n`

### Q2: 为什么有些宏观 tilt 没生效？
- 可能被 `trade_universe` 过滤（benchmark 解耦）
- 可能在 `risk_off` 下被判定为进攻型 tilt 而阻断

### Q3: 为什么资金没有完全投满？
- 这是设计行为：大缺口（>`fill_gap_max`）保留现金，避免强行买满导致过度冒险

---

## 6. 安全声明

- 本系统仅用于模拟交易（Paper Trading）
- 不连接真实券商，不构成投资建议
- 实盘前请做独立风险评估与充分回测

