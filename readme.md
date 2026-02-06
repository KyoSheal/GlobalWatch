# GlobalWatch Paper Trading (V2.9.1)

本项目是一个本地运行的自动化 Paper Trading 引擎（不连接真实券商）。
你最近新增的功能已经较多，本 README 重点覆盖：
- 如何运行
- 如何验收关键行为
- 关键配置项说明
- v2.9.1 新增 Alpha 模块（横截面排名/波动缩放/相关性去重/最小持有期）

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
- `execution.signal_refresh_minutes`
- `execution.macro_refresh_minutes`

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
- `execution.max_stale_ratio`
- `execution.price_stale_policy.allow_buy`
- `execution.price_stale_policy.allow_sell`

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
- `execution.circuit_breaker_forced_days`

行为：
- 不再使用永久 `PAUSED`
- 改为 `risk_off_forced`，并设置 `forced_until_time`
- 强制风险状态下进行结构化减仓
- Circuit breaker 交易同样写入完整上下文字段（含 `decision_trace` 中 `circuit_breaker`）

### 2.5 宏观双通路（5.12）

#### 通路1：风险雷达（影响现金）
- 现金目标由多源风险信号联合驱动
- 现金目标始终受到风险边界约束

#### 通路2：趋势放大器（影响权重上限/倾斜）
- 生成 `max_weight_per_asset_effective`
- `risk_off/risk_off_forced` 时，仅允许防守类倾斜（TLT/GLD/CASH）
- `macro_allow_new_positions` 控制风险状态下可新开仓资产

#### 缺口补足（B 方案）
相关配置：
- `execution.fill_gap_max`
- `execution.fill_gap_max_iters`

行为：
- 先执行上限约束与预算约束
- 在小缺口场景进行有限补足
- 在大缺口场景保留现金，不强行买满

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
- `execution.allow_buy_benchmarks`

行为：
- `benchmarks.tickers` 只用于：
  - `compute_regime_state()`
  - `compute_benchmark_returns()`
- 可配置是否允许基准资产参与交易池
- macro_tilt 也只对最终 trade_universe 生效；被过滤的 tilt 会记录日志

### 2.7 Scoreboard
相关配置：
- `reporting.scoreboard_path`

每次 `record_snapshot()` 后追加一行：
- `timestamp`
- `strategy_return_2w`
- `bench_avg_return_2w`
- `excess_return_2w`
- `win_flag_2w`
- `turnover_sum_2w`
- `avg_cash_2w`
- `macro_active_ratio_2w`
- `diagnostic_hint`

自动诊断：
- 连续落后时触发提示，例如：
  - `turnover_too_high`
  - `too_defensive`
  - `macro_too_noisy`
  - `regime_filter_too_strict`
- 最新 `diagnostic_hint` 会同步写入 snapshot

### 2.8 Alpha 升级（v2.9.1）
说明：
- 已启用新版选股与权重引擎，包含多层信号筛选、风险约束与交易稳定机制
- 算法实现细节、内部阈值与参数取值不在公开文档披露
- 日志中保留必要的运行状态标签用于调试，不披露核心策略细节

---

## 3. 关键配置速查（paper_config.json）

### 3.1 execution
- `signal_refresh_minutes`
- `macro_refresh_minutes`
- `weight_threshold`
- `min_trade_notional_usd`
- `max_turnover_pct_per_rebalance`
- `max_stale_ratio`
- `price_stale_policy.allow_buy`
- `price_stale_policy.allow_sell`
- `circuit_breaker_forced_days`
- `fill_gap_max`
- `fill_gap_max_iters`
- `allow_buy_benchmarks`
- `cross_section_top_n`
- `correlation_lookback_days`
- `correlation_threshold`
- `volatility_floor`
- `min_holding_cycles`

### 3.2 macro_integration
- `macro_cash_slope`
- `tilt_max_delta`
- `macro_allow_new_positions`

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
1. 信号缓存与复用：
- 快照可看到权重/宏观复用状态
- 复用场景下目标权重保持稳定

2. 强制 STALE 场景：
- 出现 STALE 防护相关日志与快照标记

3. 超换手场景：
- 出现换手限制生效的相关日志与快照标记

4. 触发回撤熔断：
- 不进入 `PAUSED`
- 进入 `risk_off_forced`
- 有结构化减仓交易与日志

5. Universe / Benchmark 解耦：
- 基准资产是否交易应与配置保持一致

6. 连续落后窗口：
- `scoreboard.jsonl` 出现 `diagnostic_hint`
- snapshot 同步该字段

7. Alpha 模块状态：
- 新版 Alpha 相关日志标签可见
- 快照中可看到对应状态字段

---

## 5. 常见问题

### Q1: 启动后提示 checkpoint，是否继续？
- 想续跑历史会话：输入 `y`
- 想重新开始：输入 `n`
- 非交互场景可设置环境变量：`GW_CHECKPOINT_ACTION=resume|fresh`

### Q2: 为什么有些宏观 tilt 没生效？
- 可能被 `trade_universe` 过滤（benchmark 解耦）
- 可能在 `risk_off` 下被判定为进攻型 tilt 而阻断

### Q3: 为什么资金没有完全投满？
- 这可能是风控与资金管理机制的设计行为，公开文档不披露内部判定细节

---

## 6. 安全声明

- 本系统仅用于模拟交易（Paper Trading）
- 不连接真实券商，不构成投资建议
- 实盘前请做独立风险评估与充分回测


