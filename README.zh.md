# GlobalWatch Paper Trading (V2.10.1)

[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)
[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)

## 1. 项目简介
- 本项目是本地自动化 Paper Trading 引擎，不连接真实券商
- 目标是在可控风险下进行策略迭代与行为验收
- 本文档公开运行方式和验收路径，不公开核心阈值与专有算法细节

## 2. 版本对比（相对上一版）
### 2.1 从 `v2.9.1` 到 `v2.10.1` 的主要新增
- Step 1：现金利用策略升级，支持高置信信号临时降低现金目标（仍保留最低现金底线）
- Step 2：动量从单周期升级为多周期融合，改善早期趋势识别
- Step 3：新增高置信单标的权重提升机制（仅单票，且受总风险约束）
- Step 4：新增持仓退出信号模块（价量形态），支持减仓或清仓模式
- Step 5：GlobalWatch 增加结构化主题信号，支持可选 LLM 主题注入链路
- Step 6：新增分数平滑/标准化/截断与组合级风险闸门（波动率与集中度）

### 2.2 延续并强化的模块
- 信号刷新解耦（signal 与 macro 分开刷新）
- STALE 强风控（买入价格策略 + stale 比例整轮中止）
- 单轮换手上限（对最终可执行交易生效）
- Circuit breaker 与 regime 一体化（`risk_off_forced`）
- Universe 与 benchmarks 解耦
- `scoreboard.jsonl` 滚动看板与自动诊断

## 3. 快速开始
### 3.1 环境
- Python 3.10+
- 主要依赖：`pandas` `numpy` `yfinance` `matplotlib`

### 3.2 启动
```bash
python -u paper_trading.py paper_config.json
```

Windows 可用：
```bash
Start_Paper_Trading.bat
```

### 3.3 关键输出文件
- `outputs/paper_trades.csv`：交易与执行轨迹
- `outputs/portfolio_snapshots.jsonl`：每轮快照（核心验收文件）
- `outputs/scoreboard.jsonl`：滚动窗口绩效和诊断提示
- `outputs/paper_summary_live.txt`：运行中的实时摘要
- `outputs/paper_summary.txt`：结束后总结

## 4. 核心行为总览
### 4.1 刷新解耦
- 每个 cycle 都记录 snapshot
- 宏观信号按宏观刷新间隔更新
- 权重按信号刷新间隔更新
- 快照可直接查看复用状态与上次刷新时间

### 4.2 执行层风控
- 买入不接受 STALE 报价（按策略白名单）
- 候选交易 stale 比例超过阈值，整轮中止
- 换手上限按最终可执行名义金额生效，不是只统计不约束

### 4.3 组合层风控（v2.10.1）
- 分数稳定化：可选平滑、标准化、极值截断
- 波动风险闸门：组合加权波动超过阈值时拦截调仓
- 分散度闸门：集中度（HHI）超过阈值时拦截调仓
- 风险闸门结果写入 snapshot，可追溯

## 5. 关键配置速查（`paper_config.json`）
说明：仅描述用途，不披露核心参数设计依据。

### 5.1 `execution`
- `signal_refresh_minutes` # 信号刷新间隔
- `macro_refresh_minutes` # 宏观刷新间隔
- `weight_threshold` # 权重变更门槛
- `min_trade_notional_usd` # 最小下单名义
- `max_turnover_pct_per_rebalance` # 单轮换手上限
- `max_stale_ratio` # stale 中止阈值
- `price_stale_policy.allow_buy` # 买入报价策略
- `price_stale_policy.allow_sell` # 卖出报价策略
- `circuit_breaker_forced_days` # 强制风控时长
- `fill_gap_max` # 小缺口补足上限
- `fill_gap_max_iters` # 缺口补足迭代
- `allow_buy_benchmarks` # 基准可买开关
- `cross_section_top_n` # 横截面选取数
- `correlation_lookback_days` # 相关性回看窗
- `correlation_threshold` # 相关性阈值
- `volatility_floor` # 波动率下限
- `min_holding_cycles` # 最小持有轮数
- `allow_high_conviction_override` # 高置信现金覆盖
- `enable_high_conviction_weighting` # 高置信加权开关
- `max_high_conviction_weight` # 高置信单票上限
- `enable_short_term_momentum` # 短周期动量开关
- `short_momentum_lookback_days` # 短周期回看窗
- `enable_exit_signals` # 退出信号开关
- `exit_signal_lookback_days` # 退出信号窗口
- `enable_score_smoothing` # 分数平滑开关
- `score_smoothing_window` # 平滑窗口长度
- `max_portfolio_volatility` # 组合波动上限
- `enable_diversity_check` # 分散度检查开关
- `max_herfindahl_index` # 集中度上限
- `portfolio_vol_min_coverage` # 波动覆盖要求

### 5.2 `macro_integration`
- `macro_cash_slope` # 宏观现金斜率
- `tilt_max_delta` # 倾斜幅度上限
- `macro_allow_new_positions` # 风险期可开仓白名单
- `enable_llm_topic_signals` # LLM 主题注入开关
- `llm_topic_confidence_threshold` # 主题置信门槛
- `llm_topic_score_threshold` # 主题强度门槛
- `llm_topic_tilt_scale` # 主题倾斜系数

### 5.3 `reporting`
- `trades_log_path` # 成交日志路径
- `portfolio_snapshots_path` # 快照日志路径
- `summary_report_path` # 总结报告路径
- `scoreboard_path` # 看板日志路径

## 6. 运行与验收建议
### 6.1 启动前检查
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

### 6.2 运行中检查
```powershell
Get-Content outputs\portfolio_snapshots.jsonl -Tail 3
Get-Content outputs\paper_trades.csv -Tail 5
Get-Content outputs\scoreboard.jsonl -Tail 5
```

### 6.3 验收重点
1. 刷新解耦是否生效（复用标记与时间字段）
2. STALE 场景是否触发 skip/abort
3. 换手上限是否对最终成交生效
4. 风险闸门是否在高波动/高集中场景下拦截
5. 熔断后是否进入 `risk_off_forced` 而非永久暂停

## 7. 中文乱码排查（重要）
- 中文乱码通常由终端编码导致，不是时区导致
- GitHub 页面显示正常但本地终端乱码，优先检查代码页
- PowerShell 建议：
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
- 若仍有乱码，检查编辑器文件编码为 UTF-8

## 8. 安全声明
- 仅用于模拟交易
- 不连接真实券商
- 不构成投资建议
