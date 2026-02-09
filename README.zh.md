# GlobalWatch Paper Trading (V2.11.3)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## 项目概览
GlobalWatch Paper Trading 是一个本地运行的量化策略仿真与执行验证框架。  
系统不连接真实券商，核心目标是让策略行为可观察、可审计、可迭代。

系统能力由三层组成：
- 行情与量化选股/权重逻辑
- Regime 与宏观主题信号叠加
- 执行风控与监控输出（供 Streamlit 使用）

## V2.11.3 版本更新
本次版本重点是执行规划器与审计输出增强，同时保持主策略流程稳定。

主要改动：
- 规划器评分量纲统一（`benefit` 与 `cost_weight` 可直接比较）
- 审计去重修复（同一笔交易不会同时出现在 `scaled` 与 `dropped`）
- 快照中新增更完整的规划器统计字段，便于调参与排障
- 引擎指纹版本更新为 `v2.11.3-2026-02-09`

## 功能说明
### 量化层
- 横截面排序与候选筛选
- 波动约束与权重分配
- 相关性过滤与分散化控制
- Regime 联动的现金/仓位管理
- 宏观主题信号倾斜
- 协方差诊断与波动目标能力接口

### 执行与风控层
- BUY/SELL 的陈旧报价策略
- stale 比例中止保护
- 单轮换手约束
- 熔断与 risk-off 去风险路径
- 强制交易优先的执行规划器
- 成本估算与结构化决策日志

### 系统层
- checkpoint 恢复与 fresh 启动
- 配置驱动的可复现实验
- 逐轮快照落盘
- 实时摘要与报告输出
- Streamlit 前端联动

## 启动方式
### 启动纸面交易引擎
```bash
python -u paper_trading.py paper_config.json
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
运行产物位于 `outputs/`：

- `snapshot_live.json`：前端实时状态主数据
- `trade_history.jsonl`：前端交易表数据
- `portfolio_snapshots.jsonl`：逐轮快照审计
- `paper_trades.csv`：执行导向交易日志
- `paper_summary_live.txt`：滚动文字摘要
- `paper_summary.txt`：最终汇总
- `scoreboard.jsonl`：滚动绩效诊断
- `equity_curve.png`：结束时净值曲线图

## Web 页面怎么看
Streamlit 里有两个主要页面：

- `Global Macro Signals`：宏观/主题信号观察与诊断
- `Portfolio Monitor`：净值、现金、持仓结构、交易历史、摘要信息

数据来源对应关系：
- 顶部指标和图表：`outputs/snapshot_live.json`
- 交易历史表：`outputs/trade_history.jsonl`（必要时可回退 `outputs/paper_trades.csv`）
- 文本摘要：`outputs/paper_summary_live.txt`

如果页面显示与引擎日志不一致，先确认 `snapshot_live.json` 是否仍在持续更新。

## 关键配置速查（`paper_config.json`）
这里只列参数用途，不公开核心阈值细节。

### execution
- `signal_refresh_minutes`：信号刷新周期
- `macro_refresh_minutes`：宏观刷新周期
- `weight_threshold`：调仓触发阈值
- `min_trade_notional_usd`：最小交易金额
- `max_turnover_pct_per_rebalance`：单轮换手上限
- `max_stale_ratio`：stale 中止阈值
- `price_stale_policy.allow_buy`：买入报价策略
- `price_stale_policy.allow_sell`：卖出报价策略
- `circuit_breaker_forced_days`：强制 risk-off 持续时长
- `fill_gap_max`：小缺口补足上限
- `fill_gap_max_iters`：补足迭代上限
- `allow_buy_benchmarks`：是否允许买入基准资产
- `cross_section_top_n`：横截面入选数量
- `correlation_lookback_days`：相关性回看窗口
- `correlation_threshold`：相关性过滤阈值
- `volatility_floor`：波动率下限
- `min_holding_cycles`：最短持有轮数
- `enable_exit_signals`：技术退出信号开关
- `exit_signal_lookback_days`：退出信号窗口
- `exit_on_gap_volume`：跳空放量退出开关
- `max_weight_boost_for_hot`：强势资产权重增量
- `hot_zscore_threshold`：强势资产 zscore 门槛
- `hot_momentum_top_k`：强势资产动量排名门槛
- `hot_persistence_cycles`：强势资产连续出现门槛

### macro_integration
- `enable_llm_topic_signals`：LLM 主题信号开关
- `topic_memory_window`：主题记忆窗口
- `llm_topic_confidence_threshold`：主题置信门槛
- `llm_topic_score_threshold`：主题强度门槛
- `llm_topic_tilt_scale`：主题倾斜缩放
- `macro_cash_slope`：宏观风险到现金映射斜率
- `tilt_max_delta`：单资产倾斜上限
- `macro_allow_new_positions`：risk-off 可新开仓白名单

### risk_model
- `enable_cov_diagnostics`：协方差诊断开关
- `shrinkage_alpha`：协方差收缩强度
- `annualization_factor`：年化系数
- `max_pair_corr_pairs`：输出相关性对数量
- `fallback_to_diag_on_error`：协方差失败回退策略
- `enable_vol_targeting`：波动目标开关
- `vol_target`：目标年化波动
- `vol_target_min_coverage`：最小覆盖率要求
- `vol_target_min_scale`：缩放下限
- `vol_target_max_scale`：缩放上限
- `vol_target_use_cov_only`：仅协方差口径缩放

### trade_planner
- `enable_trade_planner`：规划器总开关
- `allow_partial_fill`：是否允许部分成交
- `min_trade_notional`：规划器最小交易金额
- `enable_adv_limit`：ADV 容量约束开关
- `adv_limit_frac`：单笔参与率上限
- `adv_lookback_days`：ADV 回看窗口
- `adv_apply_to_forced`：强制交易是否应用 ADV 限制
- `enable_cost_sensitive_ranking`：成本敏感排序开关
- `lambda_cost`：成本惩罚强度
- `benefit_mode`：收益代理计算模式
- `max_audit_items`：审计列表最大条数

### cost_model
- `enabled`：成本估算开关
- `fee_bps`：手续费基点
- `slippage_bps`：滑点基点
- `impact_enabled`：冲击成本开关
- `impact_k`：冲击系数
- `adv_lookback_days`：参与率估算窗口

### reporting
- `trades_log_path`：交易 CSV 输出路径
- `portfolio_snapshots_path`：快照输出路径
- `summary_report_path`：汇总报告路径
- `scoreboard_path`：记分板路径
- `snapshot_live_path`：实时快照路径
- `trade_history_path`：前端交易历史路径

## 校验命令
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

可选快速冒烟：
```bash
python -u -c "from paper_trading import PaperTradingEngine; e=PaperTradingEngine('paper_config.json'); print('SMOKE_OK')"
```

## 说明
- 仅用于模拟交易。
- 不连接真实券商。
- 不构成投资建议。
