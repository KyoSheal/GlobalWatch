# GlobalWatch Paper Trading (V3.1.2)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## 项目概览
GlobalWatch Paper Trading 是一个本地优先的量化研究与模拟执行系统。  
系统不连接真实券商，目标是让策略行为可观察、可审计、可复现、可迭代。

系统能力主要由三层组成：
- 量化层：横截面排序、仓位分配、相关性过滤、风险约束
- 执行层：陈旧报价策略、换手限制、交易规划器、强制去风险路径
- 系统层：checkpoint 恢复、快照输出、前端监控联动、结构化审计数据

## V3.1.2 更新摘要（System + Quant）
本版本完成 S1-S5 的系统化升级，核心变化如下。

### System 更新
- 新增市场会话感知 gate（market-session aware gate）：闭市和开盘缓冲阶段不执行 rebalance
- 新增 rebalance attempt cooldown：失败/中止后避免每轮空转重试
- 输出改为原子写：`snapshot_live.json` 与 `trade_history.jsonl` 避免半截读取
- 运行身份增强：快照与交易记录写入 `session_id`、`config_hash`
- 新增离线可重复的 S1-S5 自动 dry-run 验收入口（PASS/FAIL + 退出码）

### Quant 更新
- stale ratio 统计口径修正为仅统计 policy-pass 的可交易候选
- stale-abort 仅允许在 OPEN 且通过 open grace 后触发
- 增强 `price_debug` 可解释性：`source`、`price_ts`、`tz_ok`、阈值与状态依据

### 引擎版本
- ENGINE_VERSION 更新为：`v3.1.2-2026-02-10`

## 快速启动
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
运行产物默认在 `outputs/`：
- `snapshot_live.json`：前端实时状态主数据源
- `trade_history.jsonl`：前端交易历史表数据
- `portfolio_snapshots.jsonl`：逐轮组合快照（审计）
- `paper_trades.csv`：交易流水
- `paper_summary_live.txt`：滚动文字摘要
- `paper_summary.txt`：结束后的汇总报告
- `scoreboard.jsonl`：滚动绩效诊断
- `equity_curve.png`：净值曲线图

## Web 页面怎么看
Streamlit 主要包含两个页面：
- `Global Macro Signals`：宏观/主题信号观察与诊断
- `Portfolio Monitor`：净值、现金、持仓结构、交易历史与摘要

数据来源对应关系：
- 顶部指标与图表：`outputs/snapshot_live.json`
- 交易历史表：`outputs/trade_history.jsonl`（必要时可回退 `outputs/paper_trades.csv`）
- 文本摘要：`outputs/paper_summary_live.txt`

如果页面显示与引擎日志不一致，先确认 `snapshot_live.json` 是否仍在持续更新。

## 关键配置速查（paper_config.json）
这里只列参数用途，不展开策略阈值细节。

### execution
- `signal_refresh_minutes`：信号刷新周期
- `macro_refresh_minutes`：宏观刷新周期
- `weight_threshold`：触发调仓的权重偏离阈值
- `min_trade_notional_usd`：最小交易金额
- `max_turnover_pct_per_rebalance`：单轮换手上限
- `max_stale_ratio`：stale 比例中止阈值
- `price_stale_policy.allow_buy`：买入允许的报价状态
- `price_stale_policy.allow_sell`：卖出允许的报价状态
- `rebalance_cooldown_minutes`：成功调仓后的冷却
- `rebalance_attempt_cooldown_minutes`：尝试级冷却（含失败/中止）

### trade_planner
- `enable_trade_planner`：是否启用规划器
- `allow_partial_fill`：预算不足时是否允许最后一笔部分缩放
- `min_trade_notional`：规划器最小交易金额
- `enable_cost_sensitive_ranking`：是否启用成本敏感排序
- `lambda_cost`：成本惩罚强度
- `benefit_mode`：收益代理模式

### reporting
- `snapshot_live_path`：实时快照路径
- `trade_history_path`：交易历史 JSONL 路径
- `trades_log_path`：交易 CSV 路径
- `daily_report_dirs`：日报输出目录列表
- `max_price_debug_items`：每轮写入 price_debug 的 ticker 上限

## 验证命令
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

S1-S5 离线自动验收（推荐）：
```bash
python paper_trading.py --debug-system-s1-5 --debug-outdir /tmp/gw_dryrun
```

或用环境变量触发：
```bash
GW_DEBUG_SYSTEM_S1_5=1 python paper_trading.py --debug-outdir /tmp/gw_dryrun
```

## 说明
- 仅用于模拟交易
- 不连接真实券商
- 不构成投资建议
