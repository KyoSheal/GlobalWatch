# GlobalWatch Paper Trading (V3.2.2)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## 项目概览
GlobalWatch Paper Trading 是一个本地优先的量化研究与纸面交易系统。
它不连接真实券商，目标是让策略行为可观察、可审计、可复现、可迭代。

系统由三层组成：
- 量化层：横截面打分、组合构建、相关性与波动约束
- 执行层：stale 报价策略、换手预算、交易规划器、风控退出
- 系统层：checkpoint 恢复、快照落盘、Streamlit 可视化监控

## V3.2.2 更新摘要
本版本整合了你最近这批核心升级，并将引擎版本指纹提升到 `v3.2.2`。

### 信号与行业管线
- 新增 L2/L3 多层多标签体系：`industry_taxonomy` + `ticker_tags`
- 行业新闻管线修复“分桶前全局截断”问题，避免桶被饿死
- 加入 seed-aware 分桶，降低行业桶污染
- 行业信号改为“证据标签 -> 确定性打分”路径，输出更可解释
- 增加宏观 risk-off prior（可开关、可门槛、可 cooldown）
- LLM 输出解析增加 schema 校验、空输出保护、fallback 兜底

### 交易侧与 Overlay
- PaperTrading 已消费 `industry_signals` 并可作用到 cash target overlay
- `min_confidence`、`risk_only`、`max_abs_delta`、confidence scaling 全部可控
- 新增一次性 overlay 诊断与校准回放工具，便于参数落地
- 协方差风险诊断链路保留并写入快照审计字段

### 执行与风控
- stale 策略、会话时段 gate、换手预算路径保持有效
- post-rebalance 快照刷新更及时，UI 持仓同步更快
- planner/cost/overlay 调试字段更完整，便于逐轮回溯

### UI 与交互
- Macro/FX 支持“仅分析当前选中目标”的轻量交互模式
- 新增可选开关 `Request <think>`，默认仍走快速 JSON
- 若模型无 `<think>`，UI 改为显示 `Reasoning Summary`，不再误导
- Equity 曲线支持交易时段可视化与轴控
- Trade History 在 Web UI 侧按“最新在前”展示（不改底层写入顺序）

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
主要运行产物在 `outputs/`：
- `snapshot_live.json`：前端卡片/图表主数据
- `trade_history.jsonl`：前端交易历史表
- `portfolio_snapshots.jsonl`：逐轮快照审计
- `paper_trades.csv`：成交导向交易流水
- `paper_summary_live.txt`：滚动文本摘要
- `paper_summary.txt`：最终汇总
- `scoreboard.jsonl`：绩效与诊断

## Web 页面如何查看
Streamlit 主要包含：
- `Global Macro Signals`：宏观/主题/行业信号诊断
- `Portfolio Monitor`：净值、现金、持仓构成、交易历史、摘要

数据映射关系：
- 顶部指标与图表：`outputs/snapshot_live.json`
- 交易表：`outputs/trade_history.jsonl`（必要时回退 `outputs/paper_trades.csv`）
- 文本摘要：`outputs/paper_summary_live.txt`

## 关键调试命令
### GlobalWatch
```bash
python GlobalWatch_V2.py --dump-industry-taxonomy --config paper_config.json
python GlobalWatch_V2.py --industry-sanity-check --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json
python GlobalWatch_V2.py --run-industry-runtime-once-debug --config paper_config.json --output outputs/industry_runtime_debug_latest.json
python GlobalWatch_V2.py --run-industry-one-bucket-debug rates_and_gold --config paper_config.json --max_evidence 4 --llm_timeout_seconds 120 --output outputs/debug_industry_one_bucket.json
python GlobalWatch_V2.py --debug-industry-news --config paper_config.json --debug-outdir outputs/gw_industry_dryrun
```

### PaperTrading
```bash
python paper_trading.py --debug-news-overlay-once --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --calibrate-news-overlay --lookback-hours 72 --target-cash-delta 0.02 --config paper_config.json --out outputs/news_overlay_calibration.json
python paper_trading.py --debug-news-overlay-phase2 --debug-outdir outputs/gw_dryrun paper_config.json
python paper_trading.py --debug-system-s1-5 --debug-outdir outputs/gw_dryrun paper_config.json
```

## 关键配置速查（paper_config.json）
这里只写用途，不展开你策略阈值细节。

### `execution`
- `signal_refresh_minutes`：信号刷新周期
- `macro_refresh_minutes`：宏观刷新周期
- `weight_threshold`：触发调仓阈值
- `min_trade_notional_usd`：最小交易金额
- `max_turnover_pct_per_rebalance`：单轮换手上限
- `max_stale_ratio`：stale 比例中止阈值
- `price_stale_policy.allow_buy`：买入允许报价状态
- `price_stale_policy.allow_sell`：卖出允许报价状态
- `rebalance_cooldown_minutes`：成功后冷却
- `rebalance_attempt_cooldown_minutes`：尝试级冷却

### `trade_planner`
- `enable_trade_planner`：是否启用规划器
- `allow_partial_fill`：预算不足时是否部分成交
- `min_trade_notional`：规划器最小交易门槛
- `enable_adv_limit`：是否启用 ADV 限制
- `adv_limit_frac`：单笔最大参与率
- `enable_cost_sensitive_ranking`：是否启用成本敏感排序
- `lambda_cost`：成本惩罚强度

### `news_overlay`
- `enabled`：行业新闻 overlay 开关
- `mode`：默认 `risk_only`
- `alpha`：overlay 敏感度
- `min_confidence`：置信度下限
- `max_abs_delta`：单桶影响截断
- `enable_confidence_scaling`：置信度缩放开关
- `max_age_hours`：新闻新鲜度窗口
- `macro_risk_off_prior.enabled`：宏观 risk-off 先验开关
- `macro_risk_off_prior.strength`：先验强度
- `macro_risk_off_prior.min_score`：触发门槛
- `macro_risk_off_prior.cooldown_minutes`：冷却分钟数

### `risk_model`
- 协方差诊断与波动目标参数（可选）

### `cost_model`
- 手续费/滑点/冲击成本估算参数

## 验证命令
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

## 说明
- 仅用于模拟交易。
- 不连接真实券商。
- 不构成投资建议。
