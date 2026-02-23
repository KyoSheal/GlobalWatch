# GlobalWatch Paper Trading (V3.2.4)

[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)
[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)

## 项目概览
GlobalWatch Paper Trading 是一个本地优先的量化研究与纸面交易系统。
它不连接真实券商，目标是让策略行为可观察、可审计、可复现、可迭代。

系统由三层组成：
- 量化层：横截面打分、组合构建、相关性与波动约束
- 执行层：stale 报价策略、换手预算、交易规划器、风控退出
- 系统层：checkpoint 恢复、快照落盘、Streamlit 可视化监控

## V3.2.4 更新摘要
本版本整合了你最近这批核心升级，并将引擎版本指纹提升到 `v3.2.4`。

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
- 新增“执行阻塞分布 + 无交易原因”量化链路：
  - `quant_packs/<date>/execution_blockers/exec_blockers.json`
  - `quant_packs/<date>/no_trade/no_trade.json`
  - 平铺日报写回：`quant_pack.execution_blockers`、`quant_pack.no_trade`
  - 索引同步字段：`exec_blocker_top1_reason`、`exec_blocked_ratio`、`no_trade_flag` 等

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
每条命令都附带“对应作用”，便于快速定位问题。

### 1) GlobalWatch 侧
1. `python GlobalWatch_V2.py --dump-industry-taxonomy --config paper_config.json`  
   对应：导出行业分类预览，检查 L2/L3 与 ticker 多标签映射是否正确。
2. `python GlobalWatch_V2.py --industry-sanity-check --config paper_config.json`  
   对应：读取最新行业信号做一致性核查（方向分布、偏置、关键 mismatch）。
3. `python GlobalWatch_V2.py --run-industry-runtime-once --config paper_config.json`  
   对应：真实跑一轮行业新闻管线（抓取→分桶→LLM→写入 `industry_signals`）。
4. `python GlobalWatch_V2.py --run-industry-runtime-once-debug --config paper_config.json --output outputs/industry_runtime_debug_latest.json`  
   对应：输出全链路调试信息（bucket items、raw/parsed/normalized、写入与读回对账）。
5. `python GlobalWatch_V2.py --run-industry-one-bucket-debug rates_and_gold --config paper_config.json --max_evidence 4 --llm_timeout_seconds 120 --output outputs/debug_industry_one_bucket.json`  
   对应：只调试单个行业桶，定位慢/卡/解析问题更快。
6. `python GlobalWatch_V2.py --debug-industry-news --config paper_config.json --debug-outdir outputs/gw_industry_dryrun`  
   对应：离线 dry-run 行业新闻链路验收，不依赖真实交易执行。

### 2) PaperTrading 侧
1. `python paper_trading.py --debug-news-overlay-once --debug-outdir outputs/gw_dryrun paper_config.json`  
   对应：单次验证新闻 overlay 消费（confidence 过滤、clip、risk_only、cash_target 变化）。
2. `python paper_trading.py --calibrate-news-overlay --lookback-hours 72 --target-cash-delta 0.02 --config paper_config.json --out outputs/news_overlay_calibration.json`  
   对应：基于历史信号分布给出 overlay 参数建议，并做回放模拟。
3. `python paper_trading.py --debug-news-overlay-phase2 --debug-outdir outputs/gw_dryrun paper_config.json`  
   对应：Phase2 消费侧确定性验收（内置用例，输出 PASS/FAIL）。
4. `python paper_trading.py --debug-system-s1-5 --debug-outdir outputs/gw_dryrun paper_config.json`  
   对应：系统级总体验收，快速确认主链路是否可运行。

### 3) Quant Pack 执行阻塞/无交易归因
1. `python scripts/quant/a19_compute_exec_blockers.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD`  
   对应：生成 cycle 级执行阻塞分布与 day-level 无交易原因产物。
2. `python scripts/quant/a20_attach_exec_blockers_to_daily.py --daily-base "outputs/Daily Report" --date YYYY-MM-DD`  
   对应：把 execution_blockers / no_trade 写回平铺日报 JSON（幂等覆盖）。

## 关键配置速查（paper_config.json）
这里只写用途，不展开具体阈值调参细节。

### `execution`
- `signal_refresh_minutes`：信号刷新周期。
- `macro_refresh_minutes`：宏观刷新周期。
- `weight_threshold`：触发调仓阈值。
- `min_trade_notional_usd`：最小交易金额。
- `max_turnover_pct_per_rebalance`：单轮换手上限。
- `max_stale_ratio`：stale 比例中止阈值。
- `price_stale_policy.allow_buy` / `price_stale_policy.allow_sell`：买卖允许的报价状态。
- `rebalance_cooldown_minutes` / `rebalance_attempt_cooldown_minutes`：调仓冷却与尝试冷却。
- `portfolio_exposure_cap`：组合最大可投入上限（默认 `0.90`）。
- `enable_target_cov_gate`：是否启用 target cov 作为风险闸门依据（默认 `false`）。
- `target_cov_gate_min_coverage`：启用 target gate 的最小覆盖率（默认 `0.60`）。
- `target_cov_gate_require_ok`：target cov 状态是否必须为 `ok`（默认 `true`）。
- `price_ttl_seconds`：价格缓存 TTL（默认 `45` 秒）。
- `price_batch_chunk_size`：批量拉价分块大小（默认 `50`）。
- `price_batch_allow_1m_fallback`：5m 缺失时是否允许 1m fallback（默认 `true`）。
- `enable_greedy_trade_filter`：是否启用贪心过滤执行计划（默认 `false`）。
- `min_trade_delta_w`：最小权重变化过滤阈值（默认 `0.002`）。
- `max_trades_per_cycle`：每轮最多执行交易数（默认 `25`）。
- `min_keep_trades`：soft trades 兜底保留数量（默认 `0`）。
- `cost_bps`：A4 过滤阶段成本估算 bps（默认 `0.0008`）。

### `trade_planner`
- `enable_trade_planner`：是否启用规划器。
- `allow_partial_fill`：预算不足时是否部分成交。
- `min_trade_notional`：规划器最小交易门槛。
- `enable_adv_limit`：是否启用 ADV 限制。
- `adv_limit_frac`：单笔最大参与率。
- `enable_cost_sensitive_ranking`：是否启用成本敏感排序。
- `lambda_cost`：成本惩罚强度。

### `news_overlay`
- `enabled`：行业新闻 overlay 开关。
- `mode`：默认 `risk_only`。
- `alpha`：overlay 敏感度。
- `min_confidence`：置信度下限。
- `max_abs_delta`：单桶影响截断。
- `enable_confidence_scaling`：置信度缩放开关。
- `max_age_hours`：新闻新鲜度窗口。
- `macro_risk_off_prior.enabled`：宏观 risk-off 先验开关。
- `macro_risk_off_prior.strength`：先验强度。
- `macro_risk_off_prior.min_score`：触发门槛。
- `macro_risk_off_prior.cooldown_minutes`：冷却分钟数。

### `risk_model`
- 协方差诊断参数（收益窗口、最小样本、收缩、覆盖阈值等）。
- 配合 target gate 使用时，建议重点观察 coverage 与 status 稳定性。

### `reporting`
- `snapshot_live_path`：Web UI 主快照路径。
- `trade_history_path`：交易历史路径。
- `scoreboard_path`：记分板路径。
- `max_price_debug_items`：价格调试字段上限（防止快照膨胀）。
- `telemetry_enabled` / `telemetry_fsync`：结构化 telemetry 开关与刷盘策略（若在配置中启用）。

### `cost_model`
- 手续费/滑点/冲击成本估算参数。

> 说明：部分新参数即使未显式写在 `paper_config.json`，引擎也会注入默认值；建议只覆盖你要调整的键。

## 验证命令
```bash
python -m py_compile paper_trading.py
python -m py_compile GlobalWatch_V2.py
```

## CI 依赖说明
CI 现已通过根目录 `requirements.txt` 安装基础依赖后再跑诊断：
- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`

## 说明
- 仅用于模拟交易。
- 不连接真实券商。
- 不构成投资建议。
