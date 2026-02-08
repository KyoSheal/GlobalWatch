# GlobalWatch Paper Trading (V2.10.5)

[![CN](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-red)](./README.zh.md)
[![EN](https://img.shields.io/badge/Language-English-blue)](./README.en.md)

## 1. 项目简介
- 本项目是本地自动化 Paper Trading 引擎，不连接真实券商。
- 目标是进行策略迭代、执行风控验证与可解释日志追踪。
- 文档公开运行与验收方法，不披露核心阈值设计细节。

## 2. 版本变化（v2.9.1 -> v2.10.5）
### 2.1 主要新增能力
- Step 1: Alpha 加入行业强度 + 成交量因子。
- Step 2: GlobalWatch 增加主题信号记忆与准确率自适应权重。
- Step 3: 新增跳空下跌 + 量能 Z 分数的智能强制退出。
- Step 4: 新增强势股自适应加仓（Top 动量 + Z 分数 + 连续出现）。
- Step 5: 保留并强化 STALE 风控与整轮中止机制。
- Step 6: 保留换手上限，对最终可执行交易生效。
- Step 7: 保留组合级风控（波动率/集中度风险闸门）。
- Step 8: 保留 Regime + Macro 双通路整合与结构化日志。

### 2.2 本次（Step 4）核心点
- 满足 `zscore > hot_zscore_threshold` 且位于 `top-k momentum` 的资产，单票 cap 可上调 `max_weight_boost_for_hot`。
- 支持连续出现轮数过滤（`hot_persistence_cycles`）。
- 总投资预算硬限制为 90%，避免整体过曝险。
- 日志新增 `[HOT BOOST]`，标注哪些资产被上调。

## 3. 快速开始
### 3.1 运行
```bash
python -u paper_trading.py paper_config.json
```

Windows:
```bash
Start_Paper_Trading.bat
```

### 3.2 主要输出
- `outputs/paper_trades.csv`：交易与执行轨迹。
- `outputs/portfolio_snapshots.jsonl`：每轮快照。
- `outputs/scoreboard.jsonl`：滚动表现看板。
- `outputs/paper_summary_live.txt`：运行时摘要。
- `outputs/paper_summary.txt`：结束总结。

## 4. 关键配置速查（paper_config.json）
### 4.1 execution（新增/关键）
- `max_weight_boost_for_hot` # 强势股cap增量
- `hot_zscore_threshold` # 强势阈值
- `hot_momentum_top_k` # 动量前k
- `hot_persistence_cycles` # 连续轮数
- `exit_on_gap_volume` # 跳空量能退出开关
- `exit_gap_down_pct` # 跳空阈值
- `exit_gap_volume_zscore` # 量能z阈值
- `exit_gap_volume_window` # 量能窗口
- `max_turnover_pct_per_rebalance` # 单轮换手上限
- `max_stale_ratio` # stale中止阈值
- `signal_refresh_minutes` # 信号刷新间隔
- `macro_refresh_minutes` # 宏观刷新间隔

### 4.2 macro_integration（关键）
- `enable_llm_topic_signals` # 主题信号开关
- `topic_memory_window` # 主题记忆窗口
- `llm_topic_confidence_threshold` # 主题置信门槛
- `llm_topic_score_threshold` # 主题强度门槛
- `llm_topic_tilt_scale` # 主题倾斜系数

## 5. 验收建议
1. 运行两轮以上，检查是否出现 `[HOT BOOST]` 日志。
2. 观察 `allocation_diagnostics` 中 `hot_boost_assets` 与 `portfolio_exposure_cap`。
3. 模拟跳空+放量场景，确认触发 `force_exit`。
4. 检查 `portfolio_snapshots.jsonl` 中风险闸门与执行字段是否完整。

## 6. 中文乱码排查
- 中文乱码通常是终端编码问题，不是时区问题。
- 若 GitHub 显示正常而本地终端乱码，请先执行：
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 7. 安全声明
- 仅用于模拟交易。
- 不连接真实券商。
- 不构成投资建议。
