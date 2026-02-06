# 🦁 GlobalWatch: AI-Powered Financial Intelligence Terminal
# 本地化 AI 金融情报终端

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Ollama](https://img.shields.io/badge/AI-Ollama%20Local-black)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-V2.5-orange)

[English](#-english-version) | [中文文档](#-中文文档-chinese-version)

---

# 🇬🇧 English Version

**GlobalWatch V2.5** is a privacy-first, real-time financial intelligence terminal that runs entirely on your local machine. 

By combining **Live RSS Feeds** (Reuters, CNBC, BBC) with **Local Large Language Models** (via Ollama), it performs autonomous market analysis, trend detection, and risk assessment without sending sensitive data to the cloud or paying for API keys.

## 🚀 What's New in V2.5

### 📈 Advanced Paper Trading System
- **48-Hour Simulation**: Automated paper trading with real-time price updates
- **Multi-Layer Strategy Engine**:
  - **Momentum + Volatility Scoring**: Dynamic asset evaluation
  - **Regime Filter**: MA50-based market state detection (risk_on/neutral/risk_off)
  - **Macro Integration**: Connects to GlobalWatch ChromaDB for macro signals
  - **Dynamic Risk Adjustment**: Auto-adjusts cash and position limits based on market conditions
- **Real-Time Monitoring**: Live summary updates, trade logs, and P&L tracking
- **Resume from Checkpoint**: Interrupt and resume trading sessions seamlessly
- **19-Asset Universe**: ETFs (SPY, QQQ, DIA, IWM, GLD, TLT) + Tech stocks (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA) + Defensive stocks (JPM, XOM, JNJ, PG, KO)
- **Advanced Risk Controls**: 
  - Cooldown protection (12-hour minimum between rebalances)
  - Weight threshold filtering (2.5% minimum change)
  - Minimum notional filtering ($400 minimum trade size)
  - Turnover cap (20% max portfolio turnover per rebalance)
  - Price freshness checks (STALE price protection)
  - Max drawdown limits, position sizing, transaction costs
- **Comprehensive Reporting**: 
  - Benchmark comparison (vs QQQ, SPY, VTI, DIA)
  - Regime state tracking
  - Macro signal integration
  - Detailed trade reasoning logs
  - Equity curves, performance metrics

### 🎯 Regime Filter System
- **MA50 Trend Analysis**: Monitors 4 benchmark indices (QQQ, SPY, VTI, DIA)
- **Dynamic State Detection**: 
  - 🟢 **Risk_On**: ≥75% indices above MA50 → Min cash 10%, Max weight 25%
  - 🟡 **Neutral**: 50-75% above MA50 → Min cash 20%, Max weight 25%
  - 🔴 **Risk_Off**: ≤50% above MA50 → Min cash 35%, Max weight 20%
- **Automatic Risk Adjustment**: Portfolio constraints adapt to market conditions

### 🌐 Macro Signal Integration
- **GlobalWatch Connection**: Reads trading signals from ChromaDB
- **Signal Age Filter**: Only considers signals within 48 hours (configurable)
- **Recent Signal Selection**: Takes only the most recent N signals per theme (default: 3)
- **Direction Counting**: Counts signals by direction (not weighted voting)
- **Confirmation Rules**: Requires k out of n same-direction signals (default: 2/3)
- **Time-Decay Weighting**: Applied after confirmation for strength calculation only
- **Theme Voting**: Aggregates signals by theme (oil_bullish, risk_off, usd_strong, etc.)
- **Risk Scoring**: 0-10 scale, higher = more risk-off
- **Signal Smoothing**: Median or EWMA smoothing over last 3 cycles to prevent whipsaws
- **Macro Cooldown**: 2-cycle cooldown after significant cash target changes (>5%)
- **Asset Tilts**: Applies macro-driven weight adjustments (±2% max per asset, universe-filtered)
- **Cash Adjustment**: Increases cash allocation based on smoothed macro risk score

### 📝 Trade Reasoning Logs
Every trade now includes complete context:
- **Regime State**: Market condition at trade time
- **Trend Score**: % of indices above MA50
- **Cash Target**: Dynamic minimum cash requirement
- **Macro Risk Score**: GlobalWatch risk assessment (0-10, smoothed)
- **Macro Topics**: Confirmed themes (e.g., "oil_bullish:bullish; risk_off:bearish")
- **Macro Tilts**: Active asset tilts (e.g., "XOM:+2.00%; TLT:+2.00%")
- **Price Status**: Data freshness (LIVE/RECENT/STALE) and age in minutes
- **Decision Trace**: Execution path (e.g., "cooldown_pass | weight_threshold_pass | min_notional_pass | stale_check_pass | turnover_cap_scale_85% | macro_tilt_+2.00% | risk_on_add-risk")

### 🛡️ Five-Layer Protection System
The paper trading engine implements five critical protection layers to prevent execution errors:

1. **Cooldown Protection**: Minimum 12-hour interval between rebalances to prevent overtrading
2. **Weight Threshold Filter**: Only trades positions with ≥2.5% weight change to reduce noise
3. **Minimum Notional Filter**: Enforces $400 minimum trade size to avoid dust trades
4. **Turnover Cap**: Limits total portfolio turnover to 20% per rebalance, scales down trades proportionally if exceeded
5. **Price Freshness Guard**: 
   - Blocks BUY orders on STALE prices (any age)
   - Allows SELL orders on STALE prices (risk reduction)
   - Aborts entire rebalance if >30% of candidate tickers have STALE prices
   - Skips individual tickers if price age >60 minutes

All protection triggers are logged in decision_trace for full transparency.

### 🚨 Early-Warning Risk Scoring System
- **Universal Risk Monitor**: Tracks risk levels for any asset (Gold, Oil, CNY, CAD, etc.)
- **Four-Dimensional Scoring**: 
  - 🌐 Macro Chain (USD/rates impact)
  - 👥 Crowding (RSI/overbought levels)
  - 📊 Microstructure (volatility/gaps)
  - ⚡ Event Risk (central bank/policy/geopolitical)
- **0-100 Risk Score**: Combined risk assessment with evidence chain
- **Risk Levels**: LOW/MEDIUM/HIGH/CRITICAL with color-coded alerts

### 🎯 Trading-Grade Performance Classification
- **Strict Risk Control**: Determines if signals are suitable for real-money execution
- **Five-Level Classification**:
  - 🟢 **Tradable Edge**: Only classification approved for live trading
  - 🟡 **Directional Signal**: Reference only, no standalone trading
  - 🟠 **Unstable**: Under observation, trading prohibited
  - 🔴 **No Edge**: Permanently banned from trading
  - 🟤 **Insufficient Data**: Sample size too small
- **Multi-Timeframe Validation**: Requires positive performance across multiple timeframes
- **Transaction Cost Consideration**: Net expected value after costs

## ✨ Core Features

### 🧠 Evidence-Based Reasoning
- **Macro Rules Integration**: AI must reference predefined macro logic
- **Forced Evidence Output**: Every prediction requires news-based evidence
- **Hallucination Detection**: Automatically validates AI-generated evidence
- **Degradation Strategy**: Warns when evidence is insufficient

### 📊 Signal Scoreboard
- **Automatic Signal Recording**: Tracks all AI predictions with timestamps
- **Multi-Timeframe Backfilling**: Validates performance at 1h/4h/1d/1w intervals
- **Enhanced Statistics**: Drawdown, volatility, profit factor, win rate
- **Performance Tracking**: Cumulative returns and accuracy over time

### 🌍 Macro & FX Analysis
- **Global Currency Monitoring**: USD/CNY, CAD/USD, EUR/GBP, etc.
- **Commodity Tracking**: Gold, Oil, Bitcoin
- **Cross-Rate Calculations**: Real-time currency pair analysis
- **Interactive Charts**: Candlestick charts with MA20

### 🇺🇸 US Stock Analysis
- **Individual Stock Deep Dive**: NVDA, TSLA, AAPL analysis
- **Company-Specific News**: Google News integration
- **Sentiment Analysis**: AI-powered bullish/bearish assessment
- **Real-Time Pricing**: Live stock price updates

### 🧠 Long-Term Memory (RAG)
- **ChromaDB Integration**: Vector database for event storage
- **Trend Detection**: "This is the 3rd oil warning this week"
- **Historical Context**: AI remembers past events for better analysis

## 🛠️ Tech Stack

* **Core**: Python 3.10+
* **UI**: Streamlit
* **AI Engine**: Ollama (DeepSeek-R1, Qwen2.5)
* **Memory**: ChromaDB (Vector Database)
* **Data**: yfinance, feedparser (RSS feeds)
* **Visualization**: Plotly
* **Notifications**: Plyer

## ⚙️ Prerequisites

1. **Hardware**: PC with NVIDIA GPU (Recommended) or Mac M-Series
2. **Software**: [Ollama](https://ollama.com/) installed
3. **AI Model**: DeepSeek-R1:8B (recommended for reasoning) or Qwen2.5:7B

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GlobalWatch.git
   cd GlobalWatch
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit yfinance feedparser ollama pandas plotly chromadb plyer
   ```

3. **Download the AI Model**:
   ```bash
   # For reasoning tasks (recommended)
   ollama pull deepseek-r1:8b
   
   # Or for faster performance
   ollama pull qwen2.5:7b
   ```

## 🚀 Usage

### GlobalWatch UI (Market Analysis)

Run the main application:

```bash
python -m streamlit run GlobalWatch_V2.py
```

Or use the batch file:
```bash
Start_GlobalWatch.bat
```

### Paper Trading System

#### Quick Start (1-hour test)
```bash
python -u paper_trading.py paper_config_quick_test.json
```

#### Full 48-Hour Simulation
```bash
python -u paper_trading.py paper_config.json
```

Or use the unbuffered batch file (recommended for Windows Terminal):
```bash
Start_Paper_Trading_Unbuffered.bat
```

#### Auto-Start at Market Open
Set up automatic trading at 6:35 AM PST (5 minutes after US market open):
```bash
# Run as Administrator
Setup_Auto_Schedule.bat
```

This creates a Windows scheduled task that automatically starts paper trading every trading day.

#### Monitor Live Progress
```bash
# View real-time summary (updates every 15 minutes)
type outputs\paper_summary_live.txt

# View trade log
type outputs\paper_trades.csv

# View detailed snapshots
type outputs\portfolio_snapshots.jsonl
```

#### Resume from Checkpoint
If you interrupt the program (Ctrl+C), simply restart it:
```bash
python -u paper_trading.py paper_config.json
```
The system will detect the checkpoint and ask if you want to continue from where you left off.

### 📱 Interface Overview

The application opens with four main tabs:

#### 🌍 Macro/FX Tab
- Select currency pairs and commodities
- View real-time cross rates
- AI analysis with evidence chain
- Risk sentiment gauge (0-10)

#### 🇺🇸 US Stocks Tab
- Enter stock ticker (e.g., NVDA)
- Company-specific news analysis
- AI sentiment: Bullish/Bearish/Neutral
- Price charts with technical indicators

#### 📊 Signal Scoreboard Tab
- **Trading-Grade Classification**: See if signals are approved for live trading
- **Performance Metrics**: Accuracy, returns, drawdown, profit factor
- **Multi-Timeframe Analysis**: 1h/4h/1d/1w performance validation
- **V1 vs V2 Classification**: Reference vs trading-grade classifications

#### 🚨 Early-Warning Tab
- **Asset Risk Monitoring**: Select from watchlist (Gold/Oil/CNY/CAD)
- **Four-Dimensional Risk Breakdown**: Detailed risk analysis
- **Evidence Chain**: Price and news evidence for each risk dimension
- **Risk Radar Chart**: Visual risk assessment
- **Alert Triggers**: Automatic risk level warnings

### 🎯 Key Workflows

#### For Risk Management:
1. Check **Early-Warning** tab daily
2. Monitor assets with MEDIUM+ risk levels
3. Reduce exposure when risk reaches HIGH/CRITICAL

#### For Trading Decisions:
1. Run analysis in **Macro/FX** or **US Stocks** tabs
2. Check **Signal Scoreboard** for trading-grade classification
3. **Only trade signals classified as 🟢 Tradable Edge**
4. All other classifications are prohibited for live trading

#### For Performance Validation:
1. Regularly click "🔄 Update Results" in Signal Scoreboard
2. Monitor classification changes over time
3. Wait for sufficient sample size (≥30 for evaluation, ≥50 for trading)

## ⚠️ Important Disclaimers

### Trading Classification System
- **🟢 Tradable Edge**: Only classification approved for real-money execution
- **All Other Classifications**: Prohibited for live trading, no exceptions
- **Sample Size Requirements**: Minimum 30 samples for evaluation, 50 for trading approval
- **Multi-Timeframe Validation**: Must show positive performance across multiple timeframes

### Risk Warnings
- This software is for **educational and research purposes only**
- AI predictions are based on news sentiment analysis and **do not constitute financial advice**
- **Always do your own due diligence** before making trading decisions
- Past performance does not guarantee future results
- **Only use 🟢 Tradable Edge signals for live trading**

---

# 🇨🇳 中文文档 (Chinese Version)

**GlobalWatch V2.5** 是一个完全在本地运行的、注重隐私的实时金融情报终端。

它结合了 **实时 RSS 新闻源**（路透社、CNBC、BBC）和 **本地大语言模型**（Ollama），能够自主进行市场分析、趋势检测和风险评估。全程无需将数据上传云端，也无需支付 API 费用。

## 🚀 V2.5 版本新功能

### 📈 高级纸上交易系统
- **48小时模拟**: 自动化纸上交易，实时价格更新
- **多层策略引擎**:
  - **动量+波动率评分**: 动态资产评估
  - **Regime Filter**: 基于MA50的市场状态检测（risk_on/neutral/risk_off）
  - **宏观整合**: 连接GlobalWatch ChromaDB获取宏观信号
  - **动态风险调整**: 根据市场条件自动调整现金和仓位限制
- **实时监控**: 实时摘要更新、交易日志、盈亏追踪
- **断点续传**: 无缝中断和恢复交易会话
- **19资产池**: ETF（SPY, QQQ, DIA, IWM, GLD, TLT）+ 科技股（AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA）+ 防御性股票（JPM, XOM, JNJ, PG, KO）
- **高级风险控制**: 
  - 冷却保护（再平衡间隔最少12小时）
  - 权重阈值过滤（最小变化2.5%）
  - 最小名义金额过滤（最小交易$400）
  - 最大回撤限制、仓位管理、交易成本
- **全面报告**: 
  - 基准比较（vs QQQ, SPY, VTI, DIA）
  - Regime状态追踪
  - 宏观信号整合
  - 详细交易理由日志
  - 资金曲线、表现指标

### 🎯 Regime Filter 系统
- **MA50趋势分析**: 监控4个基准指数（QQQ, SPY, VTI, DIA）
- **动态状态检测**: 
  - 🟢 **Risk_On**: ≥75%指数在MA50之上 → 最小现金10%，最大权重25%
  - 🟡 **Neutral**: 50-75%在MA50之上 → 最小现金20%，最大权重25%
  - 🔴 **Risk_Off**: ≤50%在MA50之上 → 最小现金35%，最大权重20%
- **自动风险调整**: 组合约束根据市场条件自适应

### 🌐 宏观信号整合
- **GlobalWatch连接**: 从ChromaDB读取交易信号
- **时间衰减加权**: 信号按`exp(-0.15 * age_hours)` × 置信度加权
- **主题投票**: 按主题聚合信号（oil_bullish, risk_off, usd_strong等）
- **确认规则**: 需要2/3信号确认一个主题
- **风险评分**: 0-10刻度，越高越risk-off
- **资产倾斜**: 应用宏观驱动的权重调整（每资产最多±2%）
- **现金调整**: 根据宏观风险分数增加现金配置

### 📝 交易理由日志
每笔交易现在包含完整上下文：
- **Regime State**: 交易时的市场状态
- **Trend Score**: 指数在MA50之上的百分比
- **Cash Target**: 动态最小现金要求
- **Macro Risk Score**: GlobalWatch风险评估（0-10）
- **Macro Topics**: 确认的主题（如"oil_bullish:bullish; risk_off:bearish"）
- **Macro Tilts**: 活跃的资产倾斜（如"XOM:+2.00%; TLT:+2.00%"）
- **Decision Trace**: 执行路径（如"cooldown_pass | weight_threshold_pass | min_notional_pass | macro_tilt_+2.00% | risk_on_add-risk"）

### 🚨 Early-Warning 风险评分系统
- **通用风险监控**: 追踪任何资产的风险水平（黄金、原油、人民币、加元等）
- **四维评分体系**: 
  - 🌐 宏观链条（美元/利率影响）
  - 👥 拥挤度（RSI/超买超卖水平）
  - 📊 微结构（波动率/跳空）
  - ⚡ 事件风险（央行/政策/地缘政治）
- **0-100 风险分数**: 综合风险评估，附带证据链
- **风险等级**: 低/中/高/极端，颜色编码预警

### 🎯 交易级性能分类体系
- **严格风控**: 判断信号是否适合真金白银执行
- **五级分类体系**:
  - 🟢 **Tradable Edge**: 唯一允许实盘交易的分类
  - 🟡 **Directional Signal**: 仅供参考，禁止单独交易
  - 🟠 **Unstable**: 观察中，禁止交易
  - 🔴 **No Edge**: 永久禁止交易
  - 🟤 **Insufficient Data**: 样本数不足
- **多时间窗口验证**: 要求在多个时间框架下都表现良好
- **交易成本考虑**: 扣除成本后的净期望值

## ✨ 核心功能

### 🧠 基于证据的推理
- **宏观规则注入**: AI 必须引用预定义的宏观逻辑
- **强制证据输出**: 每个预测都需要基于新闻的证据
- **幻觉检测**: 自动验证 AI 生成的证据
- **降级策略**: 证据不足时发出警告

### 📊 信号记分板
- **自动信号记录**: 追踪所有 AI 预测及时间戳
- **多时间框架回填**: 在 1h/4h/1d/1w 间隔验证表现
- **增强统计**: 回撤、波动率、盈亏比、胜率
- **表现追踪**: 累计收益和准确率随时间变化

### 🌍 宏观与外汇分析
- **全球货币监控**: USD/CNY, CAD/USD, EUR/GBP 等
- **大宗商品追踪**: 黄金、原油、比特币
- **交叉汇率计算**: 实时货币对分析
- **交互式图表**: K线图配 MA20

### 🇺🇸 美股分析
- **个股深度分析**: NVDA, TSLA, AAPL 分析
- **公司专属新闻**: Google News 集成
- **情绪分析**: AI 驱动的看涨/看跌评估
- **实时定价**: 实时股价更新

### 🧠 长期记忆 (RAG)
- **ChromaDB 集成**: 事件存储的向量数据库
- **趋势检测**: "这是本周第三次石油警告"
- **历史背景**: AI 记住过去事件以进行更好分析

## 🛠️ 技术栈

* **核心**: Python 3.10+
* **界面**: Streamlit
* **AI 引擎**: Ollama (DeepSeek-R1, Qwen2.5)
* **记忆库**: ChromaDB (向量数据库)
* **数据源**: yfinance, feedparser (RSS 源)
* **可视化**: Plotly
* **通知**: Plyer

## ⚙️ 环境要求

1. **硬件**: 推荐配备 NVIDIA 显卡的 PC 或 M 系列芯片的 Mac
2. **软件**: 已安装 [Ollama](https://ollama.com/)
3. **AI 模型**: DeepSeek-R1:8B（推荐用于推理）或 Qwen2.5:7B

## 📦 安装指南

1. **克隆项目**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GlobalWatch.git
   cd GlobalWatch
   ```

2. **安装依赖库**:
   ```bash
   pip install streamlit yfinance feedparser ollama pandas plotly chromadb plyer
   ```

3. **下载 AI 模型**:
   ```bash
   # 用于推理任务（推荐）
   ollama pull deepseek-r1:8b
   
   # 或用于更快性能
   ollama pull qwen2.5:7b
   ```

## 🚀 运行方法

### GlobalWatch 界面（市场分析）

启动主应用：

```bash
python -m streamlit run GlobalWatch_V2.py
```

或使用批处理文件：
```bash
Start_GlobalWatch.bat
```

### 纸上交易系统

#### 快速测试（1小时）
```bash
python -u paper_trading.py paper_config_quick_test.json
```

#### 完整48小时模拟
```bash
python -u paper_trading.py paper_config.json
```

或使用无缓冲批处理文件（推荐用于 Windows Terminal）：
```bash
Start_Paper_Trading_Unbuffered.bat
```

#### 开盘自动启动
设置在美股开盘后5分钟（太平洋时间 6:35 AM）自动启动交易：
```bash
# 以管理员身份运行
Setup_Auto_Schedule.bat
```

这会创建一个 Windows 计划任务，在每个交易日自动启动纸上交易。

#### 监控实时进度
```bash
# 查看实时摘要（每15分钟更新）
type outputs\paper_summary_live.txt

# 查看交易日志
type outputs\paper_trades.csv

# 查看详细快照
type outputs\portfolio_snapshots.jsonl
```

#### 断点续传
如果中断程序（Ctrl+C），只需重新启动：
```bash
python -u paper_trading.py paper_config.json
```
系统会检测到检查点并询问是否从上次中断处继续。

### 📱 界面概览

应用程序打开后有四个主要标签页：

#### 🌍 宏观/外汇标签页
- 选择货币对和大宗商品
- 查看实时交叉汇率
- AI 分析附带证据链
- 风险情绪仪表（0-10）

#### 🇺🇸 美股标签页
- 输入股票代码（如 NVDA）
- 公司专属新闻分析
- AI 情绪：看涨/看跌/中性
- 价格图表配技术指标

#### 📊 信号记分板标签页
- **交易级分类**: 查看信号是否获准实盘交易
- **表现指标**: 准确率、收益、回撤、盈亏比
- **多时间框架分析**: 1h/4h/1d/1w 表现验证
- **V1 vs V2 分类**: 参考级 vs 交易级分类

#### 🚨 Early-Warning 标签页
- **资产风险监控**: 从监控列表选择（黄金/原油/CNY/CAD）
- **四维风险分解**: 详细风险分析
- **证据链**: 每个风险维度的价格和新闻证据
- **风险雷达图**: 可视化风险评估
- **警报触发器**: 自动风险等级警告

### 🎯 关键工作流程

#### 风险管理：
1. 每日检查 **Early-Warning** 标签页
2. 监控中等+风险等级的资产
3. 风险达到高/极端时减少敞口

#### 交易决策：
1. 在 **宏观/外汇** 或 **美股** 标签页运行分析
2. 检查 **信号记分板** 的交易级分类
3. **只交易分类为 🟢 Tradable Edge 的信号**
4. 所有其他分类都禁止实盘交易

#### 表现验证：
1. 定期在信号记分板点击 "🔄 Update Results"
2. 监控分类随时间的变化
3. 等待足够样本量（≥30 用于评估，≥50 用于交易）

## ⚠️ 重要免责声明

### 交易分类系统
- **🟢 Tradable Edge**: 唯一获准真金白银执行的分类
- **所有其他分类**: 禁止实盘交易，无例外
- **样本量要求**: 最少 30 个样本用于评估，50 个用于交易批准
- **多时间框架验证**: 必须在多个时间框架显示正表现

### 风险警告
- 本软件仅供 **教育和研究目的**
- AI 预测基于新闻情绪分析，**不构成金融建议**
- 交易前 **务必进行自己的尽职调查**
- 过往表现不保证未来结果
- **只使用 🟢 Tradable Edge 信号进行实盘交易**

---

## 📄 Documentation

### GlobalWatch UI
- **Complete Guide**: `GLOBALWATCH_COMPLETE_GUIDE.md`
- **Early-Warning System**: `EARLY_WARNING_IMPLEMENTATION.md`
- **Trading Classification**: `TRADING_GRADE_CLASSIFICATION.md`
- **Quick Reference**: `TRADING_GRADE_QUICK_REF.md`

### Paper Trading System
- **Real-Time Price Fix**: `REAL_TIME_PRICE_FIX.md` - How real-time price fetching works
- **Price Fetching Enhanced**: `PRICE_FETCHING_ENHANCED.md` - 4-tier price fetching mechanism
- **Real-Time Monitoring**: `REAL_TIME_MONITORING.md` - How to monitor live trading progress
- **Resume Feature**: `RESUME_FEATURE.md` - Checkpoint and resume functionality
- **Restart Instructions**: `RESTART_INSTRUCTIONS.md` - Quick restart guide
- **Macro Confirmation Logic**: `MACRO_CONFIRMATION_LOGIC.md` - How macro signal confirmation works

### Configuration Files
- **`paper_config.json`**: 48-hour full simulation with advanced features
  - 19 assets, $30,000 initial capital
  - 6-hour rebalance interval (360 minutes)
  - Regime Filter enabled (MA50 trend analysis)
  - Macro Integration enabled (GlobalWatch ChromaDB)
  - Benchmark comparison (QQQ, SPY, VTI, DIA)
  - Five-layer protection system (cooldown, weight threshold, min notional, turnover cap, price freshness)
  - Macro signal smoothing (median over 3 cycles) and cooldown (2 cycles)
  - Price freshness controls (60-min skip threshold, 30% STALE abort ratio)
- **`paper_config_quick_test.json`**: 1-hour quick test (6 assets, $20,000 initial capital)

### Key Configuration Parameters
**Execution Controls**:
- `rebalance_cooldown_minutes`: 720 (12 hours between rebalances)
- `weight_threshold`: 0.025 (2.5% minimum weight change)
- `min_trade_notional_usd`: 400 (minimum trade size)
- `max_turnover_pct_per_rebalance`: 0.20 (20% max portfolio turnover)
- `stale_price_skip_minutes`: 60 (skip trades if price >60 min old)
- `stale_abort_ratio`: 0.5 (abort if >50% of prices STALE)
- `max_stale_ratio`: 0.3 (abort if >30% of candidates STALE)

**Macro Integration**:
- `confirm_k_of_n`: [2, 3] (require 2 of last 3 signals same direction)
- `signal_max_age_hours`: 48 (only use signals <48 hours old)
- `decay_lambda_per_hour`: 0.15 (time decay for strength calculation)
- `macro_cash_slope`: 0.02 (2% cash increase per risk score point)
- `tilt_max_delta`: 0.02 (±2% max tilt per asset)
- `smoothing_window`: 3 (median over last 3 risk scores)
- `smoothing_method`: "median" (or "ewma")
- `ewma_alpha`: 0.4 (EWMA smoothing parameter)
- `cooldown_cycles`: 2 (freeze cash target for 2 cycles after >5% change)

### Paper Trading Output Files
- **`outputs/paper_summary_live.txt`**: Real-time summary (updates every cycle)
  - Current performance metrics
  - Market regime state
  - Macro signals from GlobalWatch (smoothed risk score)
  - Benchmark comparison
  - Current holdings
- **`outputs/paper_summary.txt`**: Final report after completion
- **`outputs/paper_trades.csv`**: Complete trade log with reasoning
  - Columns: timestamp, ticker, side, quantity, price, cost, reason
  - **Context fields**: regime_state, trend_score, cash_target, macro_risk_score, macro_topics, macro_tilts
  - **Protection fields**: decision_trace, price_age_minutes, price_status
- **`outputs/portfolio_snapshots.jsonl`**: Detailed snapshots (one per cycle)
  - **New fields**: stale_count, stale_ratio, price_stale_skip, turnover_notional, turnover_limit, turnover_scale, turnover_capped
  - **Macro fields**: macro_risk_score_smoothed, macro_tilts_ignored, macro_cooldown_remaining
- **`outputs/equity_curve.png`**: Visual performance chart

### Engine Version Fingerprint
The paper trading engine prints a version fingerprint at startup:
- **ENGINE_VERSION**: v2.5-ABCDEF-2026-02-05
- **HAS_MACRO_SMOOTH**: True (macro signal smoothing enabled)
- **PRICE_API_RETURNS_TUPLE**: True (price freshness tracking)
- **HAS_STALE_PRICE_SKIP**: True (STALE price protection)
- **HAS_TURNOVER_CAP**: True (turnover limiting)
- **HAS_MACRO_COOLDOWN**: True (macro cooldown mechanism)
- **HAS_REGIME_FILTER**: True (MA50 regime detection)
- **HAS_MACRO_INTEGRATION**: True (GlobalWatch integration)

### Example Trade Log Entry
```csv
timestamp,ticker,side,quantity,price,cost,reason,regime_state,trend_score,cash_target,macro_risk_score,macro_topics,macro_tilts,decision_trace,price_age_minutes,price_status
2026-02-05T10:30:00,XOM,BUY,15,146.50,1.10,rebalance,risk_on,0.75,0.10,3.2,oil_bullish:bullish,XOM:+2.00%; TLT:+2.00%,cooldown_pass | weight_threshold_pass | min_notional_pass | stale_check_pass | turnover_cap_scale_92% | macro_tilt_+2.00% | risk_on_add-risk,2.5,LIVE
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

MIT License - see LICENSE file for details.

---

**Remember: Only 🟢 Tradable Edge signals are approved for live trading. No exceptions!**