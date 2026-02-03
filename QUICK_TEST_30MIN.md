# 30 分钟 Quick Test 启动指南

## 🚀 快速启动

### 方法 1: 使用批处理文件（推荐）
```bash
test_paper_trading.bat
```

### 方法 2: 直接命令行
```bash
python paper_trading.py paper_config_quick_test.json
```

---

## ⏱️ 测试配置

- **初始资金**: $20,000 USD
- **运行时长**: 30 分钟 (0.5 小时)
- **调仓间隔**: 5 分钟
- **预计调仓次数**: 6 次
- **资产池**: SPY, QQQ, AAPL, MSFT, GLD, CASH

---

## ✅ 成功判定标准（最少满足）

测试完成后，以下文件**必须**生成：

### 1. 资金曲线图 ✅
```
outputs/equity_curve_quick.png
```
- 显示 30 分钟内的资金变化
- 包含总资产曲线
- 可能包含回撤曲线

### 2. 交易日志 ✅
```
outputs/paper_trades_quick.csv
```
- **即使 0 笔交易也必须有文件**
- CSV 格式，包含表头
- 记录所有买卖操作

### 3. 汇总报告 ✅
```
outputs/paper_summary_quick.txt
```
- 文本格式
- 包含关键指标：
  - 初始资金
  - 最终资金
  - 总收益率
  - 最大回撤
  - 交易次数

### 4. 组合快照（可选）
```
outputs/portfolio_snapshots_quick.jsonl
```
- JSONL 格式（每行一个 JSON）
- 记录每次调仓后的持仓

---

## 📊 预期输出示例

### equity_curve_quick.png
- X 轴: 时间（30 分钟）
- Y 轴: 资金（$20,000 附近）
- 曲线: 平滑或波动（取决于市场）

### paper_trades_quick.csv
```csv
timestamp,ticker,side,quantity,price,cost,reason
2026-02-03 01:00:00,SPY,BUY,10,450.00,0.23,Initial allocation
2026-02-03 01:05:00,AAPL,SELL,5,180.00,0.09,Rebalance
...
```

### paper_summary_quick.txt
```
========================================
Paper Trading Summary Report
========================================

Duration: 0.5 hours
Initial Cash: $20,000.00
Final Equity: $20,050.00
Total Return: +0.25%
Max Drawdown: -0.10%
Total Trades: 12

...
```

---

## 🔍 实时监控

测试运行时，你会看到：

```
========================================
Paper Trading Simulation
========================================

Config: paper_config_quick_test.json
Duration: 0.5 hours
Rebalance: every 5 minutes

[00:00] Starting simulation...
[00:00] Initial portfolio: $20,000.00
[00:05] Rebalance #1 - Equity: $20,010.00
[00:10] Rebalance #2 - Equity: $20,025.00
[00:15] Rebalance #3 - Equity: $20,015.00
...
[00:30] Simulation complete!

Generating reports...
✓ Equity curve saved
✓ Trades log saved
✓ Summary report saved
```

---

## ⚠️ 注意事项

### 1. 市场时间
- 如果在**美股休市时间**运行，价格可能不会变化
- 建议在**美股交易时间**运行（北京时间 22:30-05:00）
- 或者接受静态价格测试（验证系统功能）

### 2. 网络连接
- 需要连接互联网获取实时价格
- 如果网络断开，会使用最后已知价格

### 3. 运行时间
- 实际运行时间 = 30 分钟 + 初始化时间（约 1-2 分钟）
- 总计约 **32 分钟**

### 4. 不要中断
- 让程序完整运行 30 分钟
- 不要按 Ctrl+C 中断
- 如果必须中断，文件可能不完整

---

## 🐛 故障排查

### 问题 1: 找不到 paper_trading.py
```bash
# 确认当前目录
dir paper_trading.py

# 如果不在当前目录，切换到正确目录
cd C:\Users\kyosh\Desktop\Project\News
```

### 问题 2: 找不到配置文件
```bash
# 确认配置文件存在
dir paper_config_quick_test.json
```

### 问题 3: 缺少依赖
```bash
pip install yfinance numpy pandas matplotlib
```

### 问题 4: 输出文件未生成
- 检查 `outputs/` 目录是否存在
- 查看控制台错误信息
- 检查磁盘空间

---

## ✅ 验收步骤

### 1. 启动测试
```bash
test_paper_trading.bat
```

### 2. 等待完成（约 32 分钟）
- 观察控制台输出
- 确认无错误信息

### 3. 检查文件
```bash
dir outputs\equity_curve_quick.png
dir outputs\paper_trades_quick.csv
dir outputs\paper_summary_quick.txt
```

### 4. 查看结果
```bash
# 查看汇总报告
type outputs\paper_summary_quick.txt

# 打开资金曲线图
start outputs\equity_curve_quick.png

# 查看交易日志
type outputs\paper_trades_quick.csv
```

---

## 🎯 成功标准总结

| 文件 | 必须存在 | 最小要求 |
|------|---------|---------|
| equity_curve_quick.png | ✅ 是 | 有图表 |
| paper_trades_quick.csv | ✅ 是 | 有表头（即使 0 笔交易） |
| paper_summary_quick.txt | ✅ 是 | 有汇总数据 |
| portfolio_snapshots_quick.jsonl | ⚪ 可选 | - |

**只要以上 3 个必需文件都生成了，Quick Test 就算成功！**

---

## 🚀 下一步：48 小时长跑

Quick Test 成功后，启动 48 小时测试：

```bash
# 方法 1: 使用批处理
Start_PaperTrading.bat

# 方法 2: 直接命令
python paper_trading.py paper_config.json
```

48 小时配置：
- 初始资金: $20,000
- 运行时长: 48 小时
- 调仓间隔: 15 分钟
- 预计调仓次数: 192 次

---

## 📞 需要帮助？

如果遇到问题：
1. 查看控制台错误信息
2. 检查 `outputs/error.log`（如果存在）
3. 确认网络连接
4. 确认 Python 依赖已安装

---

**现在就开始 30 分钟 Quick Test 吧！** 🚀

```bash
test_paper_trading.bat
```
