# 实时价格修复说明

## 🐛 发现的问题

### 原始代码问题
```python
# 旧代码 - 只获取日线收盘价
hist = self.get_market_data(ticker, period='5d', interval='1d')
return float(hist['Close'].iloc[-1])
```

**问题**：
- 使用日线数据 (`interval='1d'`)
- 在同一交易日内多次运行，获取的都是**昨天的收盘价**
- 无法反映盘中价格变化

### 症状
- 两次运行（相隔2小时）价格完全相同
- 买入决策完全相同
- 无法看到实时市场变化

---

## ✅ 修复方案

### 新的三层价格获取机制

#### 第1层：实时价格 (最优先)
```python
info = t.info
price = info['currentPrice']  # 或 regularMarketPrice
```
- 获取当前市场价格
- 最准确，最实时

#### 第2层：分钟级历史数据
```python
hist = t.history(period='1d', interval='1m')
price = hist['Close'].iloc[-1]
```
- 获取最近1分钟的收盘价
- 接近实时

#### 第3层：日线数据（降级）
```python
hist = t.history(period='5d', interval='1d')
price = hist['Close'].iloc[-1]
```
- 如果前两种方法都失败
- 至少保证程序能运行

---

## 📊 新增调试输出

现在每次获取价格都会显示：
```
[PRICE] GOOGL: $340.25 (from currentPrice)
[PRICE] AAPL: $175.50 (from 1m history)
[PRICE] MSFT: $380.00 (from daily close)
```

这样你可以：
- ✅ 看到价格来源
- ✅ 确认是否获取到实时价格
- ✅ 诊断价格获取问题

---

## 🔍 验证方法

### 测试1: 快速连续运行
```bash
# 第一次运行
python paper_trading.py paper_config.json
# Ctrl+C 停止

# 等待1-2分钟

# 第二次运行
python paper_trading.py paper_config.json
```

**预期结果**：
- 价格应该有微小变化（如果在交易时间）
- 或者至少看到 `[PRICE]` 日志显示价格来源

### 测试2: 观察周期内价格变化
运行程序，观察连续几个周期：
```
Cycle 0: GOOGL $339.71
Cycle 1: GOOGL $340.15  ← 应该有变化
Cycle 2: GOOGL $339.95  ← 应该有变化
```

---

## ⚠️ 注意事项

### 1. 市场休市时间
如果在美股休市时间运行：
- 所有方法都会返回最后的收盘价
- 价格不会变化是正常的

**美股交易时间**（北京时间）：
- 夏令时：21:30 - 04:00
- 冬令时：22:30 - 05:00

### 2. API 限制
- yfinance 可能有请求频率限制
- 如果请求太频繁，可能被限速
- 第1层方法可能失败，会自动降级到第2、3层

### 3. 网络延迟
- 实时价格获取需要网络连接
- 可能比日线数据慢1-2秒
- 这是正常的

---

## 🚀 立即测试

```bash
# 停止当前程序
taskkill /f /im python.exe

# 运行新版本
python paper_trading.py paper_config.json
```

**观察要点**：
1. ✅ 看到 `[PRICE]` 日志
2. ✅ 价格来源显示（currentPrice/1m history/daily close）
3. ✅ 连续周期价格有变化（如果在交易时间）

---

## 📈 预期改进

### 修复前
```
Run 1 (14:32): GOOGL $339.71
Run 2 (16:42): GOOGL $339.71  ← 完全相同！
```

### 修复后
```
Run 1 (14:32): GOOGL $339.71 (from currentPrice)
Run 2 (16:42): GOOGL $340.25 (from currentPrice)  ← 有变化！
```

---

## 🎯 总结

**问题**：使用日线数据导致价格不更新
**修复**：三层价格获取机制，优先使用实时价格
**验证**：观察 `[PRICE]` 日志和价格变化

**现在重新运行，你应该能看到实时价格了！** 🚀
