# 🧪 交易级分类验证指南

## 快速验证步骤

### 1. 启动应用
```bash
python -m streamlit run GlobalWatch_V2.py
```

### 2. 访问 Signal Scoreboard
- 打开浏览器 `http://localhost:8501`
- 点击 "📊 Signal Scoreboard" 标签页

### 3. 查看新的分类区域
- 找到 "🎯 Trading-Grade Performance Classification" 区域
- 应该显示在 Key Metrics 之后

---

## 测试场景

### 场景 A: 数据不足（< 30 样本）

**预期行为**：
- 分类：🟤 Insufficient Data
- 决策：🚫 NOT TRADABLE
- 原因：样本数不足（X/30）
- 警告：⚠️ 样本量过小，任何统计结论都不可靠

**验证步骤**：
1. 如果当前样本数 < 30，直接查看
2. 如果样本数 ≥ 30，等待积累更多数据或查看历史记录

**预期结果**：
```
🟤 Insufficient Data          🚫 NOT TRADABLE
                               禁止实盘交易

📋 Classification Details ▼
原因说明：
样本数不足（18/30）。需要至少 30 个已验证信号才能进行可靠评估。

风险警告：
⚠️ 样本量过小，任何统计结论都不可靠
⚠️ 禁止用于实盘交易
```

---

### 场景 B: 样本充足但未达标（30-49 样本）

**预期行为**：
- 分类：🟠 Unstable / Regime-Dependent
- 决策：🚫 NOT TRADABLE
- 原因：样本量接近下限
- 建议：继续观察

**验证步骤**：
1. 运行多次分析，积累 30-49 个样本
2. 查看分类结果

**预期结果**：
```
🟠 Unstable / Regime-Dependent    🚫 NOT TRADABLE
                                   禁止实盘交易

📋 Classification Details ▼
原因说明：
净期望值为正（0.18%），但存在以下问题：
• 样本量接近下限（38/50）
• 回撤过大（18% > 15%）
→ 标记为「观察中」
→ 不允许自动交易
→ 必须等待更多数据或行情切换验证
```

---

### 场景 C: Tradable Edge（≥ 50 样本 + 所有条件满足）

**预期行为**：
- 分类：🟢 Tradable Edge
- 决策：✅ TRADABLE
- 原因：满足所有交易级标准
- 多时间窗口验证：✅ Yes

**验证步骤**：
1. 积累至少 50 个样本
2. 确保净期望值 > 0
3. 确保回撤 ≤ 15%
4. 切换不同 timeframe 验证

**预期结果**：
```
🟢 Tradable Edge                  ✅ TRADABLE
                                   允许实盘交易

📋 Classification Details ▼
原因说明：
✅ 满足所有交易级标准：
• 样本数充足（65 ≥ 50）
• 净期望值为正（0.25% > 0）
• 回撤可控（12% ≤ 15%）
• 多时间窗口验证通过
→ 允许进入实盘交易

风险警告：
⚠️ Profit Factor 较低（1.3），建议谨慎控制仓位

关键指标：
Net Expected Value: 0.25%
Max Drawdown: 12%
Multi-TF Validated: ✅ Yes
```

---

### 场景 D: Directional Signal（高准确率但被成本侵蚀）

**预期行为**：
- 分类：🟡 Directional Signal
- 决策：🚫 NOT TRADABLE
- 原因：净期望值接近零
- 建议：仅用于辅助

**模拟方法**：
- 需要准确率 ≥ 58% 但平均收益 ≈ 0.1-0.15%
- 这样扣除成本后净期望值 ≈ 0

**预期结果**：
```
🟡 Directional Signal             🚫 NOT TRADABLE
                                   禁止实盘交易

📋 Classification Details ▼
原因说明：
方向准确率较高（60%），但净期望值接近零（0.02%）。
交易成本侵蚀了大部分收益。
→ 仅允许用于：仓位调整、风险确认、多信号共识过滤
→ 禁止单独触发交易

风险警告：
⚠️ 不允许单独用于实盘交易
✓ 可作为辅助信号使用
```

---

### 场景 E: Lucky Streak（低准确率 + 正收益）

**预期行为**：
- 分类：🔴 No Edge（如果无 fat-tail 证据）
- 或 🟠 Unstable（如果有 fat-tail 证据）
- 决策：🚫 NOT TRADABLE
- 原因：疑似运气成分

**模拟方法**：
- 需要准确率 < 55% 但平均收益 > 0
- 这种情况在实际使用中可能出现

**预期结果（无 fat-tail）**：
```
🔴 No Edge                        🚫 NOT TRADABLE
                                   禁止实盘交易

📋 Classification Details ▼
原因说明：
准确率低（48%），收益为正但无非对称结构证据。
大概率是运气（Lucky Streak）。
→ 永久禁止用于交易

风险警告：
🚫 疑似运气成分，禁止交易
```

**预期结果（有 fat-tail）**：
```
🟠 Unstable / Regime-Dependent    🚫 NOT TRADABLE
                                   禁止实盘交易

📋 Classification Details ▼
原因说明：
准确率较低（45%），但存在非对称收益结构：
• 平均盈利：1.20%
• 平均亏损：-0.35%
• Profit Factor：2.5
→ 可能是 fat-tail payoff 策略
→ 需要更多数据验证，暂不允许交易

风险警告：
⚠️ 低准确率 + 高盈亏比，需验证是否可持续
```

---

## 功能完整性检查

### UI 元素
- [ ] "🎯 Trading-Grade Performance Classification" 标题显示
- [ ] 分类标签显示（🟢/🟡/🟠/🔴/🟤）
- [ ] 交易决策显示（✅ TRADABLE / 🚫 NOT TRADABLE）
- [ ] "📋 Classification Details" 可展开区域
- [ ] 原因说明文本显示
- [ ] 风险警告列表显示
- [ ] 关键指标卡片显示（Net EV, Max DD, Multi-TF）

### V1 分类
- [ ] "📊 V1 Classification (Reference Only)" 可展开区域
- [ ] 警告文本："仅供分析参考，不可用于交易决策"
- [ ] V1 分类标签显示
- [ ] V1 分类说明显示

### 增强统计
- [ ] "📊 Enhanced Statistics" 标题显示
- [ ] Cumulative Return 显示
- [ ] Win Rate 显示
- [ ] Profit Factor 显示
- [ ] Volatility 显示
- [ ] Avg Win 显示
- [ ] Avg Loss 显示

### 使用说明
- [ ] "ℹ️ How to Use Signal Scoreboard (V2 - Trading-Grade)" 可展开
- [ ] 包含完整的 V2 分类说明
- [ ] 包含 Lucky Streak 处理说明
- [ ] 包含关键指标说明
- [ ] 包含 V1 vs V2 对比

---

## 数据验证

### 统计指标计算验证

#### 1. 最大回撤计算
```python
# 手动验证
returns = [0.5, -0.3, 0.2, -0.4, 0.6]
cumulative = [0.5, 0.2, 0.4, 0.0, 0.6]

# 最大回撤应该是：
# 从 0.5 到 0.0 = 0.5
# 或从 0.4 到 0.0 = 0.4
# 取最大值 = 0.5

# 对比 UI 显示的 Max Drawdown
```

#### 2. Profit Factor 计算
```python
# 手动验证
wins = [0.5, 0.3, 0.2]  # 总盈利 = 1.0
losses = [-0.2, -0.3]   # 总亏损 = 0.5

# Profit Factor = 1.0 / 0.5 = 2.0

# 对比 UI 显示的 Profit Factor
```

#### 3. 净期望值计算
```python
# 手动验证
avg_return = 0.25
transaction_cost = 0.1

# Net Expected Value = 0.25 - 0.1 = 0.15

# 对比 UI 显示的 Net Expected Value
```

---

## 多时间窗口验证

### 验证步骤

#### 1. 选择主题过滤
```
Theme Filter: FX
Timeframe: 1d
```

#### 2. 查看分类结果
```
如果显示：Multi-TF Validated: ✅ Yes
```

#### 3. 切换时间窗口验证
```
Timeframe: 1h
→ 查看 Net Expected Value 是否 > 0

Timeframe: 4h
→ 查看 Net Expected Value 是否 > 0

Timeframe: 1w
→ 查看 Net Expected Value 是否 > 0
```

#### 4. 确认至少 2 个窗口为正
```
如果 1h 和 1d 都为正 → Multi-TF Validated 应该是 ✅ Yes
如果只有 1d 为正 → Multi-TF Validated 应该是 ❌ No
```

---

## 边界条件测试

### 测试 1: 样本数 = 29（刚好不足）
**预期**：🟤 Insufficient Data

### 测试 2: 样本数 = 30（刚好达标）
**预期**：可能是 🟠 Unstable 或其他（不再是 Insufficient Data）

### 测试 3: 样本数 = 49（接近下限）
**预期**：如果其他条件满足，应该是 🟠 Unstable（样本量接近下限）

### 测试 4: 样本数 = 50（刚好达标）
**预期**：如果其他条件满足，可能升级为 🟢 Tradable Edge

### 测试 5: 净期望值 = 0.001（接近零）
**预期**：🟡 Directional Signal（净期望值接近零）

### 测试 6: 净期望值 = -0.001（刚好为负）
**预期**：🔴 No Edge（净期望值为负）

### 测试 7: 回撤 = 15.0%（刚好达标）
**预期**：如果其他条件满足，可能是 🟢 Tradable Edge

### 测试 8: 回撤 = 15.1%（刚好超标）
**预期**：🟠 Unstable（回撤超标）

---

## 性能验证

### 响应时间
- 分类计算: < 0.5 秒 ✅
- UI 渲染: < 1 秒 ✅
- 多时间窗口验证: < 2 秒 ✅

### 资源占用
- 内存增量: < 10MB ✅
- CPU 峰值: < 30% ✅

---

## 故障排除

### 问题 1: 分类区域不显示
**原因**: UI 代码位置错误  
**检查**: 确认代码在 `st.divider()` 之后

### 问题 2: Multi-TF Validated 始终为 False
**原因**: 未指定 theme 或 asset  
**解决**: 选择 Theme Filter（不要选 "All"）

### 问题 3: 所有分类都是 Insufficient Data
**原因**: 样本数不足  
**解决**: 运行更多分析，积累至少 30 个样本

### 问题 4: V1 分类显示错误
**原因**: 逻辑错误  
**检查**: 确认 V1 分类逻辑正确

### 问题 5: 净期望值计算错误
**原因**: 交易成本参数错误  
**检查**: 确认 transaction_cost = 0.1（默认）

---

## 验收清单

### 必须通过 (P0)
- [ ] 应用可正常启动
- [ ] Signal Scoreboard Tab 可访问
- [ ] 交易级分类区域显示
- [ ] 分类标签正确显示
- [ ] 交易决策正确显示
- [ ] 原因说明正确显示
- [ ] 样本数 < 30 时分类为 Insufficient Data
- [ ] 样本数 < 30 时 decision_allowed = False
- [ ] 只有 Tradable Edge 时 decision_allowed = True
- [ ] V1 分类标记为"仅供参考"
- [ ] Lucky Streak 不再作为正向分类

### 应该通过 (P1)
- [ ] 多时间窗口验证正确
- [ ] 增强统计指标显示
- [ ] V1 分类区域可折叠
- [ ] 使用说明完整
- [ ] 风险警告显示
- [ ] 关键指标卡片显示

### 可选通过 (P2)
- [ ] 分类历史追踪（未实现）
- [ ] 自定义阈值（未实现）
- [ ] 夏普比率（未实现）

---

## 下一步行动

### 立即行动
1. ✅ 运行应用并访问 Signal Scoreboard
2. ✅ 查看交易级分类区域
3. ✅ 验证 UI 显示正确

### 短期行动（本周）
1. 积累至少 30 个样本
2. 观察分类变化
3. 验证多时间窗口逻辑

### 中期行动（本月）
1. 积累至少 50 个样本
2. 尝试达到 Tradable Edge
3. 验证实盘交易决策

---

**验收完成后，系统即可用于交易决策！**

**记住**：
- 只有 🟢 Tradable Edge 允许实盘交易
- 其他所有分类都禁止实盘交易
- 没有例外！
