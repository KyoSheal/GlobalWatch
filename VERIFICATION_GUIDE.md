# 验收指南 (Verification Guide)

## 快速验收步骤（10-15 分钟）

### 第一步：语法检查 ✅
```bash
python -m py_compile GlobalWatch_V2.py
```
**预期结果**：无输出，退出码 0

---

### 第二步：启动 Streamlit
```bash
streamlit run GlobalWatch_V2.py
```
**预期结果**：
- 无语法错误
- 无 import 错误
- 浏览器自动打开 http://localhost:8501
- 界面正常显示

---

### 第三步：验证主要页面

#### 3.1 宏观/外汇 (Macro/FX) 页面
1. 点击 "🚀 Deep Reason Analysis" 按钮
2. 等待分析完成
3. **检查点**：
   - ✅ 无崩溃
   - ✅ 显示分析结果或 "Market is Stable"
   - ✅ 新闻源正常显示
   - ✅ 图表正常渲染

#### 3.2 美股透视 (US Stocks) 页面
1. 输入股票代码（如 NVDA）
2. 点击 "🔍 Analyze"
3. **检查点**：
   - ✅ 无崩溃
   - ✅ 显示股价和图表
   - ✅ 显示分析结果

#### 3.3 Signal Scoreboard 页面
1. 点击 "🔄 Update Results"
2. **检查点**：
   - ✅ 无崩溃
   - ✅ 显示统计数据（即使为 0）
   - ✅ 显示交易级分类

#### 3.4 Early-Warning 页面
1. 选择资产（如 Gold）
2. 点击 "🔍 Calculate Risk Score"
3. **检查点**：
   - ✅ 无崩溃
   - ✅ 显示风险评分
   - ✅ 显示雷达图

---

### 第四步：验证错误日志

#### 4.1 检查日志文件是否创建
```bash
dir outputs
```
**预期结果**：
- 存在 `outputs/` 目录
- 可能存在 `error.log` 文件（如果有错误发生）

#### 4.2 查看日志内容（如果存在）
```bash
type outputs\error.log
```
**预期结果**：
- 日志格式正确：`[YYYY-MM-DD HH:MM:SS] 错误消息`
- 错误信息清晰可读
- 无敏感信息泄露

---

### 第五步：模拟 JSON 解析错误

#### 5.1 修改模型输出（可选测试）
这一步需要手动修改代码来模拟错误，仅用于深度验证。

**跳过此步骤**，因为现有的 `robust_json_parse()` 已经有完整的降级机制。

#### 5.2 验证降级行为
如果 AI 输出非 JSON：
- ✅ 不会崩溃
- ✅ 显示友好错误消息
- ✅ 显示原始输出（可展开）
- ✅ 提供建议操作

---

### 第六步：验证异常处理

#### 6.1 断网测试（可选）
1. 断开网络连接
2. 点击 "🚀 Deep Reason Analysis"
3. **预期结果**：
   - ✅ 不会崩溃
   - ✅ 显示错误提示（如 "No news available"）
   - ✅ 错误记录到 `outputs/error.log`

#### 6.2 无效股票代码测试
1. 在 US Stocks 页面输入无效代码（如 "INVALID123"）
2. 点击 "🔍 Analyze"
3. **预期结果**：
   - ✅ 不会崩溃
   - ✅ 显示错误提示
   - ✅ 错误记录到日志

---

## 完整验收清单

### 功能验收
- [ ] Streamlit 正常启动
- [ ] 宏观/外汇页面正常工作
- [ ] 美股透视页面正常工作
- [ ] Signal Scoreboard 页面正常工作
- [ ] Early-Warning 页面正常工作
- [ ] 所有图表正常渲染
- [ ] 所有按钮正常响应

### 错误处理验收
- [ ] 无裸 `except:` 语句
- [ ] 所有异常都有日志记录
- [ ] `outputs/error.log` 正常创建
- [ ] 日志格式正确
- [ ] 错误不会导致崩溃

### 兼容性验收
- [ ] 现有功能未被破坏
- [ ] JSON schema 未改变
- [ ] Prompt A/ProE 正常工作
- [ ] memory_db 正常工作
- [ ] scoreboard 正常工作
- [ ] early-warning 正常工作
- [ ] paper trading 模块未受影响

### 性能验收
- [ ] 启动速度无明显变化
- [ ] 分析速度无明显变化
- [ ] 内存使用无明显增加

---

## 常见问题排查

### Q1: Streamlit 启动失败
**可能原因**：
- Python 环境问题
- 依赖包未安装

**解决方案**：
```bash
pip install streamlit pandas yfinance feedparser ollama plotly chromadb plyer
```

### Q2: 找不到 outputs 目录
**可能原因**：
- 尚未发生任何错误
- 权限问题

**解决方案**：
- 手动创建：`mkdir outputs`
- 检查文件权限

### Q3: 日志文件为空
**可能原因**：
- 系统运行正常，无错误发生

**解决方案**：
- 这是正常现象
- 可以通过断网测试触发错误

### Q4: JSON 解析错误
**可能原因**：
- 模型输出格式问题
- 网络问题

**解决方案**：
- 系统会自动降级，显示友好错误
- 检查 Ollama 是否正常运行
- 查看 `outputs/error.log` 获取详细信息

---

## 成功标准

✅ **必须满足**：
1. Streamlit 正常启动
2. 至少 2 个主要页面正常工作
3. 无崩溃错误
4. 错误日志正常记录

✅ **建议满足**：
1. 所有 4 个页面正常工作
2. 所有图表正常渲染
3. 异常处理测试通过
4. 日志格式清晰可读

---

## 验收完成后

1. ✅ 确认所有功能正常
2. ✅ 删除测试数据（如果有）
3. ✅ 提交代码变更
4. ✅ 更新文档（如需要）
5. ✅ 通知团队（如需要）

---

## 联系支持

如果遇到问题：
1. 检查 `outputs/error.log`
2. 查看 Streamlit 控制台输出
3. 查看 Ollama 日志
4. 参考 `CODE_QUALITY_FIXES.md`

---

**验收时间估计**：10-15 分钟
**难度等级**：简单
**需要技能**：基础 Python + Streamlit 使用
