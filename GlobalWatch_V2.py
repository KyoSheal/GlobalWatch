import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import ollama
from datetime import datetime, timedelta
import time
import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chromadb
import uuid
import urllib.parse

# === 0. 基础设置 ===
try:
    from plyer import notification
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# 【关键修改】切换为推理模型 (请确保终端已运行 ollama pull deepseek-r1:8b)
LOCAL_MODEL = "deepseek-r1:8b" 

# 初始化记忆库
chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="market_events")

# 宏观逻辑库
MACRO_LOGIC_KNOWLEDGE = """
GLOBAL MACRO RULES:
1. CAD (Loonie) is a Petro-currency. Oil UP -> CAD Stronger.
2. CNY (Yuan) is sensitive to USD Strength & Trade Wars.
3. USD is Safe Haven. Crisis -> Capital flows to USD/Gold.
4. TECH STOCKS (e.g. NVDA) are sensitive to Interest Rates & AI hype.
"""

ASSETS_DB = {
    "USD (美元)": {"ticker": "USD", "type": "fiat_base"},
    "CNY (人民币)": {"ticker": "CNY=X", "type": "fiat_quote"}, 
    "CAD (加币)": {"ticker": "CAD=X", "type": "fiat_quote"},
    "GBP (英镑)": {"ticker": "GBP=X", "type": "fiat_quote"},
    "JPY (日元)": {"ticker": "JPY=X", "type": "fiat_quote"},
    "Gold (黄金)": {"ticker": "GC=F", "type": "commodity"},  
    "Crude Oil (原油)": {"ticker": "CL=F", "type": "commodity"},
    "Bitcoin (比特币)": {"ticker": "BTC-USD", "type": "crypto"}
}

MACRO_ANCHORS = {"Crude Oil": "CL=F", "Gold": "GC=F"}

RSS_FEEDS = {
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "CNBC": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml"
}

REFRESH_OPTIONS = {"手动": 0, "5 分钟": 300, "10 分钟": 600, "30 分钟": 1800}

# ================= 1. 深度解析函数 (V3.0 新增) =================

def parse_deepseek_output(text):
    """
    专门解析 DeepSeek-R1 的输出
    返回: (思考过程文本, 纯净的JSON文本)
    """
    # 1. 提取 <think>...</think> 内部的思考过程
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thought_process = think_match.group(1).strip() if think_match else "No internal thought process detected (Direct Output)."
    
    # 2. 移除 <think> 标签，只保留剩下的 JSON 部分
    json_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理 Markdown 代码块标记
    json_text = re.sub(r'```json', '', json_text)
    json_text = re.sub(r'```', '', json_text).strip()
    
    return thought_process, json_text

def extract_json_from_text(text):
    """
    从文本中提取第一个合法的 JSON 对象
    支持前后有多余文本、markdown、解释性内容
    
    Args:
        text: 原始文本
    Returns:
        json_str: 提取的 JSON 字符串，如果未找到返回 None
    """
    # 策略 1: 查找 { ... } 包裹的内容
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # 找到完整的 JSON 对象
                json_candidate = text[start_idx:i+1]
                try:
                    # 验证是否为合法 JSON
                    json.loads(json_candidate)
                    return json_candidate
                except:
                    # 继续查找下一个
                    start_idx = -1
                    continue
    
    # 策略 2: 使用正则表达式查找
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(json_pattern, text, re.DOTALL)
    for match in matches:
        json_candidate = match.group(0)
        try:
            json.loads(json_candidate)
            return json_candidate
        except:
            continue
    
    return None

def self_repair_json(raw_output, model):
    """
    自修复：将原始输出喂回模型，要求其只输出合法 JSON
    
    Args:
        raw_output: 原始模型输出
        model: 模型名称
    Returns:
        repaired_json_str: 修复后的 JSON 字符串，如果失败返回 None
    """
    repair_prompt = f"""
The following output contains a JSON object but may have extra text or formatting issues.
Please extract and output ONLY the valid JSON object, with NO explanations, NO markdown, NO extra text.

Original output:
{raw_output}

Output ONLY the JSON:
"""
    
    try:
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': repair_prompt}],
            options={"num_ctx": 4096, "temperature": 0}  # 低温度确保确定性输出
        )
        repaired_text = response['message']['content'].strip()
        
        # 尝试提取 JSON
        json_str = extract_json_from_text(repaired_text)
        if json_str:
            # 验证是否合法
            json.loads(json_str)
            return json_str
    except Exception as e:
        pass
    
    return None

def robust_json_parse(raw_content, model, max_retries=1):
    """
    鲁棒 JSON 解析：提取 + 自修复 + 降级返回
    
    Args:
        raw_content: 模型原始输出
        model: 模型名称（用于自修复）
        max_retries: 最大自修复尝试次数
    Returns:
        dict: 解析后的 JSON 对象，或降级错误结构
    """
    # 第一步：尝试直接提取 JSON
    json_str = extract_json_from_text(raw_content)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            pass  # 继续尝试自修复
    
    # 第二步：自修复
    for attempt in range(max_retries):
        repaired_json_str = self_repair_json(raw_content, model)
        if repaired_json_str:
            try:
                return json.loads(repaired_json_str)
            except json.JSONDecodeError:
                continue
    
    # 第三步：降级返回
    return {
        "status": "error",
        "reason": "Failed to parse JSON after extraction and self-repair attempts",
        "raw_output": raw_content[:500] + "..." if len(raw_content) > 500 else raw_content,
        "evidence": [],
        "_parse_error": True
    }

# ================= 2. 基础功能函数 =================

def save_to_memory(summary, impact_score, advice):
    if impact_score < 5: return 
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    collection.add(
        documents=[f"Event: {summary}. Advice: {advice}"],
        metadatas=[{"score": impact_score, "time": timestamp}],
        ids=[str(uuid.uuid4())]
    )

def recall_history(query_text, n_results=2):
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        history = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                history.append(f"- [{meta['time']}] {doc}")
        return "\n".join(history) if history else "No history."
    except: return "Memory Empty."

def send_notification(title, msg):
    if TOAST_AVAILABLE:
        try:
            notification.notify(title=title, message=msg, app_name='GlobalWatch', timeout=10)
        except: pass

def get_full_market_context():
    data = {}
    for name, ticker in MACRO_ANCHORS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty: data[name] = round(hist['Close'].iloc[-1], 2)
        except: data[name] = "N/A"
    return data

def normalize_title(title):
    """归一化标题用于去重：小写 + 去标点 + 去多余空格"""
    import string
    # 转小写
    normalized = title.lower()
    # 移除标点
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    # 去除多余空格
    normalized = ' '.join(normalized.split())
    return normalized

def get_rss_news():
    """
    返回结构化新闻列表
    Returns:
        List[Dict]: [{"source": str, "title": str, "published": str|None, "link": str}]
    """
    news = []
    seen_links = set()
    seen_titles = set()
    
    for src, url in RSS_FEEDS.items():
        try:
            f = feedparser.parse(url)
            src_count = 0
            
            for e in f.entries:
                if src_count >= 2:  # 每个源最多2条
                    break
                
                # 提取字段
                title = e.get('title', '').strip()
                link = e.get('link', '').strip()
                
                if not title or not link:
                    continue
                
                # 去重逻辑 1: 链接完全相同
                if link in seen_links:
                    continue
                
                # 去重逻辑 2: 标题归一化后相同
                normalized_title = normalize_title(title)
                if normalized_title in seen_titles:
                    continue
                
                # 提取发布时间
                published = None
                if hasattr(e, 'published_parsed') and e.published_parsed:
                    try:
                        published = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.published_parsed)
                    except:
                        pass
                elif hasattr(e, 'updated_parsed') and e.updated_parsed:
                    try:
                        published = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.updated_parsed)
                    except:
                        pass
                
                # 添加结构化新闻
                news.append({
                    "source": src,
                    "title": title,
                    "published": published,
                    "link": link
                })
                
                seen_links.add(link)
                seen_titles.add(normalized_title)
                src_count += 1
                
        except Exception as e:
            continue
    
    return news[:8]  # 总数上限

def get_stock_news(ticker_symbol):
    try:
        query = urllib.parse.quote(f"{ticker_symbol} stock news")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        f = feedparser.parse(rss_url)
        headlines = []
        for e in f.entries[:5]:
            clean_title = e.title.split(' - ')[0]
            headlines.append(f"[News] {clean_title}")
        return headlines if headlines else ["No recent news found."]
    except Exception as e: return [f"Error fetching news: {str(e)}"]

def plot_candle_chart(ticker, title, height=300):
    try:
        df = yf.Ticker(ticker).history(period="3mo")
        if df.empty: return
        df['MA20'] = df['Close'].rolling(window=20).mean()
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'))
        fig.update_layout(height=height, margin=dict(l=0,r=0,t=30,b=0), title=dict(text=title, font=dict(color="white")), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
    except: st.caption("No Chart Data")

def plot_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Risk Sentiment (0-10)"},
        gauge = {
            'axis': {'range': [0, 10]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 3], 'color': "green"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=0,b=0))
    st.plotly_chart(fig)

def get_cross_rate(asset_a, asset_b):
    def get_val(name):
        info = ASSETS_DB.get(name)
        if not info: return None
        if info['ticker'] == "USD": return 1.0
        try:
            h = yf.Ticker(info['ticker']).history(period="1d")
            return 1.0/h['Close'].iloc[-1] if info['type'] == "fiat_quote" else h['Close'].iloc[-1]
        except: return None
    v1, v2 = get_val(asset_a), get_val(asset_b)
    return v1/v2 if v1 and v2 else None

# ================= 3. Evidence 验证函数 =================

def validate_evidence(evidence_list, input_news):
    """
    验证 AI 返回的 evidence 是否引用了真实的输入新闻
    Args:
        evidence_list: AI 返回的 evidence 数组
        input_news: 结构化新闻列表 List[Dict] with keys: source, title, published, link
    Returns:
        validated_evidence: 验证后的 evidence 列表（无效的标记 _invalid）
        valid_count: 有效证据数量
    """
    validated = []
    valid_count = 0
    
    for ev in evidence_list:
        headline = ev.get('headline', '').strip()
        is_valid = False
        
        # 检查 headline 是否存在于任何输入新闻的 title 中（子串匹配）
        for news_item in input_news:
            news_title = news_item.get('title', '')
            # 双向子串匹配
            if headline.lower() in news_title.lower() or news_title.lower() in headline.lower():
                is_valid = True
                break
        
        if is_valid:
            valid_count += 1
        else:
            ev['_invalid'] = True
            ev['_warning'] = 'Headline not found in input news (possible hallucination)'
        
        validated.append(ev)
    
    return validated, valid_count

# ================= 4. AI 分析核心 (DeepSeek Logic with Evidence) =================

def analyze_all(news, user_pairs, macro_data, lang_mode):
    if not news: return {"status": "no_update"}
    
    # 将结构化新闻转换为文本用于 prompt
    headlines = " ".join([f"[{item['source']}] {item['title']}" for item in news])
    history = recall_history(headlines)
    lang_instruction = "OUTPUT LANGUAGE: CHINESE (Simplified)" if lang_mode == "中文" else "OUTPUT LANGUAGE: ENGLISH"

    # 【核心改进】注入 MACRO_LOGIC_KNOWLEDGE + 强制 evidence 输出
    prompt = f"""
    You are a Financial Logic Engine. {lang_instruction}
    
    MACRO RULES (You MUST reference these rules in your analysis):
    {MACRO_LOGIC_KNOWLEDGE}
    
    CONTEXT:
    - News Headlines: {headlines}
    - Macro Data: {json.dumps(macro_data)}
    - Historical Memory: {history}
    
    TARGET PAIRS: {", ".join(user_pairs)}
    
    CRITICAL REQUIREMENTS:
    1. First, THINK deeply (<think>...</think>) about causal chains using the MACRO RULES above.
    2. Extract EVIDENCE from the News Headlines (you MUST quote actual headlines, DO NOT fabricate).
    3. Link each prediction to specific evidence and macro rules.
    4. If no relevant news exists, set status to "no_update" and evidence to empty array.

    STRICT JSON OUTPUT FORMAT:
    {{
        "status": "alert" or "no_update",
        "impact_score": 0-10,
        "summary": "Brief event description",
        "evidence": [
            {{
                "source": "Reuters|CNBC|BBC",
                "headline": "EXACT headline from input news",
                "why_it_matters": "Explain how this triggers MACRO RULE X and affects asset Y"
            }}
        ],
        "predictions": {{ "Pair": "Bullish/Bearish (based on evidence)" }},
        "advice": "Actionable advice based on evidence"
    }}
    
    VALIDATION RULES:
    - evidence.headline MUST be a substring of the input News Headlines
    - If evidence is empty, predictions must indicate "insufficient evidence"
    - summary/predictions/advice MUST be traceable to evidence items
    """
    
    try:
        # 增加 num_ctx 防止思考过程太长被截断
        response = ollama.chat(model=LOCAL_MODEL, messages=[{'role': 'user', 'content': prompt}], options={"num_ctx": 8192})
        raw_content = response['message']['content']
        
        # 【鲁棒解析】使用 robust_json_parse 替代直接 json.loads
        thought, json_text = parse_deepseek_output(raw_content)
        
        # 尝试鲁棒解析
        res = robust_json_parse(json_text, LOCAL_MODEL, max_retries=1)
        
        # 如果解析失败（返回降级结构），直接返回
        if res.get('_parse_error'):
            res['thought_process'] = thought
            return res
        
        # 解析成功，继续处理
        res['thought_process'] = thought
        
        # 【新增】验证 evidence 字段（传入结构化新闻）
        evidence = res.get('evidence', [])
        validated_evidence, valid_count = validate_evidence(evidence, news)
        res['evidence'] = validated_evidence
        res['_valid_evidence_count'] = valid_count
        
        # 【新增】证据不足降级策略
        if valid_count == 0 and res.get('status') == 'alert':
            res['_evidence_warning'] = True
            original_advice = res.get('advice', '')
            res['advice'] = f"{original_advice}\n\n⚠️ WARNING: No valid evidence found. Predictions may be unreliable. Please verify independently."
        
        if res.get("status") == "alert":
            save_to_memory(res.get("summary"), res.get("impact_score", 0), res.get("advice"))
        return res
    except Exception as e: 
        # 最终兜底：返回降级结构
        return {
            "status": "error",
            "reason": f"Unexpected error: {str(e)}",
            "raw_output": "",
            "evidence": [],
            "_parse_error": True
        }

def analyze_single_stock(ticker, news, lang_mode):
    lang_instruction = "OUTPUT LANGUAGE: CHINESE (Simplified)" if lang_mode == "中文" else "OUTPUT LANGUAGE: ENGLISH"
    news_str = " ".join(news)
    
    prompt = f"""
    You are a Wall Street Analyst. {lang_instruction}
    Stock: {ticker}
    News: {news_str}
    
    TASK:
    1. Think about the market sentiment and risks.
    2. Output JSON.
    
    STRICT JSON OUTPUT FORMAT:
    {{
        "sentiment": "Bullish/Bearish/Neutral",
        "reason": "...",
        "key_risk": "..."
    }}
    """
    try:
        response = ollama.chat(model=LOCAL_MODEL, messages=[{'role': 'user', 'content': prompt}], options={"num_ctx": 8192})
        raw_content = response['message']['content']
        thought, json_text = parse_deepseek_output(raw_content)
        
        # 【鲁棒解析】使用 robust_json_parse
        res = robust_json_parse(json_text, LOCAL_MODEL, max_retries=1)
        
        # 如果解析失败，返回降级结构
        if res.get('_parse_error'):
            return {
                "sentiment": "AI Error",
                "reason": f"Parse Error: {res.get('reason', 'Unknown')}",
                "key_risk": "Unable to analyze due to parsing failure",
                "thought_process": thought
            }
        
        res['thought_process'] = thought
        return res
    except Exception as e:
        return {"sentiment": "AI Error", "reason": f"Parse Error: {str(e)}", "key_risk": "N/A"}

# ================= 4. UI 界面 =================

st.set_page_config(page_title="GlobalWatch DeepSeek Edition", layout="wide", page_icon="🦁")

st.sidebar.header("⚙️ Settings")
st.sidebar.caption(f"Brain: {LOCAL_MODEL}") # 显示当前模型

# 新增：展示宏观规则库
with st.sidebar.expander("📚 Macro Rules Library"):
    st.text(MACRO_LOGIC_KNOWLEDGE)

lang_mode = st.sidebar.radio("Language", ["中文", "English"], index=0)
refresh_label = st.sidebar.selectbox("Refresh Rate", list(REFRESH_OPTIONS.keys()), index=0)
refresh_sec = REFRESH_OPTIONS[refresh_label]
enable_toast = st.sidebar.checkbox("Desktop Notify", value=True)
auto_run = st.sidebar.checkbox("Auto Run", value=True)

if 'last_run' not in st.session_state: st.session_state['last_run'] = datetime.now() - timedelta(days=1)

st.title("🦁 GlobalWatch: DeepSeek-R1 推理版")
st.caption("🚀 Powered by Chain-of-Thought Reasoning")
st.divider()

tab_macro, tab_stock = st.tabs(["🌍 宏观/外汇 (Macro/FX)", "🇺🇸 美股透视 (US Stocks)"])

# === TAB 1: 宏观外汇 ===
with tab_macro:
    cols = st.columns(4)
    macro = get_full_market_context()
    for i, (k, v) in enumerate(macro.items()): cols[i].metric(k, f"${v}")
    st.divider()
    
    c1, c2, c3 = st.columns([2, 2, 1]) 
    user_pairs = []
    
    with c1:
        with st.container(border=True):
            b1 = st.selectbox("Base", list(ASSETS_DB.keys()), index=1, key="a1") 
            q1 = st.selectbox("Quote", list(ASSETS_DB.keys()), index=2, key="a2") 
            r1 = get_cross_rate(b1, q1)
            if r1: 
                st.metric(f"{b1.split()[0]}/{q1.split()[0]}", f"{r1:,.4f}")
                if b1 != "USD (美元)": plot_candle_chart(ASSETS_DB[b1]['ticker'], b1)
                user_pairs.append(f"{b1.split()[0]}/{q1.split()[0]}")

    with c2:
        with st.container(border=True):
            b2 = st.selectbox("Base", list(ASSETS_DB.keys()), index=6, key="b1") 
            q2 = st.selectbox("Quote", list(ASSETS_DB.keys()), index=0, key="b2") 
            r2 = get_cross_rate(b2, q2)
            if r2: 
                st.metric(f"{b2.split()[0]}/{q2.split()[0]}", f"{r2:,.4f}")
                plot_candle_chart(ASSETS_DB[b2]['ticker'], b2)
                user_pairs.append(f"{b2.split()[0]}/{q2.split()[0]}")
    
    with c3:
        st.caption("AI Risk Gauge")
        score = st.session_state.get('res', {}).get('impact_score', 0)
        plot_gauge(score)

    delta = (datetime.now() - st.session_state['last_run']).total_seconds()
    remain = max(0, refresh_sec - delta) if refresh_sec > 0 else 0
    
    if st.button("🚀 Deep Reason Analysis") or (refresh_sec > 0 and remain == 0 and auto_run):
        with st.status("🧠 DeepSeek is thinking...", expanded=True) as s:
            news = get_rss_news()
            res = analyze_all(news, user_pairs, macro, lang_mode)
            
            if enable_toast and res.get("status") == "alert" and res.get("impact_score", 0) >= 7:
                send_notification("Market Alert", res.get("summary"))
                
            st.session_state['last_run'] = datetime.now()
            st.session_state['res'] = res
            st.session_state['news'] = news
            s.update(label="Reasoning Complete", state="complete", expanded=False)
            st.rerun()

    if 'res' in st.session_state:
        res = st.session_state['res']
        
        # === 新增：解析错误处理 ===
        if res.get('_parse_error'):
            st.error("🚨 AI Output Parsing Error")
            st.markdown(f"**Reason**: {res.get('reason', 'Unknown error')}")
            
            with st.expander("🔍 Raw Output (Debug)", expanded=False):
                st.code(res.get('raw_output', 'No output available'), language="text")
            
            st.warning("⚠️ The AI failed to generate valid JSON output. This may be due to:")
            st.markdown("""
            - Model output format issues
            - Context length exceeded
            - Unexpected model behavior
            
            **Suggested actions**:
            - Try again with a different model
            - Reduce the number of news items
            - Check Ollama logs for errors
            """)
            
            # 仍然显示思维过程（如果有）
            if res.get('thought_process'):
                with st.expander("🧠 DeepSeek 的思维过程 (Click to expand)", expanded=False):
                    st.markdown(res.get('thought_process', 'No thoughts recorded.'))
        # ================================
        
        # === V3.0 新增：展示思维链 ===
        elif res.get("status") != "error":
            with st.expander("🧠 DeepSeek 的思维过程 (Click to expand)", expanded=False):
                st.markdown(res.get('thought_process', 'No thoughts recorded.'))
        # ==========================

        if res.get("status") == "alert":
            st.error(f"🚨 ALERT (Score: {res.get('impact_score')})")
            st.markdown(f"**Event**: {res.get('summary')}")
            
            # === 新增：Evidence Chain 展示 ===
            evidence = res.get('evidence', [])
            valid_count = res.get('_valid_evidence_count', 0)
            
            if evidence:
                with st.expander(f"📋 Evidence Chain ({valid_count}/{len(evidence)} valid)", expanded=True):
                    for idx, ev in enumerate(evidence, 1):
                        is_invalid = ev.get('_invalid', False)
                        icon = "⚠️" if is_invalid else "✅"
                        
                        st.markdown(f"**{icon} Evidence {idx}**")
                        st.markdown(f"- **Source**: {ev.get('source', 'Unknown')}")
                        st.markdown(f"- **Headline**: _{ev.get('headline', 'N/A')}_")
                        st.markdown(f"- **Why it matters**: {ev.get('why_it_matters', 'N/A')}")
                        
                        if is_invalid:
                            st.warning(ev.get('_warning', 'Invalid evidence'))
                        st.divider()
            
            if res.get('_evidence_warning'):
                st.warning("⚠️ No valid evidence found. AI predictions may be unreliable.")
            # ================================
            
            col_p, col_a = st.columns(2)
            col_p.write(res.get("predictions"))
            col_a.warning(res.get("advice"))
        else:
            st.success("✅ Market is Stable")
            st.caption(res.get("advice"))
        
        with st.expander("📰 News Source"):
            news_list = st.session_state.get('news', [])
            if news_list:
                for idx, news_item in enumerate(news_list, 1):
                    # 结构化新闻展示
                    source = news_item.get('source', 'Unknown')
                    title = news_item.get('title', 'N/A')
                    published = news_item.get('published', None)
                    link = news_item.get('link', '')
                    
                    # 格式化时间显示
                    time_str = ""
                    if published:
                        try:
                            # 转换为更友好的格式
                            from datetime import datetime
                            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                            time_str = f"🕒 {dt.strftime('%Y-%m-%d %H:%M UTC')}"
                        except:
                            time_str = f"🕒 {published}"
                    
                    # 显示新闻
                    st.markdown(f"**{idx}. [{source}]** {title}")
                    if time_str:
                        st.caption(time_str)
                    if link:
                        st.markdown(f"[🔗 Read More]({link})")
                    st.divider()
            else:
                st.caption("No news available")

# === TAB 2: 美股个股分析 ===
with tab_stock:
    st.header("🇺🇸 US Stock Deep Dive")
    c_in, c_go = st.columns([3, 1])
    ticker = c_in.text_input("Ticker", value="NVDA").upper()
    
    if c_go.button("🔍 Analyze"):
        with st.spinner(f"Reasoning about {ticker}..."):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                price = hist['Close'].iloc[-1]
                change = (price - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1] * 100
                
                st.metric(label=ticker, value=f"${price:.2f}", delta=f"{change:.2f}%")
                plot_candle_chart(ticker, f"{ticker} Price Action")
                
                stock_news = get_stock_news(ticker)
                if stock_news:
                    with st.expander("Latest News"):
                        for n in stock_news: st.write(n)
                    
                    analysis = analyze_single_stock(ticker, stock_news, lang_mode)
                    
                    # === V3.0 新增：展示个股思维链 ===
                    with st.expander("🧠 AI Thought Process (Stock)", expanded=True):
                        st.markdown(analysis.get('thought_process', 'No thoughts.'))
                    
                    sentiment = analysis.get("sentiment", "Neutral")
                    box_col = "green" if "Bullish" in sentiment else "red" if "Bearish" in sentiment else "gray"
                    
                    st.markdown(f"""
                    <div style="padding:10px; border-left: 5px solid {box_col}; background-color: #262730;">
                        <h3>{sentiment}</h3>
                        <p><b>Reason:</b> {analysis.get('reason')}</p>
                        <p><i>Risk: {analysis.get('key_risk')}</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No news found.")
            except Exception as e:
                st.error(f"Error: {e}")