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

# 尝试导入通知库
try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# ================= 1. 核心配置 =================

LOCAL_MODEL = "qwen2.5:7b" 

# 初始化 ChromaDB (记忆数据库)
chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="market_events")

# 宏观逻辑库 (加入了更严格的方向性规则)
MACRO_LOGIC_KNOWLEDGE = """
GLOBAL MACRO RULES:
1. CAD (Loonie) is a Petro-currency. Correlation: Oil Price UP -> CAD Stronger.
2. CNY (Yuan) is sensitive to USD Strength & Trade Wars.
3. USD is Safe Haven. Crisis -> Capital flows to USD/Gold.
4. Gold (XAU) is inflation hedge.
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

REFRESH_OPTIONS = {"手动": 0, "5 分钟": 300, "10 分钟": 600, "30 分钟": 1800, "1 小时": 3600}

# ================= 2. 记忆与通知模块 =================

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
        return "\n".join(history) if history else "No relevant history."
    except: return "Memory Empty."

def send_notification(title, msg):
    if TOAST_AVAILABLE:
        try:
            toaster = ToastNotifier()
            toaster.show_toast(title, msg, duration=10, threaded=True) 
        except: pass

# ================= 3. 数据与图表模块 =================

def get_full_market_context():
    data = {}
    for name, ticker in MACRO_ANCHORS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty: data[name] = round(hist['Close'].iloc[-1], 2)
        except: data[name] = "N/A"
    return data

def get_rss_news():
    news = []
    seen = set()
    for src, url in RSS_FEEDS.items():
        try:
            f = feedparser.parse(url)
            for e in f.entries[:2]:
                if e.title not in seen:
                    news.append(f"[{src}] {e.title}")
                    seen.add(e.title)
        except: continue
    return news[:10]

def plot_candle_chart(asset_name):
    info = ASSETS_DB.get(asset_name)
    if not info or info['ticker'] == "USD": return
    try:
        df = yf.Ticker(info['ticker']).history(period="3mo")
        if df.empty: return
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'))
        
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
    except: st.caption("No Chart Data")

def get_cross_rate(asset_a, asset_b):
    def get_val(name):
        info = ASSETS_DB.get(name)
        if not info: return None
        if info['ticker'] == "USD": return 1.0
        try:
            h = yf.Ticker(info['ticker']).history(period="1d")
            return 1.0/h['Close'].iloc[-1] if info['type'] == "fiat_quote" else h['Close'].iloc[-1]
        except: return None
    
    val_a = get_val(asset_a)
    val_b = get_val(asset_b)
    if val_a and val_b: return val_a / val_b
    return None

# ================= 4. AI 分析逻辑 (核心修复) =================

def clean_llm_output(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def analyze_all(news, user_pairs, macro_data):
    if not news: return {"status": "no_update"}
    
    headlines = " ".join(news)
    history = recall_history(headlines)
    
    # === 核心修复：更严格的 Prompt ===
    prompt = f"""
    You are a Professional FX Strategist. 
    
    1. MARKET CONTEXT:
    - News: {headlines}
    - Macro Data: {json.dumps(macro_data)}
    - Memory: {history}
    - Rules: {MACRO_LOGIC_KNOWLEDGE}
    
    2. USER WATCHLIST (Target Pairs):
    {", ".join(user_pairs)}
    
    3. STRICT DIRECTION LOGIC (CRITICAL):
    - For a pair "BASE/QUOTE" (e.g., CNY/CAD):
      - If BASE gets stronger -> BULLISH (↑).
      - If QUOTE gets stronger -> BEARISH (↓).
      - If BASE gets weaker -> BEARISH (↓).
      - If QUOTE gets weaker -> BULLISH (↑).
    - LOGIC CHECK: If you say A/B is Bullish, B/A MUST be Bearish.
    - Example: Oil UP -> CAD Strong.
      - CAD/CNY = Bullish.
      - CNY/CAD = Bearish.

    TASK:
    Analyze direction for each pair in Watchlist using the Strict Logic above.
    
    OUTPUT JSON:
    {{
        "status": "alert" (or "no_update"),
        "impact_score": 0-10,
        "summary": "Key Event",
        "trend_analysis": "Memory context...",
        "predictions": {{ 
            "Pair_Name": "Bullish/Bearish (Reasoning: Base X is stronger/weaker than Quote Y)" 
        }},
        "advice": "Action",
        "upcoming_events": ["Event 1"]
    }}
    """
    
    try:
        response = ollama.chat(model=LOCAL_MODEL, messages=[{'role': 'user', 'content': prompt}], options={"num_ctx": 4096})
        content = clean_llm_output(response['message']['content'])
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            res = json.loads(match.group(0))
            if res.get("status") == "alert":
                save_to_memory(res.get("summary"), res.get("impact_score", 0), res.get("advice"))
            return res
        return {"status": "error"}
    except Exception as e: return {"status": "error", "msg": str(e)}

# ================= 5. 前端 UI (格式修复) =================

st.set_page_config(page_title="GlobalWatch GodMode", layout="wide", page_icon="🦁")

st.sidebar.header("🕹️ 终极控制台")
refresh_label = st.sidebar.selectbox("刷新频率", list(REFRESH_OPTIONS.keys()), index=2)
refresh_sec = REFRESH_OPTIONS[refresh_label]
enable_toast = st.sidebar.checkbox("开启桌面弹窗", value=True)
auto_run = st.sidebar.checkbox("开启自动运行", value=True)

if 'last_run' not in st.session_state: st.session_state['last_run'] = datetime.now() - timedelta(days=1)

st.title("🦁 GlobalWatch: 全能金融终端 (修正版)")
st.caption("Fix: 修复了方向性逻辑错误 (Base/Quote Logic Correction)")
st.divider()

cols = st.columns(4)
macro = get_full_market_context()
for i, (k, v) in enumerate(macro.items()): cols[i].metric(k, f"${v}")
st.divider()

# --- 监控区 ---
c1, c2 = st.columns(2)
user_pairs = []

with c1:
    with st.container(border=True):
        b1 = st.selectbox("Base A", list(ASSETS_DB.keys()), index=1, key="a1") # CNY
        q1 = st.selectbox("Quote A", list(ASSETS_DB.keys()), index=2, key="a2") # CAD
        r1 = get_cross_rate(b1, q1)
        if r1: 
            st.metric(f"{b1.split()[0]}/{q1.split()[0]}", f"{r1:,.4f}")
            if b1 != "USD (美元)": plot_candle_chart(b1)
            # 修正：发送标准格式 CNY/CAD
            user_pairs.append(f"{b1.split()[0]}/{q1.split()[0]}")

with c2:
    with st.container(border=True):
        b2 = st.selectbox("Base B", list(ASSETS_DB.keys()), index=2, key="b1") # CAD (这里特意设反过来测试)
        q2 = st.selectbox("Quote B", list(ASSETS_DB.keys()), index=1, key="b2") # CNY
        r2 = get_cross_rate(b2, q2)
        if r2: 
            st.metric(f"{b2.split()[0]}/{q2.split()[0]}", f"{r2:,.4f}")
            plot_candle_chart(b2)
            # 修正：发送标准格式 CAD/CNY
            user_pairs.append(f"{b2.split()[0]}/{q2.split()[0]}")

st.divider()

delta = (datetime.now() - st.session_state['last_run']).total_seconds()
remain = max(0, refresh_sec - delta) if refresh_sec > 0 else 0

c_prog, c_btn = st.columns([4, 1])
if refresh_sec > 0:
    c_prog.progress(min(delta / refresh_sec, 1.0), f"下次扫描: {int(remain)}s")
force = c_btn.button("🚀 立即扫描")

if force or (refresh_sec > 0 and remain == 0 and auto_run):
    with st.status("🧠 深度逻辑分析中 (Strict Mode)...", expanded=True) as s:
        s.write("📡 读取数据 & 绘制图表...")
        news = get_rss_news()
        
        start_t = time.time()
        res = analyze_all(news, user_pairs, macro)
        
        if enable_toast and res.get("status") == "alert" and res.get("impact_score", 0) >= 7:
            send_notification("GlobalWatch 警报", res.get("summary"))
            
        st.session_state['last_run'] = datetime.now()
        st.session_state['res'] = res
        st.session_state['news'] = news
        s.update(label="✅ 完成", state="complete", expanded=False)
        st.rerun()

if 'res' in st.session_state:
    res = st.session_state['res']
    
    if res.get("status") == "alert":
        st.error(f"🚨 **警报 (Impact: {res.get('impact_score')})**")
        st.markdown(f"### {res.get('summary')}")
        st.info(f"📈 **趋势**: {res.get('trend_analysis')}")
        
        col_p, col_a = st.columns(2)
        col_p.write(res.get("predictions"))
        col_a.warning(res.get("advice"))
    else:
        st.success("✅ 市场平静")
        st.caption("AI 未发现重大风险。")

    st.markdown("### 📅 未来关注")
    evts = res.get("upcoming_events", [])
    if evts:
        for e in evts: st.markdown(f"- ⏰ {e}")
    else: st.caption("暂无明确日程")

    with st.expander("查看新闻源"):
        for n in st.session_state.get('news', []): st.write(n)