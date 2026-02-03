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

# 初始化信号追踪数据库
signals_collection = chroma_client.get_or_create_collection(name="trading_signals")

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

# ================= 2.5. Signal Scoreboard 系统 =================

def get_asset_ticker(asset_name):
    """
    从资产名称获取 ticker
    Args:
        asset_name: 如 "CNY/CAD", "Oil", "NVDA"
    Returns:
        ticker: yfinance ticker 或 None
    """
    # 处理货币对
    if '/' in asset_name:
        parts = asset_name.split('/')
        base, quote = parts[0].strip(), parts[1].strip()
        
        # 查找对应的 ticker
        for name, info in ASSETS_DB.items():
            if base in name:
                return info['ticker']
        
        # 如果是外汇对，尝试构造
        if base != 'USD' and quote == 'USD':
            return f"{base}=X"
        elif base == 'USD' and quote != 'USD':
            return f"{quote}=X"
    
    # 处理商品
    if asset_name.lower() in ['oil', 'crude oil', 'crude']:
        return "CL=F"
    if asset_name.lower() in ['gold', 'xau']:
        return "GC=F"
    
    # 处理个股（直接返回）
    if asset_name.isupper() and len(asset_name) <= 5:
        return asset_name
    
    return None

def get_current_price(ticker):
    """
    获取当前价格
    Args:
        ticker: yfinance ticker
    Returns:
        price: float 或 None
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def record_signal(asset, direction, confidence, predictions_dict, news_sources):
    """
    记录交易信号
    Args:
        asset: 资产名称
        direction: Bullish/Bearish/Neutral
        confidence: 信心分数 (0-10)
        predictions_dict: 完整的 predictions 字典
        news_sources: 新闻来源列表
    """
    try:
        signal_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 获取 ticker 和当前价格
        ticker = get_asset_ticker(asset)
        current_price = get_current_price(ticker) if ticker else None
        
        # 确定主题
        theme = "UNKNOWN"
        if '/' in asset:
            theme = "FX"
        elif asset.upper() in ['OIL', 'GOLD', 'CRUDE']:
            theme = "MACRO"
        elif ticker and len(asset) <= 5 and asset.isupper():
            theme = "STOCK"
        
        # 提取新闻来源
        sources = list(set([src for src in news_sources if src]))
        
        # 构造元数据
        metadata = {
            "signal_id": signal_id,
            "timestamp": timestamp,
            "asset": asset,
            "ticker": ticker or "UNKNOWN",
            "direction": direction,
            "confidence": float(confidence),
            "theme": theme,
            "initial_price": float(current_price) if current_price else 0.0,
            "sources": ",".join(sources[:3]),  # 最多3个来源
            "status": "PENDING",  # PENDING / VERIFIED
            # 回填字段（初始为空）
            "price_1h": 0.0,
            "price_4h": 0.0,
            "price_1d": 0.0,
            "price_1w": 0.0,
            "correct_1h": "",
            "correct_4h": "",
            "correct_1d": "",
            "correct_1w": "",
            "return_1h": 0.0,
            "return_4h": 0.0,
            "return_1d": 0.0,
            "return_1w": 0.0
        }
        
        # 存储到 ChromaDB
        signals_collection.add(
            documents=[json.dumps(predictions_dict)],
            metadatas=[metadata],
            ids=[signal_id]
        )
        
        return signal_id
    except Exception as e:
        print(f"Error recording signal: {e}")
        return None

def backfill_signal_results():
    """
    回填信号结果
    检查所有 PENDING 信号，如果时间到了就回填价格和结果
    """
    try:
        # 获取所有 PENDING 信号
        results = signals_collection.get(
            where={"status": "PENDING"}
        )
        
        if not results or not results['ids']:
            return
        
        now = datetime.now()
        updated_count = 0
        
        for i, signal_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            
            signal_time = datetime.fromisoformat(metadata['timestamp'])
            ticker = metadata['ticker']
            initial_price = metadata['initial_price']
            direction = metadata['direction']
            
            if ticker == "UNKNOWN" or initial_price == 0.0:
                continue
            
            # 计算时间差
            time_diff = (now - signal_time).total_seconds() / 3600  # 小时
            
            updated = False
            
            # 回填 1h
            if time_diff >= 1 and metadata['price_1h'] == 0.0:
                price_1h = get_historical_price(ticker, signal_time + timedelta(hours=1))
                if price_1h:
                    metadata['price_1h'] = price_1h
                    metadata['return_1h'] = (price_1h - initial_price) / initial_price * 100
                    metadata['correct_1h'] = check_direction(direction, metadata['return_1h'])
                    updated = True
            
            # 回填 4h
            if time_diff >= 4 and metadata['price_4h'] == 0.0:
                price_4h = get_historical_price(ticker, signal_time + timedelta(hours=4))
                if price_4h:
                    metadata['price_4h'] = price_4h
                    metadata['return_4h'] = (price_4h - initial_price) / initial_price * 100
                    metadata['correct_4h'] = check_direction(direction, metadata['return_4h'])
                    updated = True
            
            # 回填 1d
            if time_diff >= 24 and metadata['price_1d'] == 0.0:
                price_1d = get_historical_price(ticker, signal_time + timedelta(days=1))
                if price_1d:
                    metadata['price_1d'] = price_1d
                    metadata['return_1d'] = (price_1d - initial_price) / initial_price * 100
                    metadata['correct_1d'] = check_direction(direction, metadata['return_1d'])
                    updated = True
            
            # 回填 1w
            if time_diff >= 168 and metadata['price_1w'] == 0.0:
                price_1w = get_historical_price(ticker, signal_time + timedelta(weeks=1))
                if price_1w:
                    metadata['price_1w'] = price_1w
                    metadata['return_1w'] = (price_1w - initial_price) / initial_price * 100
                    metadata['correct_1w'] = check_direction(direction, metadata['return_1w'])
                    metadata['status'] = "VERIFIED"  # 全部回填完成
                    updated = True
            
            # 更新元数据
            if updated:
                signals_collection.update(
                    ids=[signal_id],
                    metadatas=[metadata]
                )
                updated_count += 1
        
        return updated_count
    except Exception as e:
        print(f"Error backfilling signals: {e}")
        return 0

def get_historical_price(ticker, target_time):
    """
    获取历史价格（尽可能接近目标时间）
    """
    try:
        t = yf.Ticker(ticker)
        # 获取目标时间前后1天的数据
        start = target_time - timedelta(days=1)
        end = target_time + timedelta(days=1)
        hist = t.history(start=start, end=end, interval="1h")
        
        if not hist.empty:
            # 找到最接近目标时间的价格
            closest_idx = (hist.index - target_time).abs().argmin()
            return float(hist['Close'].iloc[closest_idx])
    except:
        pass
    return None

def check_direction(predicted_direction, actual_return):
    """
    检查方向是否正确
    Args:
        predicted_direction: Bullish/Bearish/Neutral
        actual_return: 实际收益率 (%)
    Returns:
        "CORRECT" / "WRONG" / "NEUTRAL"
    """
    if predicted_direction == "Neutral":
        return "NEUTRAL"
    
    if predicted_direction == "Bullish":
        return "CORRECT" if actual_return > 0 else "WRONG"
    elif predicted_direction == "Bearish":
        return "CORRECT" if actual_return < 0 else "WRONG"
    
    return "UNKNOWN"

def get_signal_statistics(theme=None, asset=None, timeframe="1d"):
    """
    获取信号统计
    Args:
        theme: 主题过滤 (FX/MACRO/STOCK/None)
        asset: 资产过滤 (None 表示全部)
        timeframe: 时间框架 (1h/4h/1d/1w)
    Returns:
        dict: 统计数据
    """
    try:
        # 构造查询条件
        where_clause = {}
        if theme:
            where_clause["theme"] = theme
        if asset:
            where_clause["asset"] = asset
        
        # 获取信号
        if where_clause:
            results = signals_collection.get(where=where_clause)
        else:
            results = signals_collection.get()
        
        if not results or not results['ids']:
            return {
                "total_signals": 0,
                "accuracy": 0.0,
                "avg_return": 0.0,
                "max_return": 0.0,
                "min_return": 0.0,
                "sample_size": 0,
                "statistical_significance": False
            }
        
        # 提取对应时间框架的数据
        correct_field = f"correct_{timeframe}"
        return_field = f"return_{timeframe}"
        
        correct_count = 0
        wrong_count = 0
        returns = []
        
        for metadata in results['metadatas']:
            correct_status = metadata.get(correct_field, "")
            return_value = metadata.get(return_field, 0.0)
            
            if correct_status == "CORRECT":
                correct_count += 1
                returns.append(return_value)
            elif correct_status == "WRONG":
                wrong_count += 1
                returns.append(return_value)
        
        total_verified = correct_count + wrong_count
        
        if total_verified == 0:
            accuracy = 0.0
            avg_return = 0.0
        else:
            accuracy = correct_count / total_verified * 100
            avg_return = sum(returns) / len(returns) if returns else 0.0
        
        return {
            "total_signals": len(results['ids']),
            "verified_signals": total_verified,
            "accuracy": accuracy,
            "avg_return": avg_return,
            "max_return": max(returns) if returns else 0.0,
            "min_return": min(returns) if returns else 0.0,
            "sample_size": total_verified,
            "statistical_significance": total_verified >= 30  # 至少30个样本
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {
            "total_signals": 0,
            "accuracy": 0.0,
            "avg_return": 0.0,
            "sample_size": 0,
            "statistical_significance": False
        }

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
        
        # 【新增】记录交易信号
        if res.get("status") == "alert" and res.get("predictions"):
            predictions = res.get("predictions", {})
            impact_score = res.get("impact_score", 0)
            news_sources = [item.get('source') for item in news]
            
            # 为每个预测记录信号
            for asset, prediction_text in predictions.items():
                # 提取方向
                direction = "Neutral"
                if "Bullish" in prediction_text or "bullish" in prediction_text or "↑" in prediction_text:
                    direction = "Bullish"
                elif "Bearish" in prediction_text or "bearish" in prediction_text or "↓" in prediction_text:
                    direction = "Bearish"
                
                # 记录信号
                record_signal(
                    asset=asset,
                    direction=direction,
                    confidence=impact_score,
                    predictions_dict=predictions,
                    news_sources=news_sources
                )
        
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

tab_macro, tab_stock, tab_scoreboard = st.tabs(["🌍 宏观/外汇 (Macro/FX)", "🇺🇸 美股透视 (US Stocks)", "📊 Signal Scoreboard"])

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


# === TAB 3: Signal Scoreboard ===
with tab_scoreboard:
    st.header("📊 Signal Scoreboard - Performance Tracking")
    st.caption("Track the accuracy and profitability of AI predictions over time")
    
    # 回填按钮
    col_refresh, col_info = st.columns([1, 3])
    if col_refresh.button("🔄 Update Results"):
        with st.spinner("Backfilling signal results..."):
            updated = backfill_signal_results()
            if updated:
                st.success(f"✅ Updated {updated} signals")
            else:
                st.info("No signals to update")
            st.rerun()
    
    col_info.caption("Click to check and update signal results based on actual market movements")
    
    st.divider()
    
    # 过滤器
    col_theme, col_timeframe = st.columns(2)
    theme_filter = col_theme.selectbox(
        "Theme Filter",
        ["All", "FX", "MACRO", "STOCK"],
        index=0
    )
    timeframe = col_timeframe.selectbox(
        "Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=2
    )
    
    theme = None if theme_filter == "All" else theme_filter
    
    # 获取统计数据
    stats = get_signal_statistics(theme=theme, timeframe=timeframe)
    
    # 显示关键指标
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Total Signals",
        stats['total_signals'],
        help="Total number of predictions made"
    )
    
    col2.metric(
        "Verified Signals",
        stats['verified_signals'],
        help=f"Signals with {timeframe} results available"
    )
    
    # 准确率颜色
    accuracy = stats['accuracy']
    accuracy_delta = accuracy - 50  # 相对于随机猜测
    col3.metric(
        "Accuracy",
        f"{accuracy:.1f}%",
        f"{accuracy_delta:+.1f}% vs random",
        delta_color="normal" if accuracy_delta > 0 else "inverse"
    )
    
    # 平均收益颜色
    avg_return = stats['avg_return']
    col4.metric(
        "Avg Return",
        f"{avg_return:+.2f}%",
        "per signal",
        delta_color="normal" if avg_return > 0 else "inverse"
    )
    
    st.divider()
    
    # 统计显著性警告
    if not stats['statistical_significance']:
        st.warning(f"""
        ⚠️ **Statistical Significance Warning**
        
        Sample size: {stats['sample_size']} (minimum 30 required)
        
        The current sample size is too small to draw reliable conclusions. 
        Continue running analyses to build a larger dataset.
        """)
    else:
        st.success(f"✅ Sample size: {stats['sample_size']} - Statistically significant")
    
    # 详细统计
    st.subheader("📊 Detailed Statistics")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Return Distribution**")
        st.metric("Max Return", f"{stats['max_return']:+.2f}%")
        st.metric("Min Return", f"{stats['min_return']:+.2f}%")
        st.metric("Avg Return", f"{stats['avg_return']:+.2f}%")
    
    with col_b:
        st.markdown("**Performance Analysis**")
        
        # 判断是否赚钱
        if stats['sample_size'] > 0:
            if stats['accuracy'] > 55 and stats['avg_return'] > 0:
                st.success("✅ **Positive Edge**: High accuracy + positive returns")
            elif stats['accuracy'] > 55 and stats['avg_return'] <= 0:
                st.warning("⚠️ **High Accuracy, Low Returns**: Correct direction but small moves")
            elif stats['accuracy'] <= 55 and stats['avg_return'] > 0:
                st.info("ℹ️ **Lucky Streak**: Positive returns despite low accuracy")
            else:
                st.error("❌ **No Edge**: Low accuracy + negative returns")
        else:
            st.info("No data available yet")
    
    st.divider()
    
    # 最近信号
    st.subheader("🕐 Recent Signals")
    
    try:
        # 获取最近10条信号
        recent_results = signals_collection.get(
            limit=10,
            where={"theme": theme} if theme else None
        )
        
        if recent_results and recent_results['ids']:
            signal_data = []
            
            for i, signal_id in enumerate(recent_results['ids']):
                metadata = recent_results['metadatas'][i]
                
                signal_data.append({
                    "Time": metadata['timestamp'][:16],
                    "Asset": metadata['asset'],
                    "Direction": metadata['direction'],
                    "Confidence": f"{metadata['confidence']:.1f}",
                    "Theme": metadata['theme'],
                    f"Result ({timeframe})": metadata.get(f"correct_{timeframe}", "PENDING"),
                    f"Return ({timeframe})": f"{metadata.get(f'return_{timeframe}', 0.0):+.2f}%"
                })
            
            df = pd.DataFrame(signal_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals recorded yet. Run some analyses to start tracking!")
    
    except Exception as e:
        st.error(f"Error loading recent signals: {e}")
    
    st.divider()
    
    # 使用说明
    with st.expander("ℹ️ How to Use Signal Scoreboard"):
        st.markdown("""
        ### Signal Tracking System
        
        **Automatic Recording**:
        - Every time you run an analysis, predictions are automatically recorded
        - Initial price is captured at the time of prediction
        
        **Result Backfilling**:
        - Click "🔄 Update Results" to check and update signal outcomes
        - System checks if enough time has passed (1h/4h/1d/1w)
        - Fetches actual prices and calculates returns
        
        **Interpreting Results**:
        - **Accuracy**: % of predictions where direction was correct
        - **Avg Return**: Average % return per signal
        - **Statistical Significance**: Need 30+ samples for reliable conclusions
        
        **Performance Categories**:
        - ✅ **Positive Edge**: Accuracy > 55% AND Avg Return > 0%
        - ⚠️ **High Accuracy, Low Returns**: Correct often but small moves
        - ℹ️ **Lucky Streak**: Positive returns despite low accuracy (unsustainable)
        - ❌ **No Edge**: Low accuracy AND negative returns
        
        **Important Notes**:
        - Returns are theoretical (no transaction costs)
        - Past performance doesn't guarantee future results
        - Use this data to validate your strategy before risking real money
        """)
