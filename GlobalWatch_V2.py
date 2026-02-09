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
import os

# === 0. 基础设置 ===
try:
    from plyer import notification
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# === 0.1. 日志工具函数 ===
def log_error(message):
    """
    记录错误到文件和控制台
    Args:
        message: 错误消息
    """
    try:
        # 确保 outputs 目录存在
        os.makedirs("outputs", exist_ok=True)
        
        # 写入日志文件
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open("outputs/error.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        # 日志本身失败也不能崩溃
        print(f"Logging failed: {e}")

def safe_format_number(value, decimals=2, default="N/A"):
    """
    安全格式化数字，处理 None/NaN/inf
    Args:
        value: 数值
        decimals: 小数位数
        default: 默认值（当无法格式化时）
    Returns:
        格式化后的字符串
    """
    try:
        if value is None:
            return default
        if pd.isna(value) or not pd.api.types.is_number(value):
            return default
        if not (-1e10 < value < 1e10):  # 检查 inf
            return default
        return f"{value:,.{decimals}f}"
    except Exception as e:
        log_error(f"safe_format_number error: {str(e)}")
        return default

# 【关键修改】切换为推理模型 (请确保终端已运行 ollama pull deepseek-r1:8b)
LOCAL_MODEL = "deepseek-r1:8b" 
# Temperature range: 0.1 ~ 0.3 (configured in ollama.chat calls)
TEMPERATURE = 0.2  # Default temperature for model calls

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

# Early-Warning 监控列表配置
WATCHLIST = {
    "Gold": {
        "ticker": "GC=F",
        "type": "commodity",
        "correlations": {"DXY": -0.7, "^TNX": -0.5}
    },
    "Oil": {
        "ticker": "CL=F",
        "type": "commodity",
        "correlations": {"DXY": -0.4, "CAD=X": 0.6}
    },
    "CNY": {
        "ticker": "CNY=X",
        "type": "fx",
        "correlations": {"DXY": -0.6, "CL=F": 0.3}
    },
    "CAD": {
        "ticker": "CAD=X",
        "type": "fx",
        "correlations": {"CL=F": 0.6, "DXY": -0.4}
    }
}

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

def load_paper_runtime_settings():
    """Load optional runtime settings from paper_config.json."""
    defaults = {
        "enable_llm_topic_signals": True,
        "llm_topic_confidence_threshold": 0.6,
        "llm_topic_score_threshold": 0.5,
        "llm_topic_model": LOCAL_MODEL,
        "topic_memory_window": 50
    }
    config_path = os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")
    if not os.path.exists(config_path):
        return defaults
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        macro_cfg = cfg.get("macro_integration", {})
        defaults["enable_llm_topic_signals"] = bool(macro_cfg.get("enable_llm_topic_signals", defaults["enable_llm_topic_signals"]))
        defaults["llm_topic_confidence_threshold"] = float(macro_cfg.get("llm_topic_confidence_threshold", defaults["llm_topic_confidence_threshold"]))
        defaults["llm_topic_score_threshold"] = float(macro_cfg.get("llm_topic_score_threshold", defaults["llm_topic_score_threshold"]))
        defaults["llm_topic_model"] = str(macro_cfg.get("llm_topic_model", defaults["llm_topic_model"]))
        defaults["topic_memory_window"] = int(macro_cfg.get("topic_memory_window", defaults["topic_memory_window"]))
    except Exception as e:
        log_error(f"load_paper_runtime_settings error: {str(e)}")
    return defaults


RUNTIME_SETTINGS = load_paper_runtime_settings()
TOPIC_MEMORY_PATH = "outputs/topic_signal_memory.json"


def _normalize_theme_key(theme):
    return str(theme or "").strip().lower()


def _coerce_binary_outcome(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else -1
    if isinstance(value, (int, float)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "t", "yes", "y", "1", "correct"):
            return 1
        if text in ("false", "f", "no", "n", "0", "wrong", "incorrect"):
            return -1
        try:
            return _coerce_binary_outcome(float(text))
        except Exception:
            return None
    return None


def load_topic_signal_memory():
    """Load persistent topic signal feedback memory."""
    default_state = {
        "historical_signals": {},
        "processed_signal_ids": [],
        "updated_at": datetime.now().isoformat()
    }
    try:
        os.makedirs("outputs", exist_ok=True)
        if not os.path.exists(TOPIC_MEMORY_PATH):
            return default_state
        with open(TOPIC_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default_state
        hist = data.get("historical_signals", {})
        processed = data.get("processed_signal_ids", [])
        if not isinstance(hist, dict):
            hist = {}
        if not isinstance(processed, list):
            processed = []
        return {
            "historical_signals": hist,
            "processed_signal_ids": processed,
            "updated_at": str(data.get("updated_at", datetime.now().isoformat()))
        }
    except Exception as e:
        log_error(f"load_topic_signal_memory error: {str(e)}")
        return default_state


def save_topic_signal_memory(memory_state):
    """Persist topic signal feedback memory to JSON."""
    try:
        os.makedirs("outputs", exist_ok=True)
        payload = dict(memory_state or {})
        payload["updated_at"] = datetime.now().isoformat()
        with open(TOPIC_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_error(f"save_topic_signal_memory error: {str(e)}")


def _infer_topic_outcome_from_metadata(metadata):
    """Infer topic signal outcome as +1 / -1 / 0 when possible."""
    if not isinstance(metadata, dict):
        return None

    correct_1d = _coerce_binary_outcome(metadata.get("correct_1d"))
    if correct_1d is not None:
        return int(correct_1d)

    try:
        ret_1d = float(metadata.get("return_1d", 0.0))
        topic_score = float(metadata.get("topic_score", metadata.get("topic_score_raw", 0.0)))
    except Exception:
        return None

    if abs(topic_score) <= 1e-12 or abs(ret_1d) <= 1e-12:
        return 0
    return 1 if (ret_1d * topic_score) > 0 else -1


def refresh_topic_signal_memory(memory_state=None):
    """Backfill topic memory from verified topic_sentiment records."""
    state = memory_state if isinstance(memory_state, dict) else load_topic_signal_memory()
    historical = state.setdefault("historical_signals", {})
    processed_ids = set(state.get("processed_signal_ids", []))
    window = max(1, int(RUNTIME_SETTINGS.get("topic_memory_window", 50)))
    updates = 0

    try:
        results = signals_collection.get(
            where={"signal_type": "topic_sentiment"},
            include=["metadatas"]
        )
    except Exception as e:
        log_error(f"refresh_topic_signal_memory fetch error: {str(e)}")
        return state, 0

    ids = results.get("ids", []) if isinstance(results, dict) else []
    metadatas = results.get("metadatas", []) if isinstance(results, dict) else []
    for idx, signal_id in enumerate(ids):
        if signal_id in processed_ids:
            continue
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if not isinstance(metadata, dict):
            continue
        theme = _normalize_theme_key(metadata.get("topic_sector") or metadata.get("sector") or metadata.get("theme"))
        if not theme:
            continue

        outcome = _infer_topic_outcome_from_metadata(metadata)
        if outcome is None:
            continue

        history = historical.setdefault(theme, [])
        history.append(int(outcome))
        if len(history) > window:
            del history[:-window]
        processed_ids.add(signal_id)
        updates += 1

    state["processed_signal_ids"] = list(processed_ids)[-10000:]
    state["historical_signals"] = historical
    return state, updates


def evaluate_theme_signal_accuracy(theme: str, memory_state=None):
    """Return topic accuracy summary over the last N feedback outcomes."""
    state = memory_state if isinstance(memory_state, dict) else load_topic_signal_memory()
    historical = state.get("historical_signals", {})
    key = _normalize_theme_key(theme)
    history = historical.get(key, [])
    window = max(1, int(RUNTIME_SETTINGS.get("topic_memory_window", 50)))
    sample = history[-window:] if isinstance(history, list) else []
    informative = [x for x in sample if x in (-1, 1)]
    if not informative:
        accuracy = 0.5
    else:
        accuracy = float(sum(1 for x in informative if x == 1) / len(informative))
    return {
        "theme": key,
        "accuracy": float(accuracy),
        "samples": int(len(informative)),
        "window": int(window),
        "history": list(sample)
    }


def _accuracy_to_adaptive_weight(accuracy):
    """Map accuracy bands to adaptive tilt weights."""
    acc = float(accuracy)
    if acc < 0.40:
        return 0.75
    if acc > 0.60:
        return 1.25
    return 1.00


def query_ollama(model, prompt, num_ctx=8192, temperature=None):
    """Thin wrapper for ollama.chat to keep calls consistent."""
    if temperature is None:
        temperature = TEMPERATURE
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": int(num_ctx), "temperature": float(temperature)}
    )
    return response.get("message", {}).get("content", "")

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
                except Exception as e:
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
        except Exception as e:
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
    except Exception as e:
        log_error(f"recall_history error: {str(e)}")
        return "Memory Empty."

def send_notification(title, msg):
    if TOAST_AVAILABLE:
        try:
            notification.notify(title=title, message=msg, app_name='GlobalWatch', timeout=10)
        except Exception as e:
            log_error(f"send_notification error: {str(e)}")
            pass

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
    except Exception as e:
        log_error(f"get_current_price error for {ticker}: {str(e)}")
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


def extract_llm_topic_sentiment(news, macro_data, lang_mode):
    """Extract structured sector/topic sentiment from headlines using LLM."""
    if not news:
        return None
    if not RUNTIME_SETTINGS.get("enable_llm_topic_signals", True):
        return None

    model_name = str(RUNTIME_SETTINGS.get("llm_topic_model", LOCAL_MODEL))
    lang_instruction = "OUTPUT LANGUAGE: CHINESE (Simplified)" if lang_mode == "中文" else "OUTPUT LANGUAGE: ENGLISH"
    headlines = "\n".join([f"- [{item.get('source', 'Unknown')}] {item.get('title', '')}" for item in news[:20]])

    prompt = f"""
You are a macro-news signal parser. {lang_instruction}
Read the headlines and infer sector/theme momentum.

Headlines:
{headlines}

Macro context:
{json.dumps(macro_data)}

Return ONLY JSON in this exact schema:
{{
  "timestamp": "ISO8601 string",
  "source": "llm+news",
  "confidence": 0.0,
  "summary": "1-2 sentence summary",
  "sector_mentions": [{{"sector": "semiconductors", "count": 2}}],
  "entities": [{{"entity": "NVIDIA", "sector": "semiconductors", "polarity": 0.8}}],
  "topic_recurrence": [{{"topic": "AI chip demand", "frequency": 3}}],
  "signals": {{"semiconductors": 0.7, "utilities": -0.5}}
}}

Rules:
- signal values must be in [-1.0, 1.0]
- confidence must be in [0.0, 1.0]
- include only sectors/themes with clear directional evidence
"""
    try:
        raw_content = query_ollama(model_name, prompt, num_ctx=8192)
        thought, json_text = parse_deepseek_output(raw_content)
        parsed = robust_json_parse(json_text, model_name, max_retries=1)
        if parsed.get("_parse_error"):
            return None

        raw_signals = parsed.get("signals", {})
        if not isinstance(raw_signals, dict):
            return None

        normalized_signals = {}
        for sector, value in raw_signals.items():
            if sector is None:
                continue
            sector_key = str(sector).strip().lower()
            if not sector_key:
                continue
            try:
                score = float(value)
            except Exception:
                continue
            normalized_signals[sector_key] = max(-1.0, min(1.0, score))

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        payload = {
            "timestamp": datetime.now().isoformat(),
            "source": "llm+news",
            "confidence": confidence,
            "summary": str(parsed.get("summary", "")),
            "sector_mentions": parsed.get("sector_mentions", []),
            "entities": parsed.get("entities", []),
            "topic_recurrence": parsed.get("topic_recurrence", []),
            "signals": normalized_signals,
            "thought_process": thought
        }
        return payload
    except Exception as e:
        log_error(f"extract_llm_topic_sentiment error: {str(e)}")
        return None


def record_llm_topic_signals(payload, news_sources):
    """Store structured LLM topic signals into trading_signals collection."""
    if not payload:
        return 0
    signals = payload.get("signals", {})
    if not isinstance(signals, dict) or not signals:
        return 0

    confidence_threshold = float(RUNTIME_SETTINGS.get("llm_topic_confidence_threshold", 0.6))
    score_threshold = float(RUNTIME_SETTINGS.get("llm_topic_score_threshold", 0.5))
    payload_conf = float(payload.get("confidence", 0.0))
    if payload_conf < confidence_threshold:
        return 0

    memory_state = load_topic_signal_memory()
    memory_state, memory_updates = refresh_topic_signal_memory(memory_state)
    if memory_updates > 0:
        save_topic_signal_memory(memory_state)

    count = 0
    source_str = ",".join(list(set([s for s in news_sources if s]))[:3])
    topic_recurrence = payload.get("topic_recurrence", [])
    recurrence_map = {}
    if isinstance(topic_recurrence, list):
        for item in topic_recurrence:
            if isinstance(item, dict):
                topic_name = str(item.get("topic", "")).strip().lower()
                if not topic_name:
                    continue
                try:
                    recurrence_map[topic_name] = float(item.get("frequency", 1.0))
                except Exception:
                    recurrence_map[topic_name] = 1.0

    for sector, raw_score in signals.items():
        try:
            score = float(raw_score)
        except Exception:
            continue
        if abs(score) < score_threshold:
            continue

        direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
        sector_key = str(sector).strip().lower()
        accuracy_info = evaluate_theme_signal_accuracy(sector_key, memory_state=memory_state)
        adaptive_weight = _accuracy_to_adaptive_weight(accuracy_info.get("accuracy", 0.5))
        signal_id = str(uuid.uuid4())
        topic_recur = recurrence_map.get(sector_key, 1.0)
        metadata = {
            "signal_id": signal_id,
            "timestamp": payload.get("timestamp", datetime.now().isoformat()),
            "asset": sector_key,
            "ticker": "UNKNOWN",
            "direction": direction,
            "confidence": float(payload_conf * 100.0),
            "theme": f"sector_{sector_key}",
            "initial_price": 0.0,
            "sources": source_str,
            "status": "PENDING",
            "signal_type": "topic_sentiment",
            "topic_sector": sector_key,
            "topic_score": float(score),
            "topic_confidence": float(payload_conf),
            "topic_recurrence": float(topic_recur),
            "topic_memory_window": int(accuracy_info.get("window", int(RUNTIME_SETTINGS.get("topic_memory_window", 50)))),
            "topic_accuracy": float(accuracy_info.get("accuracy", 0.5)),
            "topic_accuracy_samples": int(accuracy_info.get("samples", 0)),
            "topic_adaptive_weight": float(adaptive_weight),
            "summary": str(payload.get("summary", "")),
            "source_tag": "llm+news",
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

        try:
            signals_collection.add(
                documents=[json.dumps(payload)],
                metadatas=[metadata],
                ids=[signal_id]
            )
            if score > 0:
                print(
                    f"[GLOBALWATCH] Overweight {sector_key} due to LLM {score:+.2f} "
                    f"(acc={accuracy_info.get('accuracy', 0.5):.2f}, w={adaptive_weight:.2f})"
                )
            else:
                print(
                    f"[GLOBALWATCH] Underweight {sector_key} due to LLM {score:+.2f} "
                    f"(acc={accuracy_info.get('accuracy', 0.5):.2f}, w={adaptive_weight:.2f})"
                )
            count += 1
        except Exception as e:
            log_error(f"record_llm_topic_signals add error: {str(e)}")
            continue

    if count > 0:
        os.makedirs("outputs", exist_ok=True)
        try:
            with open("outputs/llm_topic_signals.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            log_error(f"record_llm_topic_signals file write error: {str(e)}")
    return count

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
    except Exception as e:
        log_error(f"get_historical_price error for {ticker}: {str(e)}")
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

# ================= 2.6. Early-Warning 风险评分系统 =================

def calculate_rsi(ticker, period=14):
    """计算 RSI 指标"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist.empty or len(hist) < period + 1:
            return None
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception as e:
        log_error(f"calculate_rsi error for {ticker}: {str(e)}")
        return None

def calculate_ma(ticker, period=20):
    """计算移动平均线"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist.empty or len(hist) < period:
            return None
        return float(hist['Close'].rolling(window=period).mean().iloc[-1])
    except Exception as e:
        log_error(f"calculate_ma error for {ticker}: {str(e)}")
        return None

def calculate_atr(ticker, period=14):
    """计算 ATR (Average True Range)"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist.empty or len(hist) < period + 1:
            return None
        
        high_low = hist['High'] - hist['Low']
        high_close = abs(hist['High'] - hist['Close'].shift())
        low_close = abs(hist['Low'] - hist['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1])
    except Exception as e:
        log_error(f"calculate_atr error for {ticker}: {str(e)}")
        return None

def get_price_change(ticker, period="1w"):
    """获取价格变化百分比"""
    try:
        t = yf.Ticker(ticker)
        if period == "1w":
            hist = t.history(period="1mo")
            if len(hist) < 5:
                return 0.0
            old_price = hist['Close'].iloc[-5]
        elif period == "1d":
            hist = t.history(period="5d")
            if len(hist) < 2:
                return 0.0
            old_price = hist['Close'].iloc[-2]
        else:
            return 0.0
        
        current_price = hist['Close'].iloc[-1]
        return float((current_price - old_price) / old_price * 100)
    except Exception as e:
        log_error(f"get_price_change error for {ticker}: {str(e)}")
        return 0.0

def calculate_gap(ticker):
    """计算跳空缺口"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if len(hist) < 2:
            return 0.0
        
        prev_close = hist['Close'].iloc[-2]
        current_open = hist['Open'].iloc[-1]
        gap = (current_open - prev_close) / prev_close * 100
        return float(gap)
    except Exception as e:
        log_error(f"calculate_gap error for {ticker}: {str(e)}")
        return 0.0

def get_current_volume(ticker):
    """获取当前成交量"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            return 0
        return int(hist['Volume'].iloc[-1])
    except Exception as e:
        log_error(f"get_current_volume error for {ticker}: {str(e)}")
        return 0

def get_avg_volume(ticker, period=20):
    """获取平均成交量"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        if hist.empty or len(hist) < period:
            return 1  # 避免除零
        return int(hist['Volume'].rolling(window=period).mean().iloc[-1])
    except Exception as e:
        log_error(f"get_avg_volume error for {ticker}: {str(e)}")
        return 1

def count_keyword_mentions(keywords, news_list):
    """统计关键词在新闻中的出现次数"""
    count = 0
    for news_item in news_list:
        title = news_item.get('title', '').lower()
        for keyword in keywords:
            if keyword.lower() in title:
                count += 1
                break  # 每条新闻只计数一次
    return count

def calculate_macro_chain_score(asset_name, asset_config, recent_news):
    """
    计算宏观链条分 (0-25)
    基于美元指数、利率、相关性
    """
    score = 0
    evidence = []
    
    try:
        # 1. 美元指数影响 (0-10分)
        dxy_change = get_price_change("DX-Y.NYB", period="1w")  # 美元指数
        dxy_correlation = asset_config['correlations'].get('DXY', 0)
        
        if abs(dxy_correlation) > 0.5:  # 强相关
            if (dxy_correlation < 0 and dxy_change > 2) or (dxy_correlation > 0 and dxy_change < -2):
                score += 8
                evidence.append({
                    "type": "price",
                    "indicator": "DXY",
                    "value": f"{dxy_change:+.1f}%",
                    "interpretation": f"USD {'strength' if dxy_change > 0 else 'weakness'} {'negative' if dxy_correlation < 0 else 'positive'} for {asset_name}"
                })
            elif abs(dxy_change) > 1:
                score += 4
                evidence.append({
                    "type": "price",
                    "indicator": "DXY",
                    "value": f"{dxy_change:+.1f}%",
                    "interpretation": f"Moderate USD movement, correlation: {dxy_correlation:.2f}"
                })
        
        # 2. 利率影响 (0-10分)
        tnx_change = get_price_change("^TNX", period="1w")  # 10年期国债
        tnx_correlation = asset_config['correlations'].get('^TNX', 0)
        
        if abs(tnx_correlation) > 0.3:
            if (tnx_correlation < 0 and tnx_change > 0.2) or (tnx_correlation > 0 and tnx_change < -0.2):
                score += 7
                evidence.append({
                    "type": "price",
                    "indicator": "10Y Yield",
                    "value": f"{tnx_change:+.2f}%",
                    "interpretation": f"Rate {'rise' if tnx_change > 0 else 'fall'} {'negative' if tnx_correlation < 0 else 'positive'} for {asset_name}"
                })
        
        # 3. 新闻中的宏观提及 (0-5分)
        macro_keywords = ["Fed", "interest rate", "dollar", "USD", "inflation", "央行", "利率"]
        news_mentions = count_keyword_mentions(macro_keywords, recent_news)
        
        if news_mentions >= 3:
            score += 5
            evidence.append({
                "type": "news",
                "count": news_mentions,
                "keywords": "Fed/rates/USD",
                "interpretation": "High macro news flow increases uncertainty"
            })
        elif news_mentions > 0:
            score += 2
            evidence.append({
                "type": "news",
                "count": news_mentions,
                "keywords": "Fed/rates/USD",
                "interpretation": "Moderate macro news mentions"
            })
        
    except Exception as e:
        evidence.append({
            "type": "error",
            "message": f"Error calculating macro score: {str(e)}"
        })
    
    return min(score, 25), evidence

def calculate_crowding_score(ticker):
    """
    计算拥挤度分 (0-25)
    基于 RSI、价格偏离均线、情绪
    """
    score = 0
    evidence = []
    
    try:
        # 1. RSI 超买/超卖 (0-12分)
        rsi = calculate_rsi(ticker, period=14)
        
        if rsi is not None:
            if rsi > 70:  # 超买
                score += min((rsi - 70) / 3, 12)
                evidence.append({
                    "type": "price",
                    "indicator": "RSI(14)",
                    "value": f"{rsi:.1f}",
                    "interpretation": "Overbought - potential reversal risk"
                })
            elif rsi < 30:  # 超卖
                score += min((30 - rsi) / 3, 12)
                evidence.append({
                    "type": "price",
                    "indicator": "RSI(14)",
                    "value": f"{rsi:.1f}",
                    "interpretation": "Oversold - potential bounce risk"
                })
            elif rsi > 60 or rsi < 40:
                score += 3
                evidence.append({
                    "type": "price",
                    "indicator": "RSI(14)",
                    "value": f"{rsi:.1f}",
                    "interpretation": "Moderate momentum"
                })
        
        # 2. 价格偏离均线 (0-10分)
        price = get_current_price(ticker)
        ma20 = calculate_ma(ticker, period=20)
        
        if price and ma20:
            deviation = (price - ma20) / ma20 * 100
            
            if abs(deviation) > 5:
                score += min(abs(deviation) - 5, 10)
                evidence.append({
                    "type": "price",
                    "indicator": "Price vs MA20",
                    "value": f"{deviation:+.1f}%",
                    "interpretation": f"{'Above' if deviation > 0 else 'Below'} MA20 - stretched"
                })
            elif abs(deviation) > 2:
                score += 2
                evidence.append({
                    "type": "price",
                    "indicator": "Price vs MA20",
                    "value": f"{deviation:+.1f}%",
                    "interpretation": "Moderate deviation from MA20"
                })
        
        # 3. 成交量异常 (0-3分)
        current_volume = get_current_volume(ticker)
        avg_volume = get_avg_volume(ticker, period=20)
        
        if current_volume > 0 and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 2:
                score += 3
                evidence.append({
                    "type": "price",
                    "indicator": "Volume",
                    "value": f"{volume_ratio:.1f}x avg",
                    "interpretation": "Volume spike - increased crowding"
                })
    
    except Exception as e:
        evidence.append({
            "type": "error",
            "message": f"Error calculating crowding score: {str(e)}"
        })
    
    return min(score, 25), evidence

def calculate_microstructure_score(ticker):
    """
    计算微结构分 (0-25)
    基于波动率、跳空、成交量
    """
    score = 0
    evidence = []
    
    try:
        # 1. 波动率骤增 (0-12分)
        current_atr = calculate_atr(ticker, period=14)
        avg_atr = calculate_atr(ticker, period=50)
        
        if current_atr and avg_atr and avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            
            if atr_ratio > 1.5:  # 波动率上升50%+
                score += min((atr_ratio - 1) * 6, 12)
                evidence.append({
                    "type": "price",
                    "indicator": "ATR Ratio",
                    "value": f"{atr_ratio:.2f}x",
                    "interpretation": "Volatility surge - increased risk"
                })
            elif atr_ratio > 1.2:
                score += 3
                evidence.append({
                    "type": "price",
                    "indicator": "ATR Ratio",
                    "value": f"{atr_ratio:.2f}x",
                    "interpretation": "Elevated volatility"
                })
        
        # 2. 跳空缺口 (0-8分)
        gap = calculate_gap(ticker)
        
        if abs(gap) > 2:  # 跳空超过2%
            score += min(abs(gap), 8)
            evidence.append({
                "type": "price",
                "indicator": "Gap",
                "value": f"{gap:+.1f}%",
                "interpretation": f"{'Upward' if gap > 0 else 'Downward'} gap - momentum shift"
            })
        elif abs(gap) > 1:
            score += 2
            evidence.append({
                "type": "price",
                "indicator": "Gap",
                "value": f"{gap:+.1f}%",
                "interpretation": "Small gap detected"
            })
        
        # 3. 成交量异常 (0-5分)
        current_volume = get_current_volume(ticker)
        avg_volume = get_avg_volume(ticker, period=20)
        
        if current_volume > 0 and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 2:  # 成交量翻倍
                score += min((volume_ratio - 1) * 2.5, 5)
                evidence.append({
                    "type": "price",
                    "indicator": "Volume Ratio",
                    "value": f"{volume_ratio:.1f}x",
                    "interpretation": "Volume spike - increased activity"
                })
    
    except Exception as e:
        evidence.append({
            "type": "error",
            "message": f"Error calculating microstructure score: {str(e)}"
        })
    
    return min(score, 25), evidence

def calculate_event_risk_score(recent_news):
    """
    计算事件风险分 (0-25)
    基于新闻关键词匹配
    """
    score = 0
    evidence = []
    
    try:
        # 定义事件风险关键词
        event_keywords = {
            "central_bank": ["Fed", "ECB", "央行", "interest rate", "monetary policy", "Powell", "Yellen"],
            "policy": ["tariff", "sanction", "regulation", "policy", "law", "trade war", "关税"],
            "geopolitical": ["war", "conflict", "election", "crisis", "tension", "战争", "冲突"]
        }
        
        # 1. 央行事件 (0-10分)
        cb_mentions = count_keyword_mentions(event_keywords["central_bank"], recent_news)
        if cb_mentions >= 3:
            score += 10
            evidence.append({
                "type": "news",
                "category": "Central Bank",
                "count": cb_mentions,
                "interpretation": "High central bank event risk"
            })
        elif cb_mentions > 0:
            score += cb_mentions * 2
            evidence.append({
                "type": "news",
                "category": "Central Bank",
                "count": cb_mentions,
                "interpretation": "Central bank mentions detected"
            })
        
        # 2. 政策风险 (0-8分)
        policy_mentions = count_keyword_mentions(event_keywords["policy"], recent_news)
        if policy_mentions >= 3:
            score += 8
            evidence.append({
                "type": "news",
                "category": "Policy",
                "count": policy_mentions,
                "interpretation": "High policy uncertainty"
            })
        elif policy_mentions > 0:
            score += policy_mentions * 2
            evidence.append({
                "type": "news",
                "category": "Policy",
                "count": policy_mentions,
                "interpretation": "Policy risk mentions"
            })
        
        # 3. 地缘政治 (0-7分)
        geo_mentions = count_keyword_mentions(event_keywords["geopolitical"], recent_news)
        if geo_mentions >= 3:
            score += 7
            evidence.append({
                "type": "news",
                "category": "Geopolitical",
                "count": geo_mentions,
                "interpretation": "High geopolitical risk"
            })
        elif geo_mentions > 0:
            score += geo_mentions * 2
            evidence.append({
                "type": "news",
                "category": "Geopolitical",
                "count": geo_mentions,
                "interpretation": "Geopolitical mentions detected"
            })
    
    except Exception as e:
        evidence.append({
            "type": "error",
            "message": f"Error calculating event risk score: {str(e)}"
        })
    
    return min(score, 25), evidence

def calculate_early_warning_score(asset_name, recent_news):
    """
    计算综合 Early-Warning 风险分数
    Returns:
        dict: {
            "asset": str,
            "timestamp": str,
            "total_risk_score": int (0-100),
            "risk_level": str (LOW/MEDIUM/HIGH/CRITICAL),
            "sub_scores": {
                "macro_chain": {"score": int, "evidence": list},
                "crowding": {"score": int, "evidence": list},
                "microstructure": {"score": int, "evidence": list},
                "event_risk": {"score": int, "evidence": list}
            },
            "alert_triggers": list,
            "recommendation": str
        }
    """
    if asset_name not in WATCHLIST:
        return {
            "asset": asset_name,
            "error": "Asset not in watchlist"
        }
    
    asset_config = WATCHLIST[asset_name]
    ticker = asset_config['ticker']
    
    # 计算四个子分数
    macro_score, macro_evidence = calculate_macro_chain_score(asset_name, asset_config, recent_news)
    crowding_score, crowding_evidence = calculate_crowding_score(ticker)
    micro_score, micro_evidence = calculate_microstructure_score(ticker)
    event_score, event_evidence = calculate_event_risk_score(recent_news)
    
    # 综合分数
    total_score = macro_score + crowding_score + micro_score + event_score
    
    # 风险等级
    if total_score >= 76:
        risk_level = "CRITICAL"
    elif total_score >= 51:
        risk_level = "HIGH"
    elif total_score >= 26:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # 告警触发器
    alert_triggers = []
    if total_score > 60:
        alert_triggers.append(f"Total risk score > 60 ({risk_level})")
    if macro_score > 15:
        alert_triggers.append(f"Macro chain score > 15 ({macro_score})")
    if crowding_score > 18:
        alert_triggers.append(f"Crowding score > 18 ({crowding_score})")
    if micro_score > 15:
        alert_triggers.append(f"Microstructure score > 15 ({micro_score})")
    if event_score > 15:
        alert_triggers.append(f"Event risk score > 15 ({event_score})")
    
    # 建议
    if risk_level == "CRITICAL":
        recommendation = "🔴 CRITICAL: Multiple risk factors at extreme levels. Consider reducing exposure significantly or hedging."
    elif risk_level == "HIGH":
        recommendation = "🟠 CAUTION: Elevated risk across multiple dimensions. Monitor closely and consider risk management."
    elif risk_level == "MEDIUM":
        recommendation = "🟡 WATCH: Some risk factors elevated. Stay alert to developments."
    else:
        recommendation = "🟢 NORMAL: Risk levels within normal range. Continue monitoring."
    
    return {
        "asset": asset_name,
        "timestamp": datetime.now().isoformat(),
        "total_risk_score": total_score,
        "risk_level": risk_level,
        "sub_scores": {
            "macro_chain": {"score": macro_score, "evidence": macro_evidence},
            "crowding": {"score": crowding_score, "evidence": crowding_evidence},
            "microstructure": {"score": micro_score, "evidence": micro_evidence},
            "event_risk": {"score": event_score, "evidence": event_evidence}
        },
        "alert_triggers": alert_triggers,
        "recommendation": recommendation
    }

def get_signal_statistics(theme=None, asset=None, timeframe="1d"):
    """
    获取信号统计（升级版 - 支持交易级分类）
    Args:
        theme: 主题过滤 (FX/MACRO/STOCK/None)
        asset: 资产过滤 (None 表示全部)
        timeframe: 时间框架 (1h/4h/1d/1w)
    Returns:
        dict: 统计数据（包含交易级指标）
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
                "verified_signals": 0,
                "accuracy": 0.0,
                "avg_return": 0.0,
                "max_return": 0.0,
                "min_return": 0.0,
                "sample_size": 0,
                "statistical_significance": False,
                "cumulative_return": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "returns_list": [],
                "timestamps": []
            }
        
        # 提取对应时间框架的数据
        correct_field = f"correct_{timeframe}"
        return_field = f"return_{timeframe}"
        
        correct_count = 0
        wrong_count = 0
        returns = []
        timestamps = []
        
        for metadata in results['metadatas']:
            correct_status = metadata.get(correct_field, "")
            return_value = metadata.get(return_field, 0.0)
            
            if correct_status == "CORRECT":
                correct_count += 1
                returns.append(return_value)
                timestamps.append(metadata.get('timestamp', ''))
            elif correct_status == "WRONG":
                wrong_count += 1
                returns.append(return_value)
                timestamps.append(metadata.get('timestamp', ''))
        
        total_verified = correct_count + wrong_count
        
        if total_verified == 0:
            accuracy = 0.0
            avg_return = 0.0
            cumulative_return = 0.0
            max_drawdown = 0.0
            volatility = 0.0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        else:
            accuracy = correct_count / total_verified * 100
            avg_return = sum(returns) / len(returns) if returns else 0.0
            
            # 累计收益
            cumulative_return = sum(returns)
            
            # 最大回撤计算
            cumulative_curve = []
            running_sum = 0
            for r in returns:
                running_sum += r
                cumulative_curve.append(running_sum)
            
            max_drawdown = 0.0
            if cumulative_curve:
                peak = cumulative_curve[0]
                for value in cumulative_curve:
                    if value > peak:
                        peak = value
                    drawdown = peak - value
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # 波动率（收益标准差）
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
            
            # 胜率和盈亏比
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            win_rate = len(wins) / len(returns) * 100 if returns else 0.0
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            
            # Profit Factor
            total_wins = sum(wins) if wins else 0.0
            total_losses = abs(sum(losses)) if losses else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            "total_signals": len(results['ids']),
            "verified_signals": total_verified,
            "accuracy": accuracy,
            "avg_return": avg_return,
            "max_return": max(returns) if returns else 0.0,
            "min_return": min(returns) if returns else 0.0,
            "sample_size": total_verified,
            "statistical_significance": total_verified >= 30,
            "cumulative_return": cumulative_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "returns_list": returns,  # 用于多时间窗口验证
            "timestamps": timestamps
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {
            "total_signals": 0,
            "accuracy": 0.0,
            "avg_return": 0.0,
            "sample_size": 0,
            "statistical_significance": False,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0
        }

def classify_trading_performance(stats_dict, theme=None, asset=None, 
                                  transaction_cost=0.1, max_dd_threshold=15.0):
    """
    交易级性能分类体系（Trading-Grade Performance Classification）
    
    这是决定是否允许 real-money execution 的唯一依据
    
    Args:
        stats_dict: 统计数据字典（来自 get_signal_statistics）
        theme: 主题（用于多时间窗口验证）
        asset: 资产（用于多时间窗口验证）
        transaction_cost: 估算交易成本（百分比，默认 0.1%）
        max_dd_threshold: 最大回撤阈值（百分比，默认 15%）
    
    Returns:
        dict: {
            "classification_v2": str,  # 新交易级分类
            "classification_v1": str,  # 原分类（仅供参考）
            "decision_allowed": bool,  # 是否允许实盘交易
            "reason_summary": str,     # 人类可读的原因
            "risk_warnings": list,     # 风险警告列表
            "net_expected_value": float,  # 净期望值
            "multi_timeframe_validated": bool  # 多时间窗口验证
        }
    """
    
    # 提取关键指标
    trades_count = stats_dict.get('sample_size', 0)
    accuracy = stats_dict.get('accuracy', 0.0)
    avg_return = stats_dict.get('avg_return', 0.0)
    cumulative_return = stats_dict.get('cumulative_return', 0.0)
    max_drawdown = stats_dict.get('max_drawdown', 0.0)
    volatility = stats_dict.get('volatility', 0.0)
    win_rate = stats_dict.get('win_rate', 0.0)
    profit_factor = stats_dict.get('profit_factor', 0.0)
    
    # 计算净期望值
    net_expected_value = avg_return - transaction_cost
    
    # 初始化返回值
    classification_v2 = ""
    classification_v1 = ""
    decision_allowed = False
    reason_summary = ""
    risk_warnings = []
    multi_timeframe_validated = False
    
    # ========== 第一步：V1 分类（原分类，仅供参考）==========
    if trades_count == 0:
        classification_v1 = "No Data"
    elif accuracy > 55 and avg_return > 0:
        classification_v1 = "Positive Edge (V1)"
    elif accuracy > 55 and avg_return <= 0:
        classification_v1 = "High Accuracy, Low Returns (V1)"
    elif accuracy <= 55 and avg_return > 0:
        classification_v1 = "Lucky Streak (V1)"
    else:
        classification_v1 = "No Edge (V1)"
    
    # ========== 第二步：最低可评估门槛 ==========
    if trades_count < 30:
        classification_v2 = "🟤 Insufficient Data"
        decision_allowed = False
        reason_summary = f"样本数不足（{trades_count}/30）。需要至少 30 个已验证信号才能进行可靠评估。"
        risk_warnings.append("⚠️ 样本量过小，任何统计结论都不可靠")
        risk_warnings.append("⚠️ 禁止用于实盘交易")
        
        return {
            "classification_v2": classification_v2,
            "classification_v1": classification_v1,
            "decision_allowed": decision_allowed,
            "reason_summary": reason_summary,
            "risk_warnings": risk_warnings,
            "net_expected_value": net_expected_value,
            "multi_timeframe_validated": False
        }
    
    # ========== 第三步：多时间窗口验证 ==========
    # 检查至少 2 个时间窗口的 net_expected_value 是否为正
    if theme or asset:
        timeframes_to_check = ["1h", "4h", "1d", "1w"]
        positive_timeframes = []
        
        for tf in timeframes_to_check:
            tf_stats = get_signal_statistics(theme=theme, asset=asset, timeframe=tf)
            tf_net_ev = tf_stats.get('avg_return', 0.0) - transaction_cost
            tf_sample = tf_stats.get('sample_size', 0)
            
            if tf_sample >= 10 and tf_net_ev > 0:  # 至少 10 个样本且为正
                positive_timeframes.append(tf)
        
        multi_timeframe_validated = len(positive_timeframes) >= 2
    else:
        # 无法验证多时间窗口（未指定 theme/asset）
        multi_timeframe_validated = False
    
    # ========== 第四步：交易级分类 ==========
    
    # 🟢 Tradable Edge（允许实盘）
    if (trades_count >= 50 and 
        net_expected_value > 0 and 
        max_drawdown <= max_dd_threshold and
        multi_timeframe_validated):
        
        classification_v2 = "🟢 Tradable Edge"
        decision_allowed = True
        reason_summary = (
            f"✅ 满足所有交易级标准：\n"
            f"• 样本数充足（{trades_count} ≥ 50）\n"
            f"• 净期望值为正（{net_expected_value:.2f}% > 0）\n"
            f"• 回撤可控（{max_drawdown:.2f}% ≤ {max_dd_threshold}%）\n"
            f"• 多时间窗口验证通过\n"
            f"→ 允许进入实盘交易"
        )
        
        # 即使允许交易，也要给出风险提示
        if profit_factor < 1.5:
            risk_warnings.append(f"⚠️ Profit Factor 较低（{profit_factor:.2f}），建议谨慎控制仓位")
        if volatility > 2.0:
            risk_warnings.append(f"⚠️ 收益波动较大（{volatility:.2f}%），注意风险管理")
    
    # 🟡 Directional Signal（方向参考信号）
    elif (accuracy >= 58 and abs(net_expected_value) < 0.2):
        classification_v2 = "🟡 Directional Signal"
        decision_allowed = False
        reason_summary = (
            f"方向准确率较高（{accuracy:.1f}%），但净期望值接近零（{net_expected_value:.2f}%）。\n"
            f"交易成本侵蚀了大部分收益。\n"
            f"→ 仅允许用于：仓位调整、风险确认、多信号共识过滤\n"
            f"→ 禁止单独触发交易"
        )
        risk_warnings.append("⚠️ 不允许单独用于实盘交易")
        risk_warnings.append("✓ 可作为辅助信号使用")
    
    elif (avg_return > 0 and net_expected_value <= 0):
        classification_v2 = "🟡 Directional Signal"
        decision_allowed = False
        reason_summary = (
            f"平均收益为正（{avg_return:.2f}%），但被交易成本明显侵蚀。\n"
            f"净期望值：{net_expected_value:.2f}%\n"
            f"→ 仅允许用于辅助决策，禁止单独交易"
        )
        risk_warnings.append("⚠️ 交易成本过高，侵蚀收益")
        risk_warnings.append("⚠️ 不允许单独用于实盘交易")
    
    # 🟠 Unstable / Regime-Dependent（不稳定或依赖行情）
    elif (net_expected_value > 0 and 
          (trades_count < 50 or max_drawdown > max_dd_threshold or not multi_timeframe_validated)):
        
        classification_v2 = "🟠 Unstable / Regime-Dependent"
        decision_allowed = False
        
        reasons = []
        if trades_count < 50:
            reasons.append(f"样本量接近下限（{trades_count}/50）")
        if max_drawdown > max_dd_threshold:
            reasons.append(f"回撤过大（{max_drawdown:.2f}% > {max_dd_threshold}%）")
        if not multi_timeframe_validated:
            reasons.append("未通过多时间窗口验证")
        
        reason_summary = (
            f"净期望值为正（{net_expected_value:.2f}%），但存在以下问题：\n" +
            "\n".join(f"• {r}" for r in reasons) +
            f"\n→ 标记为「观察中」\n"
            f"→ 不允许自动交易\n"
            f"→ 必须等待更多数据或行情切换验证"
        )
        risk_warnings.append("⚠️ 系统不稳定，禁止实盘交易")
        risk_warnings.append("✓ 继续观察，积累更多数据")
    
    # 处理 Lucky Streak（重点修正）
    elif (accuracy <= 55 and avg_return > 0):
        # Lucky Streak 不再作为独立正向分类
        # 检查是否有非对称收益结构证据
        avg_win = stats_dict.get('avg_win', 0.0)
        avg_loss = stats_dict.get('avg_loss', 0.0)
        
        # 非对称收益：平均盈利 > 2 * 平均亏损（绝对值）
        has_fat_tail = avg_win > 2 * abs(avg_loss) if avg_loss != 0 else False
        
        if has_fat_tail and profit_factor > 2.0:
            classification_v2 = "🟠 Unstable / Regime-Dependent"
            reason_summary = (
                f"准确率较低（{accuracy:.1f}%），但存在非对称收益结构：\n"
                f"• 平均盈利：{avg_win:.2f}%\n"
                f"• 平均亏损：{avg_loss:.2f}%\n"
                f"• Profit Factor：{profit_factor:.2f}\n"
                f"→ 可能是 fat-tail payoff 策略\n"
                f"→ 需要更多数据验证，暂不允许交易"
            )
            risk_warnings.append("⚠️ 低准确率 + 高盈亏比，需验证是否可持续")
        else:
            classification_v2 = "🔴 No Edge"
            reason_summary = (
                f"准确率低（{accuracy:.1f}%），收益为正但无非对称结构证据。\n"
                f"大概率是运气（Lucky Streak）。\n"
                f"→ 永久禁止用于交易"
            )
            risk_warnings.append("🚫 疑似运气成分，禁止交易")
        
        decision_allowed = False
    
    # 🔴 No Edge（无优势）
    else:
        classification_v2 = "🔴 No Edge"
        decision_allowed = False
        
        reasons = []
        if net_expected_value <= 0:
            reasons.append(f"净期望值为负或零（{net_expected_value:.2f}%）")
        if max_drawdown > max_dd_threshold * 1.5:  # 严重超标
            reasons.append(f"回撤严重超标（{max_drawdown:.2f}%）")
        if accuracy < 45 and avg_return <= 0:
            reasons.append(f"准确率和收益均无统计意义")
        
        reason_summary = (
            "系统无交易优势：\n" +
            "\n".join(f"• {r}" for r in reasons) +
            f"\n→ 永久禁止用于交易（除非策略逻辑发生实质性变化）"
        )
        risk_warnings.append("🚫 无交易价值，禁止使用")
    
    return {
        "classification_v2": classification_v2,
        "classification_v1": classification_v1,
        "decision_allowed": decision_allowed,
        "reason_summary": reason_summary,
        "risk_warnings": risk_warnings,
        "net_expected_value": net_expected_value,
        "multi_timeframe_validated": multi_timeframe_validated
    }

def get_full_market_context():
    data = {}
    for name, ticker in MACRO_ANCHORS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty: data[name] = round(hist['Close'].iloc[-1], 2)
        except Exception as e:
            log_error(f"get_full_market_context error for {name}/{ticker}: {str(e)}")
            data[name] = "N/A"
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
                    except Exception as e:
                        pass
                elif hasattr(e, 'updated_parsed') and e.updated_parsed:
                    try:
                        published = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.updated_parsed)
                    except Exception as e:
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
            log_error(f"get_rss_news error for {src}: {str(e)}")
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
    except Exception as e:
        log_error(f"plot_candle_chart error for {ticker}: {str(e)}")
        st.caption("No Chart Data")

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
        except Exception as e:
            log_error(f"get_cross_rate.get_val error for {name}: {str(e)}")
            return None
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

        # Optional LLM topic sentiment extraction and storage
        news_sources = [item.get('source') for item in news]
        if RUNTIME_SETTINGS.get("enable_llm_topic_signals", True):
            topic_payload = extract_llm_topic_sentiment(news, macro_data, lang_mode)
            if topic_payload and topic_payload.get("signals"):
                recorded_topic_count = record_llm_topic_signals(topic_payload, news_sources)
                res['topic_signals'] = topic_payload
                res['_topic_signals_recorded'] = recorded_topic_count
            else:
                res['_topic_signals_recorded'] = 0
        
        # 【新增】记录交易信号
        if res.get("status") == "alert" and res.get("predictions"):
            predictions = res.get("predictions", {})
            impact_score = res.get("impact_score", 0)
            
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



def _safe_read_json(path):
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log_error(f"_safe_read_json error for {path}: {str(e)}")
        return {}


def _safe_read_text(path):
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        log_error(f"_safe_read_text error for {path}: {str(e)}")
        return ""


def _safe_read_jsonl(path, limit=5000):
    rows = []
    try:
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
        return rows
    except Exception as e:
        log_error(f"_safe_read_jsonl error for {path}: {str(e)}")
        return rows


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _build_position_weights(snapshot):
    positions = snapshot.get("positions", {})
    total_equity = _to_float(snapshot.get("total_equity", snapshot.get("equity", 0.0)))
    result = {}
    if not isinstance(positions, dict):
        return result

    for ticker, raw in positions.items():
        ticker_key = str(ticker).upper().strip()
        if not ticker_key:
            continue
        weight = 0.0
        if isinstance(raw, dict):
            if "weight" in raw:
                weight = _to_float(raw.get("weight", 0.0))
            elif "value" in raw and total_equity > 0:
                weight = _to_float(raw.get("value", 0.0)) / total_equity
        else:
            weight = _to_float(raw, 0.0)
        if weight > 0:
            result[ticker_key] = float(weight)
    return result


def _theme_colorize(text):
    lower = str(text).lower()
    positive_keys = ["bull", "positive", "up", "risk_on", "strong", "cooling", "cut"]
    negative_keys = ["bear", "negative", "down", "risk_off", "weak", "spike", "crisis"]
    if any(k in lower for k in negative_keys):
        return "#ff4b4b"
    if any(k in lower for k in positive_keys):
        return "#00c853"
    return "#dddddd"


@st.cache_data(ttl=30)
def _parse_equity_history_cached(history_payload):
    """Parse equity_history payload into a clean DataFrame."""
    try:
        raw = json.loads(history_payload)
    except Exception:
        return pd.DataFrame(columns=["time", "equity"])
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame(columns=["time", "equity"])

    rows = []
    for point in raw:
        if not isinstance(point, dict):
            continue
        rows.append(
            {
                "time": point.get("time"),
                "equity": _to_float(point.get("equity", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["time", "equity"])

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    if df.empty:
        return pd.DataFrame(columns=["time", "equity"])
    return df[["time", "equity"]]


@st.cache_data(ttl=30)
def _prepare_equity_curve_cached(history_payload, window_hours, resample_rule):
    """Apply time-window filtering and optional resampling."""
    df = _parse_equity_history_cached(history_payload)
    if df.empty:
        return df

    cutoff = df["time"].max() - pd.Timedelta(hours=int(window_hours))
    df = df[df["time"] >= cutoff]
    if df.empty:
        return pd.DataFrame(columns=["time", "equity"])

    if str(resample_rule).lower() != "raw":
        rule_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "1h": "1h",
            "1d": "1d",
        }
        rule = rule_map.get(str(resample_rule).lower())
        if rule:
            df = (
                df.set_index("time")["equity"]
                .resample(rule)
                .last()
                .dropna()
                .reset_index()
            )

    return df[["time", "equity"]]


def _infer_initial_equity(snapshot, summary_text, fallback_equity):
    """Infer initial equity from snapshot/summary, fallback to first point."""
    candidates = [
        snapshot.get("initial_cash"),
        snapshot.get("initial_cash_usd"),
        snapshot.get("initial_equity"),
        snapshot.get("starting_equity"),
    ]
    for value in candidates:
        num = _to_float(value, None)
        if num is not None and num > 0:
            return float(num)

    if isinstance(summary_text, str) and summary_text.strip():
        match = re.search(r"Initial Cash:\s*\$([0-9,]+(?:\.[0-9]+)?)", summary_text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except Exception:
                pass

    return float(fallback_equity)


def render_portfolio_monitor():
    st.title("\U0001F4BC Portfolio Monitor")
    st.caption("Live paper trading monitor (auto-refresh every 30 seconds)")

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=30000, key="portfolio_auto_refresh")
    except Exception:
        st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

    snapshot_path = "outputs/snapshot_live.json"
    summary_path = "outputs/paper_summary_live.txt"
    trades_path = "outputs/trade_history.jsonl"

    snapshot = _safe_read_json(snapshot_path)
    summary_text = _safe_read_text(summary_path)
    trades = _safe_read_jsonl(trades_path)

    total_equity = _to_float(snapshot.get("total_equity", snapshot.get("equity", 0.0)))
    cash = _to_float(snapshot.get("cash", 0.0))
    positions_value = _to_float(snapshot.get("positions_value", max(0.0, total_equity - cash)))
    drawdown = _to_float(snapshot.get("drawdown", 0.0))

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Equity", f"${total_equity:,.2f}")
    metric_col2.metric("Cash", f"${cash:,.2f}")
    metric_col3.metric("Positions", f"${positions_value:,.2f}")

    risk_cfg = snapshot.get("risk_config", {})
    if not isinstance(risk_cfg, dict):
        risk_cfg = {}
    cash_target = _to_float(risk_cfg.get("cash_target", snapshot.get("cash_target", 0.0)))
    risk_state = str(risk_cfg.get("risk_state", snapshot.get("regime_state", "UNKNOWN"))).upper()
    trend_score = _to_float(risk_cfg.get("regime_trend_score", snapshot.get("trend_score", 0.0)))
    regime_icon = "\U0001F7E2" if "RISK_ON" in risk_state else ("\U0001F534" if "RISK_OFF" in risk_state else "\U0001F7E1")
    st.markdown(
        f"**Regime**: {regime_icon} `{risk_state}` | "
        f"**cash_target**: `{cash_target:.2%}` | **trend_score**: `{trend_score:.2%}`"
    )

    st.subheader("Current Holdings Composition")
    position_weights = _build_position_weights(snapshot)
    if position_weights:
        position_df = pd.DataFrame(
            [{"ticker": t, "weight": w} for t, w in position_weights.items()]
        ).sort_values("weight", ascending=False).set_index("ticker")
        st.bar_chart(position_df)
    else:
        st.info("No position composition found in outputs/snapshot_live.json")

    st.subheader("Trade History")
    if trades:
        trade_rows = []
        for trade in trades:
            timestamp = trade.get("timestamp") or trade.get("time") or trade.get("datetime") or ""
            ticker = str(trade.get("ticker", ""))
            side = str(trade.get("side", trade.get("direction", "")))
            amount = _to_float(
                trade.get("cost", trade.get("notional", trade.get("amount", trade.get("desired_trade_value", 0.0))))
            )
            if "weight_change" in trade:
                weight_change = _to_float(trade.get("weight_change", 0.0))
            else:
                old_weight = _to_float(trade.get("old_target_weight", trade.get("current_weight", trade.get("old_weight", 0.0))))
                new_weight = _to_float(trade.get("new_target_weight", trade.get("target_weight", trade.get("new_weight", 0.0))))
                weight_change = new_weight - old_weight
            trade_rows.append(
                {
                    "time": str(timestamp),
                    "ticker": ticker,
                    "side": side,
                    "amount": amount,
                    "weight_change": weight_change,
                }
            )

        trade_df = pd.DataFrame(trade_rows)
        if not trade_df.empty:
            trade_df["time_sort"] = pd.to_datetime(trade_df["time"], errors="coerce")
            trade_df = trade_df.sort_values("time_sort", ascending=False).drop(columns=["time_sort"])
            st.dataframe(trade_df, width='stretch', hide_index=True)
    else:
        st.info("No trade history found in outputs/trade_history.jsonl")

    st.subheader("Theme Summary")
    last_macro = snapshot.get("last_macro", {})
    if not isinstance(last_macro, dict):
        last_macro = {}
    topic_summary = snapshot.get("topic_summary") or last_macro.get("topic_summary")
    macro_summary = snapshot.get("macro_summary") or last_macro.get("summary")
    if topic_summary:
        color = _theme_colorize(topic_summary)
        st.markdown(f"<span style='color:{color}'>{topic_summary}</span>", unsafe_allow_html=True)
    if macro_summary:
        st.markdown(macro_summary)

    theme_confidence = snapshot.get("theme_confidence", {})
    if isinstance(theme_confidence, dict) and theme_confidence:
        st.markdown("**Theme Confidence**")
        for theme_name, score in theme_confidence.items():
            score_val = _to_float(score, 0.0)
            color = "#00c853" if score_val >= 0 else "#ff4b4b"
            st.markdown(f"- <span style='color:{color}'>{theme_name}: {score_val:+.2f}</span>", unsafe_allow_html=True)

    st.subheader("Equity Curve")
    equity_history = snapshot.get("equity_history")
    if not isinstance(equity_history, list) or not equity_history:
        st.info("No equity_history found in outputs/snapshot_live.json")
    else:
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        window_hours = control_col1.selectbox("Window (hours)", [6, 12, 24, 48, 168], index=3, key="pm_window_hours")
        resample_rule = control_col2.selectbox(
            "Resample",
            ["raw", "1min", "5min", "15min", "1h", "1d"],
            index=3,
            key="pm_resample_rule",
        )
        x_tick_mode = control_col3.selectbox("X Tick Mode", ["auto", "fixed"], index=0, key="pm_x_tick_mode")
        x_tick_minutes = control_col4.selectbox("X Tick (min)", [5, 15, 30, 60], index=1, key="pm_x_tick_minutes")

        y_ctrl_col1, y_ctrl_col2, y_ctrl_col3, y_ctrl_col4 = st.columns(4)
        y_mode = y_ctrl_col1.selectbox("Y Mode", ["auto", "manual"], index=0, key="pm_y_mode")
        y_metric = y_ctrl_col2.selectbox(
            "Y Metric",
            ["equity", "pnl_from_initial"],
            index=0,
            key="pm_y_metric",
        )

        history_payload = json.dumps(equity_history, ensure_ascii=False, sort_keys=True)
        equity_df = _prepare_equity_curve_cached(history_payload, int(window_hours), str(resample_rule))
        if equity_df.empty:
            st.info("No valid equity points in selected window.")
        else:
            first_equity = _to_float(equity_df["equity"].iloc[0], 0.0)
            initial_equity = _infer_initial_equity(snapshot, summary_text, first_equity)
            y_series = equity_df["equity"] if y_metric == "equity" else (equity_df["equity"] - initial_equity)
            plot_df = pd.DataFrame({"time": equity_df["time"], "y": y_series})

            y_min_default = float(plot_df["y"].min())
            y_max_default = float(plot_df["y"].max())
            if y_max_default <= y_min_default:
                y_max_default = y_min_default + 1.0

            manual_inputs_disabled = (y_mode != "manual")
            y_min = y_ctrl_col3.number_input(
                "Y Min",
                value=float(y_min_default),
                step=1.0,
                disabled=manual_inputs_disabled,
                key="pm_y_min",
            )
            y_max = y_ctrl_col4.number_input(
                "Y Max",
                value=float(y_max_default),
                step=1.0,
                disabled=manual_inputs_disabled,
                key="pm_y_max",
            )

            y_opt_col1, y_opt_col2 = st.columns(2)
            y_dtick = y_opt_col1.number_input(
                "Y dtick (0=auto)",
                value=0.0,
                step=1.0,
                disabled=manual_inputs_disabled,
                key="pm_y_dtick",
            )
            y_title = "PnL ($)" if y_metric == "pnl_from_initial" else "Equity ($)"
            x_tick_label = "auto" if x_tick_mode == "auto" else f"{x_tick_minutes}min"
            y_mode_label = "auto"
            if y_mode == "manual":
                y_mode_label = f"manual [{y_min:.2f}, {y_max:.2f}]"
            st.caption(f"X: last {window_hours}h @ {resample_rule} | X tick: {x_tick_label} | Y: {y_metric} {y_mode_label}")

            # Plotly first, Altair fallback.
            plot_rendered = False
            try:
                import plotly.graph_objects as pgo

                fig = pgo.Figure()
                fig.add_trace(
                    pgo.Scatter(
                        x=plot_df["time"],
                        y=plot_df["y"],
                        mode="lines",
                        name=y_title,
                        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:,.2f}<extra></extra>",
                    )
                )

                xaxis_config = {"title": "Time"}
                if x_tick_mode == "fixed":
                    dtick_ms = int(x_tick_minutes) * 60 * 1000
                    time_span = (plot_df["time"].max() - plot_df["time"].min()).total_seconds()
                    tick_fmt = "%m-%d %H:%M" if time_span <= 7 * 24 * 3600 else "%m-%d"
                    xaxis_config.update(
                        {
                            "tickmode": "linear",
                            "dtick": dtick_ms,
                            "tickformat": tick_fmt,
                        }
                    )

                yaxis_config = {"title": y_title}
                if y_mode == "manual":
                    y_low = float(min(y_min, y_max))
                    y_high = float(max(y_min, y_max))
                    yaxis_config["range"] = [y_low, y_high]
                    if y_dtick and y_dtick > 0:
                        yaxis_config["dtick"] = float(y_dtick)

                fig.update_layout(
                    margin={"l": 20, "r": 20, "t": 20, "b": 20},
                    xaxis=xaxis_config,
                    yaxis=yaxis_config,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")
                plot_rendered = True
            except Exception:
                plot_rendered = False

            if not plot_rendered:
                try:
                    import altair as alt

                    y_scale = alt.Scale()
                    if y_mode == "manual":
                        y_scale = alt.Scale(domain=[float(min(y_min, y_max)), float(max(y_min, y_max))])

                    x_axis = alt.Axis(title="Time")
                    if x_tick_mode == "fixed":
                        tick_count = max(2, int((int(window_hours) * 60) / max(1, int(x_tick_minutes))))
                        x_axis = alt.Axis(title="Time", tickCount=tick_count)

                    y_axis = alt.Axis(title=y_title)
                    if y_mode == "manual" and y_dtick and y_dtick > 0:
                        y_axis = alt.Axis(title=y_title, tickMinStep=float(y_dtick))

                    chart = (
                        alt.Chart(plot_df)
                        .mark_line()
                        .encode(
                            x=alt.X("time:T", axis=x_axis),
                            y=alt.Y("y:Q", axis=y_axis, scale=y_scale),
                            tooltip=[alt.Tooltip("time:T", title="Time"), alt.Tooltip("y:Q", title=y_title, format=",.2f")],
                        )
                        .interactive()
                    )
                    st.altair_chart(chart, width="stretch")
                except Exception as e:
                    st.info(f"Unable to render equity curve with Plotly/Altair: {e}")

    if summary_text:
        with st.expander("Live Summary Text", expanded=False):
            st.text(summary_text)

    cash_ratio = (cash / total_equity) if total_equity > 1e-12 else 0.0
    if drawdown >= 0.10:
        st.error(f"Major drawdown warning: {drawdown:.2%}")
    if cash_ratio > 0.90 and total_equity > 0:
        st.error(f"Position anomaly warning: cash ratio too high ({cash_ratio:.2%})")

st.set_page_config(page_title="GlobalWatch DeepSeek Edition", layout="wide", page_icon="🦁")

st.sidebar.header("⚙️ Settings")
st.sidebar.caption(f"Brain: {LOCAL_MODEL}")

page_choice = st.sidebar.selectbox(
    "Page",
    ["\U0001F4E1 Global Macro Signals", "\U0001F4BC Portfolio Monitor"],
    index=0
)

# 新增：展示宏观规则库
with st.sidebar.expander("📚 Macro Rules Library"):
    st.text(MACRO_LOGIC_KNOWLEDGE)

lang_mode = st.sidebar.radio("Language", ["中文", "English"], index=0)
refresh_label = st.sidebar.selectbox("Refresh Rate", list(REFRESH_OPTIONS.keys()), index=0)
refresh_sec = REFRESH_OPTIONS[refresh_label]
enable_toast = st.sidebar.checkbox("Desktop Notify", value=True)
auto_run = st.sidebar.checkbox("Auto Run", value=True)

if 'last_run' not in st.session_state: st.session_state['last_run'] = datetime.now() - timedelta(days=1)

if page_choice == "\U0001F4BC Portfolio Monitor":
    render_portfolio_monitor()
    st.stop()

st.title("🦁 GlobalWatch: DeepSeek-R1 推理版")
st.caption("🚀 Powered by Chain-of-Thought Reasoning")
st.divider()

tab_macro, tab_stock, tab_scoreboard, tab_warning = st.tabs(["🌍 宏观/外汇 (Macro/FX)", "🇺🇸 美股透视 (US Stocks)", "📊 Signal Scoreboard", "🚨 Early-Warning"])

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
                        except Exception as e:
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
    
    # ========== 新增：交易级性能分类 ==========
    st.subheader("🎯 Trading-Grade Performance Classification")
    st.caption("⚠️ 这是决定是否允许 real-money execution 的唯一依据")
    
    # 获取交易级分类
    classification = classify_trading_performance(
        stats, 
        theme=theme, 
        asset=None,  # 可以改为特定资产
        transaction_cost=0.1,  # 0.1% 交易成本
        max_dd_threshold=15.0  # 15% 最大回撤阈值
    )
    
    # 显示分类结果
    col_class, col_decision = st.columns([2, 1])
    
    with col_class:
        # 分类标签
        class_v2 = classification['classification_v2']
        
        # 根据分类设置颜色
        if "🟢" in class_v2:
            st.success(f"### {class_v2}")
        elif "🟡" in class_v2:
            st.warning(f"### {class_v2}")
        elif "🟠" in class_v2:
            st.warning(f"### {class_v2}")
        elif "🔴" in class_v2:
            st.error(f"### {class_v2}")
        else:
            st.info(f"### {class_v2}")
    
    with col_decision:
        # 交易决策
        if classification['decision_allowed']:
            st.success("### ✅ TRADABLE")
            st.caption("允许实盘交易")
        else:
            st.error("### 🚫 NOT TRADABLE")
            st.caption("禁止实盘交易")
    
    # 显示详细原因
    with st.expander("📋 Classification Details", expanded=True):
        st.markdown("**原因说明：**")
        st.info(classification['reason_summary'])
        
        # 风险警告
        if classification['risk_warnings']:
            st.markdown("**风险警告：**")
            for warning in classification['risk_warnings']:
                st.markdown(f"- {warning}")
        
        # 关键指标
        st.markdown("**关键指标：**")
        col_i1, col_i2, col_i3 = st.columns(3)
        
        col_i1.metric(
            "Net Expected Value",
            f"{classification['net_expected_value']:.2f}%",
            help="平均收益 - 交易成本"
        )
        
        col_i2.metric(
            "Max Drawdown",
            f"{stats['max_drawdown']:.2f}%",
            help="最大回撤"
        )
        
        col_i3.metric(
            "Multi-TF Validated",
            "✅ Yes" if classification['multi_timeframe_validated'] else "❌ No",
            help="是否通过多时间窗口验证"
        )
    
    # V1 分类（仅供参考）
    with st.expander("📊 V1 Classification (Reference Only)", expanded=False):
        st.caption("⚠️ 以下分类仅供分析参考，不可用于交易决策")
        st.markdown(f"**V1 Classification**: {classification['classification_v1']}")
        
        st.markdown("**V1 分类说明：**")
        if "Positive Edge" in classification['classification_v1']:
            st.success("✅ **Positive Edge (V1)**: 高准确率 + 正收益")
        elif "High Accuracy" in classification['classification_v1']:
            st.warning("⚠️ **High Accuracy, Low Returns (V1)**: 方向对但收益小")
        elif "Lucky Streak" in classification['classification_v1']:
            st.info("ℹ️ **Lucky Streak (V1)**: 低准确率但正收益（可能是运气）")
        elif "No Edge" in classification['classification_v1']:
            st.error("❌ **No Edge (V1)**: 低准确率 + 负收益")
        else:
            st.info("No data")
    
    st.divider()
    
    # 增强的统计指标
    st.subheader("📊 Enhanced Statistics")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    col_s1.metric(
        "Cumulative Return",
        f"{stats['cumulative_return']:+.2f}%",
        help="所有信号的累计收益"
    )
    
    col_s2.metric(
        "Win Rate",
        f"{stats['win_rate']:.1f}%",
        help="盈利信号占比"
    )
    
    col_s3.metric(
        "Profit Factor",
        f"{stats['profit_factor']:.2f}",
        help="总盈利 / 总亏损"
    )
    
    col_s4.metric(
        "Volatility",
        f"{stats['volatility']:.2f}%",
        help="收益标准差"
    )
    
    # 盈亏分布
    col_w, col_l = st.columns(2)
    
    with col_w:
        st.metric(
            "Avg Win",
            f"{stats['avg_win']:+.2f}%",
            help="平均盈利"
        )
    
    with col_l:
        st.metric(
            "Avg Loss",
            f"{stats['avg_loss']:+.2f}%",
            help="平均亏损"
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
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("No signals recorded yet. Run some analyses to start tracking!")
    
    except Exception as e:
        st.error(f"Error loading recent signals: {e}")
    
    st.divider()
    
    # 使用说明
    with st.expander("ℹ️ How to Use Signal Scoreboard (V2 - Trading-Grade)"):
        st.markdown("""
        ### Signal Tracking System
        
        **Automatic Recording**:
        - Every time you run an analysis, predictions are automatically recorded
        - Initial price is captured at the time of prediction
        
        **Result Backfilling**:
        - Click "🔄 Update Results" to check and update signal outcomes
        - System checks if enough time has passed (1h/4h/1d/1w)
        - Fetches actual prices and calculates returns
        
        ---
        
        ### 🎯 Trading-Grade Performance Classification (V2)
        
        **这是决定是否允许 real-money execution 的唯一依据**
        
        #### 分类体系：
        
        **🟢 Tradable Edge（允许实盘）**
        - 样本数 ≥ 50
        - 净期望值 > 0（扣除交易成本后）
        - 最大回撤 ≤ 15%
        - 至少 2 个时间窗口验证通过
        - **→ 只有此分类允许实盘交易**
        
        **🟡 Directional Signal（方向参考）**
        - 准确率 ≥ 58% 但净期望值 ≈ 0
        - 或收益被交易成本侵蚀
        - **→ 仅用于：仓位调整、风险确认、多信号过滤**
        - **→ 禁止单独触发交易**
        
        **🟠 Unstable / Regime-Dependent（不稳定）**
        - 净期望值 > 0
        - 但样本量不足 或 回撤过大 或 未通过多时间窗口验证
        - **→ 标记为「观察中」**
        - **→ 禁止自动交易**
        - **→ 需要更多数据验证**
        
        **🔴 No Edge（无优势）**
        - 净期望值 ≤ 0
        - 或回撤严重超标
        - 或准确率和收益均无意义
        - **→ 永久禁止交易**
        
        **🟤 Insufficient Data（数据不足）**
        - 样本数 < 30
        - **→ 任何结论都不可靠**
        - **→ 禁止交易**
        
        ---
        
        #### Lucky Streak 的处理（重要）：
        
        - 原 V1 的 "Lucky Streak"（低准确率 + 正收益）不再作为正向分类
        - 一律并入 🟠 Unstable 或 🔴 No Edge
        - **除非**存在明确的非对称收益结构（fat-tail payoff）：
          - 平均盈利 > 2 × 平均亏损（绝对值）
          - Profit Factor > 2.0
        
        ---
        
        #### 关键指标说明：
        
        - **Net Expected Value**: 平均收益 - 交易成本（默认 0.1%）
        - **Max Drawdown**: 从峰值到谷底的最大跌幅
        - **Profit Factor**: 总盈利 / 总亏损（> 1 为盈利）
        - **Win Rate**: 盈利信号占比
        - **Volatility**: 收益标准差（波动性）
        - **Multi-TF Validated**: 是否在多个时间窗口都为正
        
        ---
        
        #### V1 vs V2 分类：
        
        - **V1 分类**（旧版）：仅供分析参考，不可用于交易决策
        - **V2 分类**（新版）：交易级标准，是实盘交易的唯一依据
        
        ---
        
        ### ⚠️ 重要原则
        
        1. **样本数不足时，系统会主动拒绝**
           - 即使看起来"很准"，样本 < 30 也不允许交易
        
        2. **清楚区分「分析上有意思」vs「可以用真钱」**
           - 分析上有意思 → V1 分类
           - 可以用真钱 → 只有 V2 的 🟢 Tradable Edge
        
        3. **Real-money execution 只接受 Tradable Edge**
           - 其他所有分类都禁止实盘交易
           - 没有例外
        
        ---
        
        ### 📊 使用建议
        
        1. **定期回填结果**：每天点击 "🔄 Update Results"
        2. **关注分类变化**：从 Unstable → Tradable Edge 需要时间
        3. **多时间窗口验证**：切换不同 timeframe 查看一致性
        4. **风险管理**：即使是 Tradable Edge，也要控制仓位
        5. **持续监控**：市场环境变化可能导致分类降级
        
        ---
        
        **Important Notes**:
        - Returns are theoretical (no transaction costs in calculation, but considered in classification)
        - Past performance doesn't guarantee future results
        - Use this data to validate your strategy before risking real money
        - Only 🟢 Tradable Edge signals are approved for live trading
        """)


# === TAB 4: Early-Warning ===
with tab_warning:
    st.header("🚨 Early-Warning Risk Monitor")
    st.caption("Universal risk scoring system for monitored assets")
    
    st.divider()
    
    # 监控列表选择
    st.subheader("📋 Watchlist")
    
    col_select, col_analyze = st.columns([3, 1])
    
    selected_asset = col_select.selectbox(
        "Select Asset to Analyze",
        list(WATCHLIST.keys()),
        index=0
    )
    
    if col_analyze.button("🔍 Calculate Risk Score"):
        with st.spinner(f"Analyzing risk for {selected_asset}..."):
            # 获取最新新闻
            recent_news = st.session_state.get('news', [])
            if not recent_news:
                recent_news = get_rss_news()
            
            # 计算风险分数
            risk_result = calculate_early_warning_score(selected_asset, recent_news)
            
            # 存储到 session state
            st.session_state['risk_result'] = risk_result
            st.rerun()
    
    st.divider()
    
    # 显示风险评分结果
    if 'risk_result' in st.session_state:
        risk = st.session_state['risk_result']
        
        if 'error' in risk:
            st.error(f"Error: {risk['error']}")
        else:
            # 风险总览
            st.subheader(f"📊 Risk Assessment: {risk['asset']}")
            
            # 综合风险分数
            total_score = risk['total_risk_score']
            risk_level = risk['risk_level']
            
            # 风险等级颜色
            level_colors = {
                "LOW": "green",
                "MEDIUM": "yellow",
                "HIGH": "orange",
                "CRITICAL": "red"
            }
            level_color = level_colors.get(risk_level, "gray")
            
            # 显示综合分数
            col_score, col_level = st.columns(2)
            
            col_score.metric(
                "Total Risk Score",
                f"{total_score}/100",
                help="Combined score from all risk dimensions"
            )
            
            col_level.markdown(f"""
            <div style="padding:20px; background-color:{level_color}; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # 四维子分数
            st.subheader("📈 Risk Breakdown")
            
            sub_scores = risk['sub_scores']
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "🌐 Macro Chain",
                f"{sub_scores['macro_chain']['score']}/25",
                help="USD/rates/macro environment impact"
            )
            
            col2.metric(
                "👥 Crowding",
                f"{sub_scores['crowding']['score']}/25",
                help="Technical overbought/oversold levels"
            )
            
            col3.metric(
                "📊 Microstructure",
                f"{sub_scores['microstructure']['score']}/25",
                help="Volatility/gaps/volume anomalies"
            )
            
            col4.metric(
                "⚡ Event Risk",
                f"{sub_scores['event_risk']['score']}/25",
                help="Central bank/policy/geopolitical events"
            )
            
            st.divider()
            
            # 雷达图
            st.subheader("🎯 Risk Radar")
            
            categories = ['Macro Chain', 'Crowding', 'Microstructure', 'Event Risk']
            values = [
                sub_scores['macro_chain']['score'],
                sub_scores['crowding']['score'],
                sub_scores['microstructure']['score'],
                sub_scores['event_risk']['score']
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # 闭合图形
                theta=categories + [categories[0]],
                fill='toself',
                name='Risk Score',
                line_color='red'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 25]
                    )
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.divider()
            
            # 证据链展示
            st.subheader("📋 Evidence Chain")
            
            # Macro Chain Evidence
            with st.expander("🌐 Macro Chain Evidence", expanded=True):
                macro_evidence = sub_scores['macro_chain']['evidence']
                if macro_evidence:
                    for idx, ev in enumerate(macro_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"⚠️ {ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Type**: {ev.get('type', 'N/A')}")
                            if 'indicator' in ev:
                                st.markdown(f"- **Indicator**: {ev.get('indicator')}")
                            if 'value' in ev:
                                st.markdown(f"- **Value**: {ev.get('value')}")
                            if 'count' in ev:
                                st.markdown(f"- **Count**: {ev.get('count')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No macro chain risks detected")
            
            # Crowding Evidence
            with st.expander("👥 Crowding Evidence"):
                crowding_evidence = sub_scores['crowding']['evidence']
                if crowding_evidence:
                    for idx, ev in enumerate(crowding_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"⚠️ {ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Indicator**: {ev.get('indicator', 'N/A')}")
                            st.markdown(f"- **Value**: {ev.get('value', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No crowding risks detected")
            
            # Microstructure Evidence
            with st.expander("📊 Microstructure Evidence"):
                micro_evidence = sub_scores['microstructure']['evidence']
                if micro_evidence:
                    for idx, ev in enumerate(micro_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"⚠️ {ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Indicator**: {ev.get('indicator', 'N/A')}")
                            st.markdown(f"- **Value**: {ev.get('value', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No microstructure risks detected")
            
            # Event Risk Evidence
            with st.expander("⚡ Event Risk Evidence"):
                event_evidence = sub_scores['event_risk']['evidence']
                if event_evidence:
                    for idx, ev in enumerate(event_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"⚠️ {ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Category**: {ev.get('category', 'N/A')}")
                            st.markdown(f"- **Count**: {ev.get('count', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No event risks detected")
            
            st.divider()
            
            # 告警触发器
            if risk['alert_triggers']:
                st.subheader("⚠️ Alert Triggers")
                for trigger in risk['alert_triggers']:
                    st.warning(f"• {trigger}")
            
            # 建议
            st.subheader("💡 Recommendation")
            st.info(risk['recommendation'])
            
            # 时间戳
            st.caption(f"Analysis Time: {risk['timestamp'][:19]}")
    
    else:
        st.info("👆 Select an asset and click 'Calculate Risk Score' to begin analysis")
    
    st.divider()
    
    # 使用说明
    with st.expander("ℹ️ How to Use Early-Warning System"):
        st.markdown("""
        ### Early-Warning Risk Scoring System
        
        **Purpose**:
        - Detect elevated risk BEFORE major moves
        - Provide evidence-based risk assessment
        - Help answer: "Should I reduce exposure now?"
        
        **Four Risk Dimensions**:
        
        1. **🌐 Macro Chain (0-25)**:
           - USD strength/weakness impact
           - Interest rate movements
           - Macro news flow
           - Based on correlations with DXY, 10Y yields
        
        2. **👥 Crowding (0-25)**:
           - RSI overbought/oversold levels
           - Price deviation from moving averages
           - Volume spikes indicating crowding
        
        3. **📊 Microstructure (0-25)**:
           - Volatility surges (ATR ratio)
           - Price gaps
           - Volume anomalies
        
        4. **⚡ Event Risk (0-25)**:
           - Central bank events (Fed, ECB)
           - Policy changes (tariffs, regulations)
           - Geopolitical tensions
        
        **Risk Levels**:
        - 🟢 **LOW (0-25)**: Normal market environment
        - 🟡 **MEDIUM (26-50)**: Some factors elevated, monitor
        - 🟠 **HIGH (51-75)**: Multiple risks, consider caution
        - 🔴 **CRITICAL (76-100)**: Extreme risk, reduce exposure
        
        **Current Watchlist**:
        - Gold (GC=F)
        - Oil (CL=F)
        - CNY (CNY=X)
        - CAD (CAD=X)
        
        **Important Notes**:
        - Risk scores are RELATIVE, not absolute
        - Use with Signal Scoreboard to validate effectiveness
        - Combine with other analysis methods
        - Not a trading signal - a risk management tool
        
        **Best Practices**:
        - Run analysis regularly (daily or before major positions)
        - Compare scores over time to identify trends
        - Pay attention to evidence - not just the score
        - Act on CRITICAL levels, monitor HIGH levels
        """)


