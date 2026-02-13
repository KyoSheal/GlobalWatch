import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import ollama
import argparse
import sys
from datetime import datetime, timedelta, timezone
import time
import json
import re
import hashlib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import urllib.parse
import urllib.request
import os
from atomic_io import atomic_write_json as io_atomic_write_json, safe_read_json as io_safe_read_json

try:
    import chromadb
    CHROMADB_IMPORTED = True
except Exception:
    chromadb = None  # type: ignore
    CHROMADB_IMPORTED = False

try:
    import daily_reporter
except Exception:
    daily_reporter = None  # type: ignore

# === 0. Base Setup ===
try:
    from plyer import notification
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# === 0.1. Logging Helpers ===
def log_error(message):
    """
    Write error messages to file and console safely.
    Args:
        message: Error message string.
    """
    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Append to error log file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open("outputs/error.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        # Logging itself should never crash the app
        print(f"Logging failed: {e}")

def safe_format_number(value, decimals=2, default="N/A"):
    """
    Safely format numeric values and handle None/NaN/inf.
    Args:
        value: numeric value
        decimals: decimal precision
        default: fallback text when value is invalid
    Returns:
        Formatted number string
    """
    try:
        if value is None:
            return default
        if pd.isna(value) or not pd.api.types.is_number(value):
            return default
        if not (-1e10 < value < 1e10):  # inf guard
            return default
        return f"{value:,.{decimals}f}"
    except Exception as e:
        log_error(f"safe_format_number error: {str(e)}")
        return default

# Model configuration (ensure `ollama pull gemma3:12` is available)
LOCAL_MODEL = str(os.environ.get("GW_LOCAL_MODEL", "gemma3:12b"))
# Temperature range: 0.1 ~ 0.3 (configured in ollama.chat calls)
TEMPERATURE = 0.2  # Default temperature for model calls

try:
    DEFAULT_LLM_NUM_CTX = max(1024, int(os.environ.get("GW_OLLAMA_NUM_CTX", "4096")))
except Exception:
    DEFAULT_LLM_NUM_CTX = 4096
try:
    DEFAULT_OLLAMA_RETRIES = max(0, int(os.environ.get("GW_OLLAMA_RETRIES", "2")))
except Exception:
    DEFAULT_OLLAMA_RETRIES = 2
try:
    DEFAULT_OLLAMA_RETRY_BACKOFF = max(0.2, float(os.environ.get("GW_OLLAMA_RETRY_BACKOFF", "1.5")))
except Exception:
    DEFAULT_OLLAMA_RETRY_BACKOFF = 1.5
OLLAMA_KEEP_ALIVE = str(os.environ.get("GW_OLLAMA_KEEP_ALIVE", "20m"))
OLLAMA_FALLBACK_MODELS = [
    x.strip()
    for x in str(os.environ.get("GW_OLLAMA_FALLBACK_MODELS", "deepseek-r1:8b")).split(",")
    if str(x).strip()
]

# Runtime throttle for industry-news pipeline calls from UI.
_INDUSTRY_PIPELINE_LAST_RUN_TS = 0.0

class _NoopCollection:
    """Fallback collection that keeps the app running when Chroma is unavailable."""

    def add(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None
    
    def get(self, *args, **kwargs):
        return {"ids": [], "metadatas": [], "documents": []}

    def query(self, *args, **kwargs):
        return {"ids": [[]], "metadatas": [[]], "documents": [[]]}


CHROMA_AVAILABLE = False
CHROMA_INIT_ERROR = ""
chroma_client = None
collection = _NoopCollection()
signals_collection = _NoopCollection()

if CHROMADB_IMPORTED:
    try:
        # 
        chroma_client = chromadb.PersistentClient(path="./memory_db")
        collection = chroma_client.get_or_create_collection(name="market_events")
        # 
        signals_collection = chroma_client.get_or_create_collection(name="trading_signals")
        CHROMA_AVAILABLE = True
        if str(os.environ.get("GW_CHROMA_VERBOSE", "0")).strip() in {"1", "true", "TRUE", "yes"}:
            print("[CHROMA] PersistentClient ready: ./memory_db")
    except Exception as e:
        CHROMA_INIT_ERROR = str(e)
        log_error(f"Chroma initialization failed, using no-op memory: {CHROMA_INIT_ERROR}")
        print(f"[WARN] Chroma disabled (no-op memory): {CHROMA_INIT_ERROR}")
else:
    CHROMA_INIT_ERROR = "chromadb import failed"
    print("[WARN] Chroma disabled: chromadb package not available")

# Macro reasoning knowledge injected into prompts
MACRO_LOGIC_KNOWLEDGE = """
GLOBAL MACRO RULES:
1. CAD (Loonie) is a Petro-currency. Oil UP -> CAD Stronger.
2. CNY (Yuan) is sensitive to USD Strength & Trade Wars.
3. USD is Safe Haven. Crisis -> Capital flows to USD/Gold.
4. TECH STOCKS (e.g. NVDA) are sensitive to Interest Rates & AI hype.
"""

# Early-Warning 
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
    "USD (US Dollar)": {"ticker": "USD", "type": "fiat_base"},
    "CNY (Chinese Yuan)": {"ticker": "CNY=X", "type": "fiat_quote"},
    "CAD (Canadian Dollar)": {"ticker": "CAD=X", "type": "fiat_quote"},
    "GBP (British Pound)": {"ticker": "GBP=X", "type": "fiat_quote"},
    "JPY (Japanese Yen)": {"ticker": "JPY=X", "type": "fiat_quote"},
    "Gold (Futures)": {"ticker": "GC=F", "type": "commodity"},
    "Crude Oil (Futures)": {"ticker": "CL=F", "type": "commodity"},
    "Bitcoin (Spot)": {"ticker": "BTC-USD", "type": "crypto"},
}

MACRO_ANCHORS = {"Crude Oil": "CL=F", "Gold": "GC=F"}

RSS_FEEDS = {
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "CNBC": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml"
}

REFRESH_OPTIONS = {"Manual": 0, "5 min": 300, "10 min": 600, "30 min": 1800}

# ================= 1.  (V3.0 ) =================

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

CANONICAL_INDUSTRY_BUCKETS = [
    "technology",
    "energy",
    "financials",
    "healthcare",
    "industrials",
    "industrials_defense",
    "consumer",
    "utilities",
    "real_estate",
    "broad_market_etf",
    "rates_and_gold",
    "cash_equivalent",
]

# Baseline expectation is only for sanity-check diagnostics; not used for trading decisions.
BASELINE_INDUSTRY_EXPECTATION = {
    "technology": {"direction_label": "overweight", "risk_delta": 0.15, "confidence": 0.60},
    "energy": {"direction_label": "neutral_to_overweight", "risk_delta": 0.05, "confidence": 0.55},
    "financials": {"direction_label": "overweight", "risk_delta": 0.10, "confidence": 0.60},
    "healthcare": {"direction_label": "slight_overweight", "risk_delta": 0.05, "confidence": 0.55},
    "industrials": {"direction_label": "neutral", "risk_delta": 0.00, "confidence": 0.50},
    "industrials_defense": {"direction_label": "overweight", "risk_delta": 0.25, "confidence": 0.70},
    "consumer": {"direction_label": "underweight", "risk_delta": -0.05, "confidence": 0.55},
    "utilities": {"direction_label": "underweight", "risk_delta": -0.10, "confidence": 0.65},
    "real_estate": {"direction_label": "underweight", "risk_delta": -0.05, "confidence": 0.60},
    "broad_market_etf": {"direction_label": "slight_overweight", "risk_delta": 0.05, "confidence": 0.55},
    "rates_and_gold": {"direction_label": "neutral", "risk_delta": 0.00, "confidence": 0.50},
    "cash_equivalent": {"direction_label": "slight_overweight", "risk_delta": 0.05, "confidence": 0.55},
}


def _direction_from_delta(value):
    v = float(_safe_float(value, 0.0))
    if v > 1e-12:
        return "overweight"
    if v < -1e-12:
        return "underweight"
    return "neutral"


def _normalize_direction_text(value):
    txt = str(value or "").strip().lower()
    if txt in {"overweight", "bullish", "positive", "up"}:
        return "overweight"
    if txt in {"underweight", "bearish", "negative", "down"}:
        return "underweight"
    if txt in {"neutral", "mixed", "flat", ""}:
        return "neutral"
    if "underweight" in txt:
        return "underweight"
    if "overweight" in txt:
        return "overweight"
    return "neutral"


def _load_latest_industry_signals_per_bucket(config_path="paper_config.json"):
    latest = {}
    rows_out = []
    meta = {"status": "ok", "error": None, "collection": "industry_signals", "count": 0}
    try:
        cfg = _load_json_config(config_path)
    except Exception as e:
        meta["status"] = "config_error"
        meta["error"] = str(e)
        cfg = {}

    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    collection_name = str(overlay_cfg.get("industry_collection", "industry_signals"))
    chroma_path = str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    meta["collection"] = collection_name
    meta["chroma_path"] = chroma_path

    if not CHROMADB_IMPORTED:
        meta["status"] = "chroma_unavailable"
        return latest, rows_out, meta

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_or_create_collection(name=collection_name)
        data = coll.get(include=["metadatas", "documents"])
        metadatas = data.get("metadatas", []) if isinstance(data, dict) else []
        documents = data.get("documents", []) if isinstance(data, dict) else []
        meta["count"] = len(metadatas)

        for idx, md in enumerate(metadatas):
            if not isinstance(md, dict):
                continue
            if str(md.get("scope", "")).lower() != "industry":
                continue

            l2 = str(md.get("L2", "")).strip()
            if not l2:
                continue

            ts_raw = md.get("timestamp")
            ts = _parse_iso_or_none(ts_raw)
            if ts is None:
                continue

            risk_delta = float(np.clip(_safe_float(md.get("risk_delta", 0.0), 0.0), -1.0, 1.0))
            confidence = float(np.clip(_safe_float(md.get("confidence", 0.0), 0.0), 0.0, 1.0))
            direction = _direction_from_delta(risk_delta)
            doc_obj = {}

            doc_text = documents[idx] if idx < len(documents) else None
            if isinstance(doc_text, str) and doc_text.strip():
                try:
                    doc_obj = json.loads(doc_text)
                except Exception:
                    doc_obj = {}

            if isinstance(doc_obj, dict):
                direction = _normalize_direction_text(
                    doc_obj.get("direction")
                    or (doc_obj.get("signal", {}) if isinstance(doc_obj.get("signal"), dict) else {}).get("direction")
                    or direction
                )
                if "risk_delta" in doc_obj:
                    risk_delta = float(np.clip(_safe_float(doc_obj.get("risk_delta", risk_delta), risk_delta), -1.0, 1.0))
                elif isinstance(doc_obj.get("signal"), dict) and "strength" in doc_obj.get("signal", {}):
                    s = float(np.clip(_safe_float(doc_obj["signal"].get("strength", 0.0), 0.0), -1.0, 1.0))
                    if direction == "overweight":
                        risk_delta = abs(s)
                    elif direction == "underweight":
                        risk_delta = -abs(s)
                    else:
                        risk_delta = 0.0
                if "confidence" in doc_obj:
                    confidence = float(np.clip(_safe_float(doc_obj.get("confidence", confidence), confidence), 0.0, 1.0))
                elif isinstance(doc_obj.get("signal"), dict) and "confidence" in doc_obj.get("signal", {}):
                    confidence = float(np.clip(_safe_float(doc_obj["signal"].get("confidence", confidence), confidence), 0.0, 1.0))

            row = {
                "bucket": l2,
                "local_direction": direction,
                "local_risk_delta": float(risk_delta),
                "local_confidence": float(confidence),
                "local_timestamp": ts.isoformat(),
            }
            prev = latest.get(l2)
            if prev is None:
                latest[l2] = row
            else:
                prev_ts = _parse_iso_or_none(prev.get("local_timestamp"))
                if prev_ts is None or ts > prev_ts:
                    latest[l2] = row
    except Exception as e:
        meta["status"] = "read_error"
        meta["error"] = str(e)

    for bucket in CANONICAL_INDUSTRY_BUCKETS:
        if bucket in latest:
            rows_out.append(dict(latest[bucket]))
        else:
            rows_out.append(
                {
                    "bucket": bucket,
                    "local_direction": "missing",
                    "local_risk_delta": None,
                    "local_confidence": None,
                    "local_timestamp": None,
                }
            )
    return latest, rows_out, meta


def _compute_effective_cash_impact_for_bucket(row, news_overlay_cfg):
    if not isinstance(row, dict):
        return 0.0
    if row.get("local_risk_delta") is None or row.get("local_confidence") is None:
        return 0.0
    try:
        alpha = float(np.clip(_safe_float(news_overlay_cfg.get("alpha", 0.08), 0.08), 0.0, 1.0))
        max_abs_delta = float(np.clip(_safe_float(news_overlay_cfg.get("max_abs_delta", 0.10), 0.10), 0.0, 1.0))
        min_conf = float(np.clip(_safe_float(news_overlay_cfg.get("min_confidence", 0.55), 0.55), 0.0, 1.0))
        mode = str(news_overlay_cfg.get("mode", "risk_only")).strip().lower()
        max_age_hours = float(max(0.0, _safe_float(news_overlay_cfg.get("max_age_hours", 48), 48.0)))
        confidence = float(np.clip(_safe_float(row.get("local_confidence", 0.0), 0.0), 0.0, 1.0))
        if confidence < min_conf:
            return 0.0
        ts = _parse_iso_or_none(row.get("local_timestamp"))
        if ts is None:
            return 0.0
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        if age_h > max_age_hours:
            return 0.0
        risk_delta = float(np.clip(_safe_float(row.get("local_risk_delta", 0.0), 0.0), -1.0, 1.0))
        delta = float(np.clip(risk_delta * alpha, -max_abs_delta, max_abs_delta))
        if mode == "risk_only" and delta > 0:
            delta = 0.0
        return float(np.clip(abs(min(0.0, delta)), 0.0, max_abs_delta))
    except Exception:
        return 0.0


def build_industry_sanity_check_report(config_path="paper_config.json"):
    try:
        cfg = _load_json_config(config_path)
    except Exception:
        cfg = {}
    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    latest_map, local_rows, read_meta = _load_latest_industry_signals_per_bucket(config_path=config_path)

    rows = []
    local_risks = []
    local_dirs = []
    baseline_risks = []
    for row in local_rows:
        bucket = str(row.get("bucket", "")).strip()
        base = BASELINE_INDUSTRY_EXPECTATION.get(bucket, {"direction_label": "neutral", "risk_delta": 0.0, "confidence": 0.5})
        baseline_risk = float(_safe_float(base.get("risk_delta", 0.0), 0.0))
        baseline_conf = float(_safe_float(base.get("confidence", 0.5), 0.5))
        baseline_dir = _direction_from_delta(baseline_risk)

        local_risk = row.get("local_risk_delta")
        local_conf = row.get("local_confidence")
        local_dir = str(row.get("local_direction", "missing"))
        if local_risk is not None:
            local_risk = float(local_risk)
            local_risks.append(local_risk)
            baseline_risks.append(baseline_risk)
            local_dirs.append(local_dir)
        diff_risk = (float(local_risk) - baseline_risk) if local_risk is not None else None
        diff_conf = (float(local_conf) - baseline_conf) if local_conf is not None else None

        rows.append(
            {
                "bucket": bucket,
                "local_direction": local_dir,
                "local_risk_delta": local_risk,
                "local_confidence": local_conf,
                "local_timestamp": row.get("local_timestamp"),
                "baseline_direction": str(base.get("direction_label", baseline_dir)),
                "baseline_risk_delta": baseline_risk,
                "baseline_confidence": baseline_conf,
                "diff_risk_delta": diff_risk,
                "diff_confidence": diff_conf,
                "direction_match": bool(local_dir == baseline_dir) if local_risk is not None else False,
                "effective_cash_impact": _compute_effective_cash_impact_for_bucket(row, overlay_cfg),
                "flags": "",
            }
        )

    direction_counts = {}
    for d in local_dirs:
        direction_counts[d] = direction_counts.get(d, 0) + 1
    dominant_ratio = 0.0
    if direction_counts and len(local_dirs) > 0:
        dominant_ratio = float(max(direction_counts.values()) / max(1, len(local_dirs)))
    std_local = float(np.std(local_risks)) if local_risks else 0.0
    overweight_ratio = float(sum(1 for d in local_dirs if d == "overweight") / max(1, len(local_dirs)))
    flag_uniformity = bool((std_local < 0.05) or (dominant_ratio > 0.80))

    flag_rate_sensitive_mismatch = False
    for b in ("utilities", "real_estate"):
        row = next((x for x in rows if x["bucket"] == b), None)
        if row and row.get("local_direction") == "overweight":
            base_dir = _direction_from_delta(float(_safe_float(row.get("baseline_risk_delta", 0.0), 0.0)))
            if base_dir == "underweight":
                flag_rate_sensitive_mismatch = True
                break

    row_consumer = next((x for x in rows if x["bucket"] == "consumer"), None)
    flag_consumer_mismatch = False
    if row_consumer and row_consumer.get("local_direction") == "overweight":
        base_dir = _direction_from_delta(float(_safe_float(row_consumer.get("baseline_risk_delta", 0.0), 0.0)))
        if base_dir == "underweight":
            flag_consumer_mismatch = True

    for row in rows:
        fl = []
        if flag_uniformity:
            fl.append("FLAG_UNIFORMITY")
        if row.get("bucket") in {"utilities", "real_estate"}:
            if row.get("local_direction") == "overweight" and _direction_from_delta(float(_safe_float(row.get("baseline_risk_delta", 0.0), 0.0))) == "underweight":
                fl.append("FLAG_RATE_SENSITIVE_MISMATCH")
        if row.get("bucket") == "consumer":
            if row.get("local_direction") == "overweight" and _direction_from_delta(float(_safe_float(row.get("baseline_risk_delta", 0.0), 0.0))) == "underweight":
                fl.append("FLAG_CONSUMER_MISMATCH")
        row["flags"] = ",".join(fl)

    correlation = None
    if len(local_risks) >= 2 and len(baseline_risks) == len(local_risks):
        try:
            local_arr = np.array(local_risks, dtype=float)
            base_arr = np.array(baseline_risks, dtype=float)
            # Guard zero-variance cases to avoid RuntimeWarning invalid value encountered in divide.
            if np.std(local_arr) > 1e-12 and np.std(base_arr) > 1e-12:
                correlation = float(np.corrcoef(local_arr, base_arr)[0, 1])
            else:
                correlation = None
        except Exception:
            correlation = None

    df = pd.DataFrame(rows)
    summary = {
        "read_status": read_meta.get("status"),
        "read_error": read_meta.get("error"),
        "collection": read_meta.get("collection"),
        "chroma_path": read_meta.get("chroma_path"),
        "available_buckets": int(sum(1 for r in rows if r.get("local_risk_delta") is not None)),
        "missing_buckets": int(sum(1 for r in rows if r.get("local_risk_delta") is None)),
        "std_local_risk_delta": std_local,
        "dominant_direction_ratio": dominant_ratio,
        "pct_overweight": overweight_ratio,
        "correlation_local_vs_baseline": correlation,
        "FLAG_UNIFORMITY": flag_uniformity,
        "FLAG_RATE_SENSITIVE_MISMATCH": flag_rate_sensitive_mismatch,
        "FLAG_CONSUMER_MISMATCH": flag_consumer_mismatch,
        "overlay_cfg": {
            "mode": str(overlay_cfg.get("mode", "risk_only")),
            "alpha": float(_safe_float(overlay_cfg.get("alpha", 0.08), 0.08)),
            "min_confidence": float(_safe_float(overlay_cfg.get("min_confidence", 0.55), 0.55)),
            "max_abs_delta": float(_safe_float(overlay_cfg.get("max_abs_delta", 0.10), 0.10)),
            "max_age_hours": float(_safe_float(overlay_cfg.get("max_age_hours", 48), 48)),
        },
    }
    return {"df": df, "summary": summary}


def build_industry_membership(config):
    """Build validated L2/L3 membership maps from config."""
    taxonomy = config.get("industry_taxonomy", {}) if isinstance(config, dict) else {}
    l2_list = taxonomy.get("L2", []) if isinstance(taxonomy, dict) else []
    l3_map = taxonomy.get("L3", {}) if isinstance(taxonomy, dict) else {}
    ticker_tags = config.get("ticker_tags", {}) if isinstance(config, dict) else {}

    if not isinstance(l2_list, list):
        raise ValueError("industry_taxonomy.L2 must be a list")
    if not isinstance(l3_map, dict):
        raise ValueError("industry_taxonomy.L3 must be a dict")
    if not isinstance(ticker_tags, dict):
        raise ValueError("ticker_tags must be a dict")

    l2_clean = [str(x).strip() for x in l2_list if str(x).strip()]
    if not (8 <= len(l2_clean) <= 10):
        raise ValueError(f"L2 count must be within 8-10, got {len(l2_clean)}")
    l2_set = set(l2_clean)

    l3_to_parent = {}
    for l2, l3_values in l3_map.items():
        l2_name = str(l2).strip()
        if l2_name not in l2_set:
            raise ValueError(f"L3 parent '{l2_name}' is not declared in L2")
        if not isinstance(l3_values, list):
            raise ValueError(f"industry_taxonomy.L3.{l2_name} must be a list")
        for l3 in l3_values:
            l3_name = str(l3).strip()
            if not l3_name:
                continue
            if l3_name in l3_to_parent and l3_to_parent[l3_name] != l2_name:
                raise ValueError(f"L3 tag '{l3_name}' is mapped to multiple L2 parents")
            l3_to_parent[l3_name] = l2_name

    l2_to_tickers = {l2: set() for l2 in l2_clean}
    l3_to_tickers = {}
    ticker_to_tags = {}

    for raw_ticker, raw_tags in ticker_tags.items():
        ticker = str(raw_ticker).strip().upper()
        if not ticker:
            continue
        if not isinstance(raw_tags, dict):
            raise ValueError(f"ticker_tags.{ticker} must be an object")

        l2_tags = [str(x).strip() for x in raw_tags.get("L2", []) if str(x).strip()]
        l3_tags = [str(x).strip() for x in raw_tags.get("L3", []) if str(x).strip()]
        keywords = [str(x).strip() for x in raw_tags.get("keywords", []) if str(x).strip()]

        invalid_l2 = [x for x in l2_tags if x not in l2_set]
        if invalid_l2:
            raise ValueError(f"ticker {ticker} references undefined L2 tags: {invalid_l2}")

        invalid_l3 = [x for x in l3_tags if x not in l3_to_parent]
        if invalid_l3:
            raise ValueError(f"ticker {ticker} references undefined L3 tags: {invalid_l3}")

        for l3 in l3_tags:
            parent_l2 = l3_to_parent.get(l3)
            if parent_l2 and parent_l2 not in l2_tags:
                raise ValueError(f"ticker {ticker} has L3 '{l3}' but missing parent L2 '{parent_l2}'")

        dedup_l2 = sorted(set(l2_tags))
        dedup_l3 = sorted(set(l3_tags))
        dedup_keywords = sorted(set(keywords))

        ticker_to_tags[ticker] = {
            "L2": dedup_l2,
            "L3": dedup_l3,
            "keywords": dedup_keywords
        }

        for l2 in dedup_l2:
            l2_to_tickers[l2].add(ticker)
        for l3 in dedup_l3:
            parent_l2 = l3_to_parent[l3]
            key = (parent_l2, l3)
            l3_to_tickers.setdefault(key, set()).add(ticker)

    weak_l2 = []
    for l2, members in l2_to_tickers.items():
        if l2 == "cash_equivalent":
            continue
        if len(members) < 3:
            weak_l2.append((l2, len(members)))
    if weak_l2:
        weak_desc = ", ".join([f"{name}={count}" for name, count in weak_l2])
        raise ValueError(f"L2 ticker count below minimum (3): {weak_desc}")

    return l2_to_tickers, l3_to_tickers, ticker_to_tags


def match_news_to_tags(news_item, ticker_to_tags):
    """Match a news item against ticker tags for future routing."""
    item = news_item if isinstance(news_item, dict) else {}
    title = str(item.get("title", ""))
    summary = str(item.get("summary", ""))
    text = f"{title} {summary}".lower()

    matched_tickers = set()
    matched_l2 = set()
    matched_l3 = set()
    matched_keywords = set()

    for ticker, tags in (ticker_to_tags or {}).items():
        tag_obj = tags if isinstance(tags, dict) else {}
        l2_tags = [str(x) for x in tag_obj.get("L2", [])]
        l3_tags = [str(x) for x in tag_obj.get("L3", [])]
        keywords = [str(x).lower() for x in tag_obj.get("keywords", [])]

        ticker_hit = ticker.lower() in text
        keyword_hits = [kw for kw in keywords if kw and kw in text]
        if ticker_hit or keyword_hits:
            matched_tickers.add(ticker)
            matched_l2.update(l2_tags)
            matched_l3.update(l3_tags)
            matched_keywords.update(keyword_hits)

    return {
        "matched_tickers": sorted(matched_tickers),
        "matched_L2": sorted(matched_l2),
        "matched_L3": sorted(matched_l3),
        "matched_keywords": sorted(matched_keywords)
    }


def dump_industry_taxonomy_preview(config_path, output_path=None):
    """Load taxonomy config, validate, print summary, and dump preview JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    l2_to_tickers, l3_to_tickers, ticker_to_tags = build_industry_membership(cfg)

    if output_path is None:
        output_path = (
            cfg.get("reporting", {}).get("industry_taxonomy_preview_path")
            or cfg.get("taxonomy_preview_path")
            or "outputs/industry_taxonomy_preview.json"
        )

    l2_counts = {k: len(v) for k, v in l2_to_tickers.items()}
    l3_counts = {
        f"{l2}:{l3}": len(v)
        for (l2, l3), v in sorted(l3_to_tickers.items(), key=lambda x: (x[0][0], x[0][1]))
    }
    cross_industry = sorted(
        [t for t, tags in ticker_to_tags.items() if len(tags.get("L2", [])) > 1]
    )

    preview = {
        "generated_at": datetime.now().isoformat(),
        "config_path": config_path,
        "output_path": output_path,
        "industry_taxonomy": cfg.get("industry_taxonomy", {}),
        "ticker_tags": ticker_to_tags,
        "stats": {
            "l2_count": len(l2_to_tickers),
            "ticker_tag_count": len(ticker_to_tags),
            "l2_counts": l2_counts,
            "l3_counts": l3_counts,
            "cross_industry_tickers": cross_industry
        }
    }

    io_atomic_write_json(output_path, preview, indent=2)

    print(f"[TAXONOMY] L2 count: {len(l2_to_tickers)}")
    for l2 in sorted(l2_to_tickers.keys()):
        print(f"[TAXONOMY] {l2}: {len(l2_to_tickers[l2])} tickers")
    for ticker in sorted(ticker_to_tags.keys()):
        tags = ticker_to_tags.get(ticker, {})
        l2_tags = ",".join(tags.get("L2", []))
        l3_tags = ",".join(tags.get("L3", []))
        keywords = ",".join(tags.get("keywords", []))
        print(
            f"[TAXONOMY] ticker={ticker} L2=[{l2_tags}] "
            f"L3=[{l3_tags}] keywords=[{keywords}]"
        )
    if cross_industry:
        sample = ", ".join(cross_industry[:10])
        print(f"[TAXONOMY] Cross-industry tickers: {sample}")
    else:
        print("[TAXONOMY] Cross-industry tickers: none")
    print(f"[TAXONOMY_PREVIEW] Wrote: {output_path}")


def _run_taxonomy_cli_if_requested():
    """Run taxonomy dump CLI and return exit code or None when not requested."""
    if "--dump-industry-taxonomy" not in sys.argv:
        return None

    parser = argparse.ArgumentParser(description="Industry taxonomy preview dump")
    parser.add_argument("--dump-industry-taxonomy", action="store_true")
    parser.add_argument("--config", default="paper_config.json")
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    try:
        dump_industry_taxonomy_preview(args.config, output_path=args.output)
        return 0
    except Exception as e:
        print(f"[TAXONOMY] ERROR: {e}")
        return 1


def _run_industry_sanity_cli_if_requested():
    if "--industry-sanity-check" not in sys.argv:
        return None
    parser = argparse.ArgumentParser(description="Industry signals sanity check")
    parser.add_argument("--industry-sanity-check", action="store_true")
    parser.add_argument("--config", default="paper_config.json")
    parser.add_argument("--output", default="outputs/industry_sanity_check_latest.json")
    args, _ = parser.parse_known_args()
    try:
        report = build_industry_sanity_check_report(config_path=args.config)
        df = report.get("df", pd.DataFrame())
        summary = report.get("summary", {})
        print("[SANITY] Industry signal sanity check")
        print(
            f"[SANITY] status={summary.get('read_status')} available={summary.get('available_buckets')} "
            f"missing={summary.get('missing_buckets')} pct_overweight={_safe_float(summary.get('pct_overweight', 0.0), 0.0):.2f} "
            f"std={_safe_float(summary.get('std_local_risk_delta', 0.0), 0.0):.4f} "
            f"corr={summary.get('correlation_local_vs_baseline')}"
        )
        print(
            f"[SANITY] flags uniformity={summary.get('FLAG_UNIFORMITY')} "
            f"rate_sensitive_mismatch={summary.get('FLAG_RATE_SENSITIVE_MISMATCH')} "
            f"consumer_mismatch={summary.get('FLAG_CONSUMER_MISMATCH')}"
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [
                "bucket",
                "local_direction",
                "local_risk_delta",
                "local_confidence",
                "local_timestamp",
                "baseline_direction",
                "baseline_risk_delta",
                "baseline_confidence",
                "diff_risk_delta",
                "diff_confidence",
                "direction_match",
                "effective_cash_impact",
                "flags",
            ]
            safe_cols = [c for c in cols if c in df.columns]
            print(df[safe_cols].to_string(index=False))
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "rows": df.to_dict("records") if isinstance(df, pd.DataFrame) else [],
        }
        io_atomic_write_json(args.output, payload, indent=2)
        print(f"[SANITY] wrote: {args.output}")
        return 0
    except Exception as e:
        print(f"[SANITY] ERROR: {e}")
        return 1


def _run_industry_runtime_once_cli_if_requested():
    if "--run-industry-runtime-once" not in sys.argv:
        return None
    parser = argparse.ArgumentParser(description="Run industry runtime pipeline once and exit")
    parser.add_argument("--run-industry-runtime-once", action="store_true")
    parser.add_argument("--config", default="paper_config.json")
    args, _ = parser.parse_known_args()
    try:
        result = run_industry_news_pipeline_runtime(config_path=args.config)
        print(f"[INDUSTRY_RUNTIME] status={result.get('status')}")
        latest_map, rows, meta = _load_latest_industry_signals_per_bucket(config_path=args.config)
        print(
            f"[INDUSTRY_RUNTIME] latest_read_status={meta.get('status')} "
            f"collection={meta.get('collection')} count={meta.get('count')}"
        )
        payload = {
            "runtime_result": result,
            "latest_rows": rows,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(f"[INDUSTRY_RUNTIME] ERROR: {e}")
        return 1


def _build_industry_write_id(signal_obj):
    if not isinstance(signal_obj, dict):
        return None
    l2 = str(signal_obj.get("L2", "unknown"))
    asof = str(signal_obj.get("asof_utc", datetime.now(timezone.utc).isoformat()))
    date_key = asof[:10]
    # Mirror write_industry_signals_to_chroma hash input (which does not include debug-only status).
    payload_obj = dict(signal_obj)
    payload_obj.pop("status", None)
    payload = json.dumps(payload_obj, ensure_ascii=False)
    item_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"industry::{l2}::{date_key}::{item_hash}"


def _safe_load_json_text(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _optional_float(value):
    try:
        if value is None:
            return None
        v = float(value)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _read_back_industry_signal(collection_name, chroma_path, bucket, write_timestamp_used, write_id=None):
    fallback = {
        "id": write_id,
        "bucket": bucket,
        "timestamp": write_timestamp_used,
        "status": "not_found",
        "direction": None,
        "risk_delta": None,
        "confidence": None,
        "metadata": {},
        "document_obj": None,
    }
    if not CHROMADB_IMPORTED:
        fallback["status"] = "chroma_unavailable"
        return fallback
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_or_create_collection(name=collection_name)
        ids = []
        metadatas = []
        documents = []
        # Chroma where parser in some versions accepts only single-field equality.
        # Query by L2 first, then filter timestamp client-side.
        by_l2 = coll.get(
            where={"L2": str(bucket)},
            include=["metadatas", "documents"],
            limit=200,
        )
        by_l2_ids = by_l2.get("ids", []) if isinstance(by_l2, dict) else []
        by_l2_metas = by_l2.get("metadatas", []) if isinstance(by_l2, dict) else []
        by_l2_docs = by_l2.get("documents", []) if isinstance(by_l2, dict) else []
        target_ts = str(write_timestamp_used)
        picked_idx = None
        for idx, md in enumerate(by_l2_metas):
            if isinstance(md, dict) and str(md.get("timestamp", "")) == target_ts:
                picked_idx = idx
                break
        if picked_idx is not None:
            ids = [by_l2_ids[picked_idx]] if picked_idx < len(by_l2_ids) else []
            metadatas = [by_l2_metas[picked_idx]]
            documents = [by_l2_docs[picked_idx]] if picked_idx < len(by_l2_docs) else []
        elif write_id:
            by_id = coll.get(ids=[str(write_id)], include=["metadatas", "documents"])
            ids = by_id.get("ids", []) if isinstance(by_id, dict) else []
            metadatas = by_id.get("metadatas", []) if isinstance(by_id, dict) else []
            documents = by_id.get("documents", []) if isinstance(by_id, dict) else []

        if not ids:
            return fallback
        row_id = ids[0]
        md = metadatas[0] if metadatas else {}
        doc_obj = _safe_load_json_text(documents[0] if documents else None)
        risk_delta = None
        confidence = None
        direction = None
        status = "ok"
        if isinstance(md, dict):
            risk_delta = _optional_float(md.get("risk_delta", None))
            confidence = _optional_float(md.get("confidence", None))
            direction = md.get("direction")
            status = str(md.get("status", status))
        if isinstance(doc_obj, dict):
            if "risk_delta" in doc_obj:
                risk_delta = _optional_float(doc_obj.get("risk_delta", risk_delta))
            if "confidence" in doc_obj:
                confidence = _optional_float(doc_obj.get("confidence", confidence))
            direction = doc_obj.get("direction", direction)
            status = str(doc_obj.get("status", status))
        return {
            "id": row_id,
            "bucket": bucket,
            "timestamp": write_timestamp_used,
            "status": status,
            "direction": direction,
            "risk_delta": risk_delta,
            "confidence": confidence,
            "metadata": md if isinstance(md, dict) else {},
            "document_obj": doc_obj,
        }
    except Exception as e:
        fallback["status"] = "read_error"
        fallback["error"] = str(e)
        return fallback


def _print_industry_runtime_debug_summary(debug_entries):
    if not isinstance(debug_entries, list) or not debug_entries:
        print("[INDUSTRY_RUNTIME_DEBUG] no bucket entries")
        return
    print("\n[INDUSTRY_RUNTIME_DEBUG] Summary")
    print(
        f"{'bucket':<22} {'items':>5} {'llm':>5} "
        f"{'final(direction,delta,conf,status)':<44} {'read_back(status)':<18} {'neutral_cap':<12}"
    )
    print("-" * 120)
    for row in debug_entries:
        bucket = str(row.get("bucket", ""))[:22]
        items = int(_safe_int(row.get("bucket_items_count", 0), 0))
        llm_called = "Y" if bool(row.get("llm_called", False)) else "N"
        final = row.get("final_to_write_obj", {}) if isinstance(row.get("final_to_write_obj"), dict) else {}
        read_back = row.get("read_back_obj", {}) if isinstance(row.get("read_back_obj"), dict) else {}
        f_dir = str(final.get("direction", "n/a"))
        f_conf = _optional_float(final.get("confidence", None))
        f_delta = _optional_float(final.get("risk_delta", None))
        f_status = str(final.get("status", "n/a"))
        rb_status = str(read_back.get("status", "n/a"))
        neutral_cap = bool(row.get("neutral_confidence_capped", False))
        f_txt = f"{f_dir},{f_delta!s},{f_conf!s},{f_status}"[:44]
        print(f"{bucket:<22} {items:>5} {llm_called:>5} {f_txt:<44} {rb_status:<18} {str(neutral_cap):<12}")


def run_industry_runtime_once_debug(config_path="paper_config.json", output_path="outputs/industry_runtime_debug_latest.json"):
    try:
        cfg = _load_json_config(config_path)
    except Exception as e:
        payload = {
            "status": "config_error",
            "error": str(e),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "bucket_debug": [],
        }
        io_atomic_write_json(output_path, payload, indent=2)
        return payload

    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    if not bool(overlay_cfg.get("enabled", False)):
        payload = {
            "status": "disabled",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "bucket_debug": [],
            "news_overlay_enabled": False,
        }
        io_atomic_write_json(output_path, payload, indent=2)
        return payload

    portfolio_tickers = _get_runtime_portfolio_tickers(
        snapshot_path=cfg.get("reporting", {}).get("snapshot_live_path", "outputs/snapshot_live.json")
    )
    candidate_tickers = _get_runtime_candidate_tickers(
        cfg,
        limit=_safe_int(cfg.get("news_sources", {}).get("max_candidate_tickers", 20), 20),
    )
    debug_trace = {"bucket_entries": {}}
    result = run_industry_news_pipeline(
        cfg,
        portfolio_tickers=portfolio_tickers,
        candidate_tickers=candidate_tickers,
        debug_trace=debug_trace,
    )

    collection_name = str(overlay_cfg.get("industry_collection", "industry_signals"))
    chroma_path = str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    entries = []
    bucket_entries = debug_trace.get("bucket_entries", {}) if isinstance(debug_trace, dict) else {}
    write_rows = result.get("write_info", {}).get("rows", [])
    write_row_by_bucket = {}
    if isinstance(write_rows, list):
        for wr in write_rows:
            if isinstance(wr, dict):
                write_row_by_bucket[str(wr.get("L2", "")).strip()] = wr
    for bucket in sorted(bucket_entries.keys()):
        row = dict(bucket_entries.get(bucket, {}))
        final_obj = row.get("final_to_write_obj", {})
        if not isinstance(final_obj, dict):
            final_obj = {}
        write_timestamp_used = str(final_obj.get("asof_utc", row.get("asof_utc", datetime.now(timezone.utc).isoformat())))
        write_id = _build_industry_write_id(final_obj) if final_obj else None
        read_back_obj = _read_back_industry_signal(
            collection_name=collection_name,
            chroma_path=chroma_path,
            bucket=bucket,
            write_timestamp_used=write_timestamp_used,
            write_id=write_id,
        )
        row["write_timestamp_used"] = write_timestamp_used
        row["write_id"] = write_id
        row["read_back_obj"] = read_back_obj
        wr = write_row_by_bucket.get(str(bucket), {})
        row["status_written_to_chroma"] = (
            str(wr.get("status", "")) if isinstance(wr, dict) and str(wr.get("status", "")).strip() else None
        )
        row["neutral_confidence_capped"] = bool(final_obj.get("_neutral_confidence_capped", False))
        row["bull_score"] = _safe_int(final_obj.get("bull_score", 0), 0)
        row["bear_score"] = _safe_int(final_obj.get("bear_score", 0), 0)
        row["net_score"] = _safe_int(final_obj.get("net_score", 0), 0)
        row["postprocess_overrode_direction"] = bool(final_obj.get("postprocess_overrode_direction", False))
        row["postprocess_overrode_risk_delta"] = bool(final_obj.get("postprocess_overrode_risk_delta", False))
        row["postprocess_downgrade_reason"] = str(final_obj.get("postprocess_downgrade_reason", ""))
        row["postprocess_kept_direction"] = bool(final_obj.get("postprocess_kept_direction", False))
        labels_head = final_obj.get("labels_head", [])
        row["labels_head"] = [dict(x) for x in labels_head[:4] if isinstance(x, dict)] if isinstance(labels_head, list) else []
        row["score_mapping_reason"] = str(final_obj.get("score_mapping_reason", ""))
        row["macro_prior_applied"] = bool(final_obj.get("macro_prior_applied", False))
        row["macro_prior_reasons"] = list(final_obj.get("macro_prior_reasons", [])) if isinstance(final_obj.get("macro_prior_reasons", []), list) else []
        row["macro_prior_strength"] = int(_safe_int(final_obj.get("macro_prior_strength", 0), 0))
        row["macro_risk_off_score"] = int(_safe_int(final_obj.get("macro_risk_off_score", 0), 0))
        row["macro_prior_skipped_due_to_score"] = bool(final_obj.get("macro_prior_skipped_due_to_score", False))
        row["macro_prior_skipped_due_to_cooldown"] = bool(final_obj.get("macro_prior_skipped_due_to_cooldown", False))
        entries.append(row)

    status_match_all = True
    for row in entries:
        final_status = str((row.get("final_to_write_obj", {}) or {}).get("status", ""))
        rb_status = str((row.get("read_back_obj", {}) or {}).get("status", ""))
        if final_status != rb_status:
            status_match_all = False
            break

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "runtime_result_status": result.get("write_info", {}).get("written", None),
        "collection": collection_name,
        "chroma_path": chroma_path,
        "all_buckets_readback_status_match_final_status": bool(status_match_all),
        "bucket_debug": entries,
        "write_info": result.get("write_info", {}),
    }

    dir_counter = Counter()
    risk_values = set()
    conf_values = set()
    net_scores = []
    overrode_direction = 0
    overrode_risk = 0
    negative_risk_delta_count = 0
    for row in entries:
        final_obj = row.get("final_to_write_obj", {}) if isinstance(row.get("final_to_write_obj"), dict) else {}
        direction = str(final_obj.get("direction", "neutral")).strip().lower() or "neutral"
        dir_counter[direction] += 1
        rv = _optional_float(final_obj.get("risk_delta", None))
        cv = _optional_float(final_obj.get("confidence", None))
        nv = _optional_float(final_obj.get("net_score", None))
        if rv is not None:
            risk_values.add(float(rv))
            if float(rv) < -1e-12:
                negative_risk_delta_count += 1
        if cv is not None:
            conf_values.add(float(cv))
        if nv is not None:
            net_scores.append(float(nv))
        if bool(final_obj.get("postprocess_overrode_direction", False)):
            overrode_direction += 1
        if bool(final_obj.get("postprocess_overrode_risk_delta", False)):
            overrode_risk += 1
    net_min = float(np.min(net_scores)) if net_scores else 0.0
    net_mean = float(np.mean(net_scores)) if net_scores else 0.0
    net_max = float(np.max(net_scores)) if net_scores else 0.0
    payload["distribution_summary"] = {
        "direction_counts": dict(dir_counter),
        "underweight_count": int(dir_counter.get("underweight", 0)),
        "negative_risk_delta_count": int(negative_risk_delta_count),
        "risk_delta_unique": sorted(risk_values),
        "confidence_unique": sorted(conf_values),
        "net_score_min": net_min,
        "net_score_mean": net_mean,
        "net_score_max": net_max,
        "count_overrode_direction": int(overrode_direction),
        "count_overrode_risk_delta": int(overrode_risk),
    }

    io_atomic_write_json(output_path, payload, indent=2)
    _print_industry_runtime_debug_summary(entries)
    print(
        "[INDUSTRY_RUNTIME_DEBUG] direction_counts "
        f"overweight={dir_counter.get('overweight', 0)} "
        f"neutral={dir_counter.get('neutral', 0)} "
        f"underweight={dir_counter.get('underweight', 0)}"
    )
    print(f"[INDUSTRY_RUNTIME_DEBUG] negative_risk_delta_count={negative_risk_delta_count}")
    print(f"[INDUSTRY_RUNTIME_DEBUG] risk_delta_unique={sorted(risk_values)}")
    print(f"[INDUSTRY_RUNTIME_DEBUG] confidence_unique={sorted(conf_values)}")
    print(
        f"[INDUSTRY_RUNTIME_DEBUG] count_overrode_direction={overrode_direction} "
        f"count_overrode_risk_delta={overrode_risk}"
    )
    print(f"[INDUSTRY_RUNTIME_DEBUG] net_score_min_mean_max=({net_min:.2f},{net_mean:.2f},{net_max:.2f})")
    print(f"[INDUSTRY_RUNTIME_DEBUG] all_buckets_readback_status_match_final_status={status_match_all}")
    print(f"[INDUSTRY_RUNTIME_DEBUG] wrote: {output_path}")
    return payload


def _run_industry_runtime_once_debug_cli_if_requested():
    if "--run-industry-runtime-once-debug" not in sys.argv:
        return None
    parser = argparse.ArgumentParser(description="Run industry runtime pipeline once with debug trace and exit")
    parser.add_argument("--run-industry-runtime-once-debug", action="store_true")
    parser.add_argument("--config", default="paper_config.json")
    parser.add_argument("--output", default="outputs/industry_runtime_debug_latest.json")
    args, _ = parser.parse_known_args()
    try:
        run_industry_runtime_once_debug(config_path=args.config, output_path=args.output)
        return 0
    except Exception as e:
        print(f"[INDUSTRY_RUNTIME_DEBUG] ERROR: {e}")
        return 1


def _extract_parsed_confidence(parsed_obj):
    if not isinstance(parsed_obj, dict):
        return None
    if "confidence" in parsed_obj:
        try:
            return float(parsed_obj.get("confidence"))
        except Exception:
            pass
    signal_obj = parsed_obj.get("signal", {})
    if isinstance(signal_obj, dict) and "confidence" in signal_obj:
        try:
            return float(signal_obj.get("confidence"))
        except Exception:
            pass
    return None


def _run_debug_industry_one_bucket_cli_if_requested():
    if ("--debug-industry-one-bucket" not in sys.argv) and ("--run-industry-one-bucket-debug" not in sys.argv):
        return None

    parser = argparse.ArgumentParser(description="Debug one industry bucket end-to-end")
    parser.add_argument("--debug-industry-one-bucket", default=None, help="Target L2 bucket name (legacy flag)")
    parser.add_argument("--run-industry-one-bucket-debug", default=None, help="Target L2 bucket name (step8 flag)")
    parser.add_argument("--config", default="paper_config.json")
    parser.add_argument("--output", default="outputs/debug_industry_one_bucket.json")
    parser.add_argument("--max_evidence", type=int, default=4)
    parser.add_argument("--llm_timeout_seconds", type=int, default=120)
    args, _ = parser.parse_known_args()

    target_l2 = str(args.run_industry_one_bucket_debug or args.debug_industry_one_bucket or "").strip()
    if not target_l2:
        print("[DEBUG_BUCKET] ERROR: empty bucket")
        return 1

    try:
        cfg = _load_json_config(args.config)
        news_overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
        sources_cfg = _merge_cfg(_default_news_sources_cfg(), cfg.get("news_sources", {}))
        industry_taxonomy = cfg.get("industry_taxonomy", {})

        l2_to_tickers, _l3_to_tickers, ticker_to_tags = build_industry_membership(cfg)
        l2_list = list(l2_to_tickers.keys())

        portfolio_tickers = _get_runtime_portfolio_tickers(
            snapshot_path=cfg.get("reporting", {}).get("snapshot_live_path", "outputs/snapshot_live.json")
        )
        candidate_tickers = _get_runtime_candidate_tickers(
            cfg,
            limit=_safe_int(cfg.get("news_sources", {}).get("max_candidate_tickers", 20), 20),
        )
        context = {
            "portfolio_tickers": portfolio_tickers,
            "candidate_tickers": candidate_tickers,
            "industry_topics": l2_list,
        }

        all_items = []
        providers = []
        if bool(sources_cfg.get("market_rss_enabled", True)):
            providers.append(
                MarketRSSProvider(
                    sources_cfg.get("market_rss_feeds", {}),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                )
            )
        if bool(sources_cfg.get("ticker_rss_enabled", True)):
            providers.append(
                TickerYahooRSSProvider(
                    sources_cfg.get("ticker_rss_template", ""),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                    max_tickers=_safe_int(
                        max(
                            _safe_int(sources_cfg.get("max_portfolio_tickers", 20), 20),
                            _safe_int(sources_cfg.get("max_candidate_tickers", 20), 20),
                        ),
                        20,
                    ),
                )
            )
        if bool(sources_cfg.get("industry_rss_enabled", True)):
            providers.append(
                IndustryGoogleRSSProvider(
                    sources_cfg.get("industry_rss_template", ""),
                    sources_cfg.get("industry_topic_queries", {}),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                )
            )

        provider_errors = []
        for provider in providers:
            try:
                fetched = provider.fetch(context)
                if isinstance(fetched, list):
                    all_items.extend(fetched)
            except Exception as e:
                provider_errors.append({"provider": getattr(provider, "provider_name", "unknown"), "error": str(e)})

        max_age_hours = _safe_float(news_overlay_cfg.get("max_age_hours", 48), 48.0)
        age_filtered = []
        for item in all_items:
            row = dict(item)
            if not row.get("published_at"):
                row["published_at"] = datetime.now(timezone.utc).isoformat()
            if _within_hours(row.get("published_at"), max_age_hours):
                age_filtered.append(row)

        deduped = _dedup_news_sorted(age_filtered)
        mapped = map_news_items_to_taxonomy(
            deduped,
            ticker_to_tags=ticker_to_tags,
            industry_taxonomy=industry_taxonomy,
            industry_keyword_map=sources_cfg.get("industry_keyword_map", {}),
        )
        buckets = bucket_news_by_l2(
            mapped,
            l2_list=l2_list,
            max_per_l2=_safe_int(sources_cfg.get("max_per_l2", 8), 8),
            prefer_seed_primary=bool(sources_cfg.get("prefer_seed_primary", True)),
        )
        buckets = _apply_post_bucket_total_cap(buckets, sources_cfg.get("post_bucket_max_total"))
        bucket_items_all = buckets.get(target_l2, [])
        max_evidence = max(1, _safe_int(args.max_evidence, 4))
        bucket_items = list(bucket_items_all[:max_evidence]) if isinstance(bucket_items_all, list) else []

        seed_counter = Counter()
        matched_l2_counter = Counter()
        bucket_preview = []
        for item in bucket_items:
            seed = str(item.get("seed_l2", "")).strip()
            if seed:
                seed_counter[seed] += 1
            for l2 in item.get("matched_L2", []) if isinstance(item.get("matched_L2"), list) else []:
                matched_l2_counter[str(l2)] += 1
            if len(bucket_preview) < 10:
                bucket_preview.append(
                    {
                        "title": str(item.get("title", "")),
                        "source": str(item.get("source", "")),
                        "seed_l2": item.get("seed_l2"),
                        "matched_L2": item.get("matched_L2", []),
                    }
                )

        asof_utc = datetime.now(timezone.utc).isoformat()
        llm_model = str(
            cfg.get("news_overlay", {}).get(
                "llm_model",
                cfg.get("macro_integration", {}).get("llm_topic_model", LOCAL_MODEL),
            )
        )
        macro_context_for_prior = _build_macro_context_for_industry_pipeline(cfg, mapped)
        macro_prior_cfg = _get_macro_risk_off_prior_cfg(news_overlay_cfg)
        collection_name = str(news_overlay_cfg.get("industry_collection", "industry_signals"))
        chroma_path = str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))

        raw_text = ""
        parsed_obj = {}
        normalized_obj = _neutral_industry_signal(target_l2, asof_utc, bucket_items, reason="no_items")
        llm_error = None
        llm_time_seconds = 0.0
        llm_timeout_hit = False

        if bucket_items:
            started = time.time()
            normalized_obj, raw_text, llm_error, parsed_obj = _generate_industry_signal_with_llm(
                target_l2,
                bucket_items,
                llm_model,
                return_debug=True,
                macro_context=macro_context_for_prior,
                macro_risk_off_prior_cfg=macro_prior_cfg,
                collection_name=collection_name,
                chroma_path=chroma_path,
                llm_timeout_seconds=_safe_int(args.llm_timeout_seconds, 120),
            )
            llm_time_seconds = float(max(0.0, time.time() - started))
            llm_timeout_hit = bool(isinstance(normalized_obj, dict) and normalized_obj.get("llm_timeout_hit", False))
            if (not llm_timeout_hit) and isinstance(llm_error, str) and "timeout" in llm_error.lower():
                llm_timeout_hit = True

        write_info = {
            "written": 0,
            "collection": collection_name,
        }
        read_back_obj = {}
        if isinstance(normalized_obj, dict) and bucket_items:
            try:
                write_info = write_industry_signals_to_chroma(
                    [normalized_obj],
                    collection_name=collection_name,
                    chroma_path=chroma_path,
                )
                write_rows = write_info.get("rows", []) if isinstance(write_info, dict) else []
                wr = write_rows[0] if isinstance(write_rows, list) and write_rows else {}
                read_back_obj = _read_back_industry_signal(
                    collection_name=collection_name,
                    chroma_path=chroma_path,
                    bucket=target_l2,
                    write_timestamp_used=str((wr or {}).get("timestamp", normalized_obj.get("asof_utc", asof_utc))),
                    write_id=str((wr or {}).get("id", _build_industry_write_id(normalized_obj))),
                )
            except Exception as e:
                log_error(f"[ONE_BUCKET_STEP8] chroma write/readback failed: {e}")

        parsed_confidence_value = _extract_parsed_confidence(parsed_obj)
        normalized_confidence_value = _safe_float(normalized_obj.get("confidence", 0.0), 0.0)
        raw_has_confidence_field = bool(re.search(r'"confidence"\s*:', str(raw_text or ""), re.IGNORECASE))
        evidence_ids_used = [
            str(item.get("id", "")).strip()
            for item in bucket_items
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]

        output_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": args.config,
            "target_bucket": target_l2,
            "max_evidence": int(max_evidence),
            "llm_timeout_seconds": int(_safe_int(args.llm_timeout_seconds, 120)),
            "runtime_context": {
                "portfolio_tickers_count": len(portfolio_tickers),
                "candidate_tickers_count": len(candidate_tickers),
                "providers_enabled": [getattr(p, "provider_name", "unknown") for p in providers],
                "provider_errors": provider_errors,
                "raw_items_count": len(all_items),
                "age_filtered_count": len(age_filtered),
                "dedup_count": len(deduped),
                "bucket_items_count": len(bucket_items),
                "bucket_items_count_before_max_evidence": len(bucket_items_all) if isinstance(bucket_items_all, list) else 0,
                "evidence_ids_used": evidence_ids_used,
            },
            "bucket_mapping_debug": {
                "seed_l2_distribution": dict(seed_counter),
                "matched_L2_distribution": dict(matched_l2_counter),
                "items_preview_top10": bucket_preview,
            },
            "llm_debug": {
                "model": llm_model,
                "llm_error": llm_error,
                "llm_time_seconds": float(llm_time_seconds),
                "llm_timeout_hit": bool(llm_timeout_hit),
                "raw_text": raw_text,
                "raw_text_head_4000": str(raw_text or "")[:4000],
                "parsed_obj": parsed_obj,
                "normalized_obj": normalized_obj,
                "raw_has_confidence_field": raw_has_confidence_field,
                "parsed_confidence_value": parsed_confidence_value,
                "normalized_confidence_value": normalized_confidence_value,
            },
            "step8_summary": {
                "bucket_items_count": len(bucket_items),
                "evidence_ids_used": evidence_ids_used,
                "macro_risk_off_score": int(_safe_int(normalized_obj.get("macro_risk_off_score", 0), 0)) if isinstance(normalized_obj, dict) else 0,
                "macro_prior_applied": bool(normalized_obj.get("macro_prior_applied", False)) if isinstance(normalized_obj, dict) else False,
                "macro_prior_skipped_due_to_score": bool(normalized_obj.get("macro_prior_skipped_due_to_score", False)) if isinstance(normalized_obj, dict) else False,
                "macro_prior_skipped_due_to_cooldown": bool(normalized_obj.get("macro_prior_skipped_due_to_cooldown", False)) if isinstance(normalized_obj, dict) else False,
                "macro_prior_reasons": list(normalized_obj.get("macro_prior_reasons", [])) if isinstance(normalized_obj, dict) else [],
                "bull_score": int(_safe_int(normalized_obj.get("bull_score", 0), 0)) if isinstance(normalized_obj, dict) else 0,
                "bear_score": int(_safe_int(normalized_obj.get("bear_score", 0), 0)) if isinstance(normalized_obj, dict) else 0,
                "net_score": int(_safe_int(normalized_obj.get("net_score", 0), 0)) if isinstance(normalized_obj, dict) else 0,
                "final": {
                    "direction": str(normalized_obj.get("direction", "neutral")) if isinstance(normalized_obj, dict) else "neutral",
                    "risk_delta": float(_safe_float(normalized_obj.get("risk_delta", 0.0), 0.0)) if isinstance(normalized_obj, dict) else 0.0,
                    "confidence": float(_safe_float(normalized_obj.get("confidence", 0.0), 0.0)) if isinstance(normalized_obj, dict) else 0.0,
                    "status": str(normalized_obj.get("status", "error")) if isinstance(normalized_obj, dict) else "error",
                },
                "llm_time_seconds": float(llm_time_seconds),
                "llm_timeout_hit": bool(llm_timeout_hit),
            },
            "write_info": write_info,
            "read_back_obj": read_back_obj,
        }

        io_atomic_write_json(args.output, output_payload, indent=2)

        ss = output_payload.get("step8_summary", {})
        ff = ss.get("final", {}) if isinstance(ss.get("final"), dict) else {}
        print(
            f"[ONE_BUCKET_STEP8] bucket={target_l2} items={ss.get('bucket_items_count', 0)} "
            f"score={ss.get('macro_risk_off_score', 0)} reasons={ss.get('macro_prior_reasons', [])} "
            f"applied={ss.get('macro_prior_applied', False)} "
            f"skip_score={ss.get('macro_prior_skipped_due_to_score', False)} "
            f"skip_cooldown={ss.get('macro_prior_skipped_due_to_cooldown', False)} "
            f"bull={ss.get('bull_score', 0)} bear={ss.get('bear_score', 0)} net={ss.get('net_score', 0)} "
            f"final=({ff.get('risk_delta', 0.0)},{ff.get('confidence', 0.0)},{ff.get('direction', 'neutral')},{ff.get('status', 'error')}) "
            f"llm_time={ss.get('llm_time_seconds', 0.0):.2f}s timeout={ss.get('llm_timeout_hit', False)}"
        )
        print(f"[DEBUG_BUCKET] wrote: {args.output}")
        return 0
    except Exception as e:
        print(f"[DEBUG_BUCKET] ERROR: {e}")
        return 1


def _load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_news_overlay_cfg():
    return {
        "enabled": False,
        "industry_collection": "industry_signals",
        "max_age_hours": 48,
        "alpha": 0.08,
        "mode": "risk_only",
        "min_confidence": 0.55,
        "max_abs_delta": 0.10,
        "runtime_min_interval_seconds": 900,
        "macro_risk_off_prior": {
            "enabled": False,
            "strength": 1,
            "min_score": 2,
            "cooldown_minutes": 180,
            "buckets": [
                "utilities",
                "real_estate",
                "rates_and_gold",
                "broad_market_etf",
                "cash_equivalent",
            ],
            "mode": "add_bearish_label",
            "min_macro_confidence": 0.55,
        },
    }


def _default_news_sources_cfg():
    return {
        "market_rss_enabled": True,
        "ticker_rss_enabled": True,
        "industry_rss_enabled": True,
        "prefer_seed_primary": True,
        "timeout_seconds": 8,
        "retries": 1,
        "max_per_l2": 8,
        "max_total": 60,
        "post_bucket_max_total": None,
        "max_portfolio_tickers": 20,
        "max_candidate_tickers": 20,
        "market_rss_feeds": dict(RSS_FEEDS),
        "ticker_rss_template": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "industry_rss_template": "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "industry_topic_queries": {
            "technology": ["semiconductors", "cloud software", "internet advertising", "ai infrastructure"],
            "energy": ["oil market", "natural gas demand", "energy producers"],
            "healthcare": ["pharma pipeline", "biotech approvals", "medical devices"],
            "financials": ["banking sector", "payment networks", "insurance trends"],
            "industrials_defense": ["defense contracts", "industrial demand", "rail logistics"],
            "consumer": ["consumer spending", "retail sales", "restaurant traffic"],
            "utilities": ["regulated utility", "power grid", "renewables"],
            "broad_market_etf": ["equity index flows", "etf rotation", "market breadth"],
            "rates_and_gold": ["treasury yields", "gold demand", "rate expectations"],
            "cash_equivalent": ["cash allocation", "money market yields"],
        },
        "industry_keyword_map": {
            "energy": ["opec", "oil", "brent", "wti", "natural gas"],
            "technology": ["semiconductor", "chip", "ai", "cloud", "saas"],
            "healthcare": ["fda", "drug", "pharma", "biotech"],
            "financials": ["bank", "credit", "payment", "insurance", "yield curve"],
            "industrials_defense": ["defense", "aerospace", "rail", "logistics"],
            "consumer": ["retail", "consumer", "restaurant", "beverage"],
            "utilities": ["utility", "grid", "power", "renewable"],
            "rates_and_gold": ["treasury", "bond", "rate hike", "gold"],
            "broad_market_etf": ["index", "etf", "market breadth", "equity market"],
        },
    }


def _merge_cfg(defaults, override):
    merged = dict(defaults)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_iso_utc(dt_obj):
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _parse_feed_entry_time(entry):
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _fetch_feed_entries(url, timeout_seconds=8, retries=1):
    timeout_seconds = max(1, _safe_int(timeout_seconds, 8))
    retries = max(0, _safe_int(retries, 1))
    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; GlobalWatch/2.11; +https://localhost)"
                },
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                payload = resp.read()
            feed = feedparser.parse(payload)
            return getattr(feed, "entries", []) or []
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.2)
                continue
    if last_error:
        raise last_error
    return []


class NewsProvider:
    provider_name = "base"

    def fetch(self, context):
        return []


class MarketRSSProvider(NewsProvider):
    provider_name = "market_rss"

    def __init__(self, feed_map, timeout_seconds, retries):
        self.feed_map = feed_map if isinstance(feed_map, dict) else {}
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def fetch(self, context):
        out = []
        for source, url in self.feed_map.items():
            try:
                entries = _fetch_feed_entries(url, timeout_seconds=self.timeout_seconds, retries=self.retries)
            except Exception as e:
                log_error(f"[TAXONOMY_PREVIEW] market feed failed {source}: {e}")
                continue
            for entry in entries:
                title = str(getattr(entry, "title", "") or "").strip()
                link = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                if not title or not link:
                    continue
                published_dt = _parse_feed_entry_time(entry)
                out.append(
                    {
                        "source": str(source),
                        "published_at": _to_iso_utc(published_dt) if published_dt else None,
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "raw_text": "",
                    }
                )
        return out


class TickerYahooRSSProvider(NewsProvider):
    provider_name = "ticker_yahoo_rss"

    def __init__(self, template, timeout_seconds, retries, max_tickers=20):
        self.template = template
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.max_tickers = max_tickers

    def fetch(self, context):
        tickers = []
        for group_key in ("portfolio_tickers", "candidate_tickers"):
            for ticker in context.get(group_key, []):
                t = str(ticker).strip().upper()
                if t and t not in tickers and t != "CASH":
                    tickers.append(t)
        tickers = tickers[: max(1, _safe_int(self.max_tickers, 20))]

        out = []
        for ticker in tickers:
            url = str(self.template).format(ticker=urllib.parse.quote(ticker))
            try:
                entries = _fetch_feed_entries(url, timeout_seconds=self.timeout_seconds, retries=self.retries)
            except Exception as e:
                log_error(f"[TAXONOMY_PREVIEW] ticker feed failed {ticker}: {e}")
                continue
            for entry in entries[:6]:
                title = str(getattr(entry, "title", "") or "").strip()
                link = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                if not title or not link:
                    continue
                published_dt = _parse_feed_entry_time(entry)
                out.append(
                    {
                        "source": f"YahooRSS:{ticker}",
                        "published_at": _to_iso_utc(published_dt) if published_dt else None,
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "raw_text": "",
                        "seed_ticker": ticker,
                    }
                )
        return out


class IndustryGoogleRSSProvider(NewsProvider):
    provider_name = "industry_google_rss"

    def __init__(self, template, topic_queries, timeout_seconds, retries):
        self.template = template
        self.topic_queries = topic_queries if isinstance(topic_queries, dict) else {}
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def fetch(self, context):
        out = []
        requested_l2 = context.get("industry_topics", [])
        for l2 in requested_l2:
            queries = self.topic_queries.get(l2, [l2])
            if not isinstance(queries, list):
                queries = [str(queries)]
            for query_text in queries[:3]:
                query = urllib.parse.quote(str(query_text))
                url = str(self.template).format(query=query)
                try:
                    entries = _fetch_feed_entries(url, timeout_seconds=self.timeout_seconds, retries=self.retries)
                except Exception as e:
                    log_error(f"[TAXONOMY_PREVIEW] industry feed failed {l2}:{query_text}: {e}")
                    continue
                for entry in entries[:5]:
                    title = str(getattr(entry, "title", "") or "").strip()
                    link = str(getattr(entry, "link", "") or "").strip()
                    summary = str(getattr(entry, "summary", "") or "").strip()
                    if not title or not link:
                        continue
                    published_dt = _parse_feed_entry_time(entry)
                    out.append(
                        {
                            "source": f"GoogleNews:{l2}",
                            "published_at": _to_iso_utc(published_dt) if published_dt else None,
                            "title": title,
                            "summary": summary,
                            "url": link,
                            "raw_text": "",
                            "seed_l2": l2,
                            "seed_query": str(query_text),
                        }
                    )
        return out


def _stable_news_id(item):
    source = str(item.get("source", "")).strip().lower()
    title = str(item.get("title", "")).strip().lower()
    published = str(item.get("published_at", "")).strip().lower()
    url = str(item.get("url", "")).strip().lower()
    base = f"{source}|{title}|{published}|{url}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


def _parse_iso_or_none(value):
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except Exception:
        return None


def _within_hours(value, max_age_hours):
    dt_obj = _parse_iso_or_none(value)
    if dt_obj is None:
        return False
    age = datetime.now(timezone.utc) - dt_obj
    return age.total_seconds() <= max(0.0, float(max_age_hours)) * 3600.0


def _dedup_and_limit_news(items, max_total=60):
    dedup = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not title and not url:
            continue
        url_key = url.lower()
        title_key = normalize_title(title) if title else ""
        key = url_key or f"title::{title_key}"
        if key in dedup:
            continue
        item = dict(raw)
        item["id"] = _stable_news_id(item)
        dedup[key] = item
    merged = list(dedup.values())
    merged.sort(
        key=lambda x: _parse_iso_or_none(x.get("published_at")) or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return merged[: max(1, _safe_int(max_total, 60))]


def _dedup_news_sorted(items):
    """Deduplicate and sort by published_at desc without global truncation."""
    dedup = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not title and not url:
            continue
        url_key = url.lower()
        title_key = normalize_title(title) if title else ""
        key = url_key or f"title::{title_key}"
        if key in dedup:
            continue
        item = dict(raw)
        item["id"] = _stable_news_id(item)
        dedup[key] = item
    merged = list(dedup.values())
    merged.sort(
        key=lambda x: _parse_iso_or_none(x.get("published_at")) or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return merged


def _apply_post_bucket_total_cap(buckets, post_bucket_max_total):
    """Optional post-bucket cap with a one-item-per-nonempty-bucket floor."""
    try:
        cap = _safe_int(post_bucket_max_total, 0)
    except Exception:
        cap = 0
    if cap <= 0 or not isinstance(buckets, dict):
        return buckets

    bucket_keys = list(buckets.keys())
    nonempty_keys = [k for k in bucket_keys if isinstance(buckets.get(k), list) and len(buckets.get(k)) > 0]
    if not nonempty_keys:
        return buckets

    effective_cap = max(cap, len(nonempty_keys))
    base = {k: [] for k in bucket_keys}
    used = 0

    # First pass: keep one item for each non-empty bucket to avoid starvation.
    for k in nonempty_keys:
        if used >= effective_cap:
            break
        base[k].append(buckets[k][0])
        used += 1

    if used >= effective_cap:
        return base

    # Round-robin fill from remaining items.
    cursor = {k: 1 for k in nonempty_keys}
    progressed = True
    while used < effective_cap and progressed:
        progressed = False
        for k in nonempty_keys:
            idx = cursor.get(k, 1)
            rows = buckets.get(k, [])
            if idx < len(rows):
                base[k].append(rows[idx])
                cursor[k] = idx + 1
                used += 1
                progressed = True
                if used >= effective_cap:
                    break
    return base


def _build_l3_to_l2_map(industry_taxonomy):
    l3_map = {}
    if not isinstance(industry_taxonomy, dict):
        return l3_map
    l3_obj = industry_taxonomy.get("L3", {})
    if not isinstance(l3_obj, dict):
        return l3_map
    for l2, l3_values in l3_obj.items():
        if not isinstance(l3_values, list):
            continue
        for l3 in l3_values:
            l3_map[str(l3)] = str(l2)
    return l3_map


def map_news_items_to_taxonomy(news_items, ticker_to_tags, industry_taxonomy, industry_keyword_map):
    mapped = []
    l3_to_l2 = _build_l3_to_l2_map(industry_taxonomy)
    keyword_map = industry_keyword_map if isinstance(industry_keyword_map, dict) else {}
    for item in news_items:
        row = dict(item)
        match = match_news_to_tags(row, ticker_to_tags)
        matched_tickers = set(match.get("matched_tickers", []))
        matched_l2 = set(match.get("matched_L2", []))
        matched_l3 = set(match.get("matched_L3", []))
        matched_keywords = set(match.get("matched_keywords", []))

        seed_l2 = str(row.get("seed_l2", "")).strip()
        if seed_l2:
            row["primary_l2"] = seed_l2
            # Strongly retain the original industry intent from provider.
            matched_l2.add(seed_l2)
        else:
            row["primary_l2"] = None

        text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
        for l2, keywords in keyword_map.items():
            if not isinstance(keywords, list):
                continue
            for kw in keywords:
                key = str(kw).strip().lower()
                if key and key in text:
                    matched_l2.add(str(l2))
                    matched_keywords.add(key)

        for l3 in list(matched_l3):
            parent = l3_to_l2.get(str(l3))
            if parent:
                matched_l2.add(parent)

        row["matched_tickers"] = sorted(matched_tickers)
        row["matched_L2"] = sorted(matched_l2)
        row["matched_L3"] = sorted(matched_l3)
        row["matched_keywords"] = sorted(matched_keywords)
        mapped.append(row)
    return mapped


def bucket_news_by_l2(mapped_news, l2_list, max_per_l2=8, prefer_seed_primary=True):
    buckets = {str(l2): [] for l2 in l2_list}
    max_per = max(1, _safe_int(max_per_l2, 8))

    if not prefer_seed_primary:
        for item in mapped_news:
            l2_tags = item.get("matched_L2", [])
            if not isinstance(l2_tags, list):
                continue
            for l2 in l2_tags:
                key = str(l2)
                if key not in buckets:
                    continue
                if len(buckets[key]) >= max_per:
                    continue
                buckets[key].append(item)
        return buckets

    # Phase 1: route seed/primary items exclusively to their primary bucket.
    non_seed_items = []
    for item in mapped_news:
        primary = str(item.get("primary_l2") or item.get("seed_l2") or "").strip()
        if primary and primary in buckets:
            if len(buckets[primary]) < max_per:
                buckets[primary].append(item)
            continue
        non_seed_items.append(item)

    # Phase 2: only allow non-seed expansion into buckets that currently have no primary-seed coverage.
    seeded_covered = {k for k, rows in buckets.items() if len(rows) > 0}
    for item in non_seed_items:
        l2_tags = item.get("matched_L2", [])
        if not isinstance(l2_tags, list):
            continue
        for l2 in l2_tags:
            key = str(l2)
            if key not in buckets:
                continue
            if key in seeded_covered:
                continue
            if len(buckets[key]) >= max_per:
                continue
            buckets[key].append(item)
    return buckets


def _neutral_industry_signal(l2, asof_utc, evidence_items, reason="neutral"):
    sample_evidence = []
    for item in evidence_items[:8]:
        sample_evidence.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "title": item.get("title"),
                "url": item.get("url"),
            }
        )
    return {
        "asof_utc": asof_utc,
        "scope": "industry",
        "L2": l2,
        "L3_focus": [],
        "risk_delta": 0.0,
        "confidence": 0.3,
        "horizon": "1d",
        "top_drivers": [reason],
        "impacted_tickers": [],
        "evidence": sample_evidence,
        "notes_speculative": True,
    }


def _normalize_industry_signal(signal, l2, asof_utc, evidence_items):
    if not isinstance(signal, dict):
        signal = {}
    normalized = _neutral_industry_signal(l2, asof_utc, evidence_items, reason="fallback")
    normalized["L2"] = str(signal.get("L2") or signal.get("bucket") or l2)
    normalized["scope"] = "industry"
    normalized["L3_focus"] = [str(x) for x in signal.get("L3_focus", []) if str(x).strip()][:8]

    # Accept both legacy schema and batch schema.
    signal_block = signal.get("signal", {}) if isinstance(signal.get("signal"), dict) else {}
    direction = str(signal_block.get("direction", signal.get("direction", "neutral"))).strip().lower()
    strength = float(np.clip(_safe_float(signal_block.get("strength", signal.get("risk_delta", 0.0)), 0.0), -1.0, 1.0))
    confidence = float(
        np.clip(
            _safe_float(signal_block.get("confidence", signal.get("confidence", 0.0)), 0.0),
            0.0,
            1.0,
        )
    )
    if direction == "overweight":
        risk_delta = abs(strength)
    elif direction == "underweight":
        risk_delta = -abs(strength)
    elif direction == "neutral":
        risk_delta = 0.0
    else:
        risk_delta = strength

    # Prefer explicit risk_delta schema when provided.
    if "risk_delta" in signal:
        risk_delta = float(_safe_float(signal.get("risk_delta", risk_delta), risk_delta))
    normalized["risk_delta"] = float(np.clip(risk_delta, -0.30, 0.30))
    normalized["confidence"] = confidence

    expiry_hours = int(np.clip(_safe_int(signal.get("expiry_hours", 24), 24), 12, 72))
    if expiry_hours <= 36:
        horizon = "1d"
    elif expiry_hours <= 72:
        horizon = "1w"
    else:
        horizon = "1m"
    normalized["horizon"] = str(signal.get("horizon", horizon))
    normalized["expiry_hours"] = expiry_hours
    normalized["direction"] = direction if direction in {"overweight", "underweight", "neutral"} else "neutral"

    drivers = signal.get("drivers", signal.get("top_drivers", []))
    if isinstance(drivers, list):
        normalized["top_drivers"] = [str(x) for x in drivers if str(x).strip()][:5]
    impacted = []
    raw_impacted = signal.get("impacted_tickers", [])
    if isinstance(raw_impacted, list):
        for item in raw_impacted[:8]:
            if not isinstance(item, dict):
                continue
            impacted.append(
                {
                    "ticker": str(item.get("ticker", "")).upper(),
                    "direction": str(item.get("direction", "neutral")),
                    "magnitude": float(np.clip(_safe_float(item.get("magnitude", 0.0), 0.0), -1.0, 1.0)),
                    "reason": str(item.get("reason", ""))[:240],
                }
            )
    normalized["impacted_tickers"] = impacted
    evidence = []
    raw_evidence = signal.get("headline_evidence", signal.get("evidence", []))
    if isinstance(raw_evidence, list) and raw_evidence:
        for item in raw_evidence[:8]:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "why": item.get("why"),
                }
            )
    if not evidence:
        for item in evidence_items[:8]:
            evidence.append(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                }
            )
    normalized["evidence"] = evidence
    normalized["notes_speculative"] = bool(
        signal.get("notes_speculative", signal.get("confidence", 0.0) and confidence < 0.5)
    )
    normalized["mode"] = str(signal.get("mode", "BATCH"))
    normalized["notes"] = str(signal.get("notes", ""))[:240]
    counterpoints = signal.get("counterpoints", [])
    if isinstance(counterpoints, list):
        normalized["counterpoints"] = [str(x)[:180] for x in counterpoints if str(x).strip()][:2]
    else:
        normalized["counterpoints"] = []
    rs = str(signal.get("rate_sensitivity", "")).strip().lower()
    if rs not in {"low", "med", "high"}:
        rs = ""
    normalized["rate_sensitivity"] = rs
    normalized["asof_utc"] = str(signal.get("asof_utc", asof_utc))
    if normalized["confidence"] < 0.4:
        normalized["risk_delta"] = 0.0
        normalized["direction"] = "neutral"
    else:
        # Enforce directional consistency with risk_delta.
        if normalized["risk_delta"] > 1e-12:
            normalized["direction"] = "overweight"
        elif normalized["risk_delta"] < -1e-12:
            normalized["direction"] = "underweight"
        else:
            normalized["direction"] = "neutral"
    return normalized


def _build_industry_llm_prompt(l2, news_items, asof_utc):
    compact_rows = []
    for item in news_items[:6]:
        compact_rows.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "title": str(item.get("title", ""))[:180],
                "summary": str(item.get("summary", ""))[:200],
                "url": item.get("url"),
                "matched_tickers": item.get("matched_tickers", []),
                "matched_L2": item.get("matched_L2", []),
                "matched_L3": item.get("matched_L3", []),
            }
        )
    bucket_description = (
        f"Industry bucket '{l2}' for portfolio risk overlays. "
        "Label each evidence item for short-term (1d) directional impact."
    )
    evidence_ids = [str(x.get("id")) for x in compact_rows if isinstance(x, dict) and str(x.get("id", "")).strip()]
    prompt = f"""
You are GlobalWatch Batch Industry Evidence Labeler.

GOAL
- Label each evidence item for ONE industry bucket.
- Do NOT output final direction/risk_delta/confidence.
- Final scoring is computed by code.

MODE
- mode: BATCH
- bucket_name: {l2}
- bucket_description: {bucket_description}

INPUTS (already fetched; do NOT browse)
- bucket_news_items:
{json.dumps(compact_rows, ensure_ascii=False)}

- optional macro context:
{json.dumps({"asof_utc": asof_utc, "bucket": l2}, ensure_ascii=False)}

OUTPUT REQUIREMENTS (STRICT)
- Output MUST be valid JSON only. No markdown.
- Keep it compact: <= 400 output tokens.
- Do NOT include chain-of-thought. Use short rationale.

RULES
- For each input evidence id, produce exactly one label item.
- Allowed sentiment: bullish | bearish | neutral
- Allowed strength:
  * bullish/bearish: integer 1..3
  * neutral: must be 1
- rationale max 140 chars and must be bucket-specific.
- If evidence is unrelated to bucket: sentiment=neutral, rationale="not bucket-specific"
- Avoid generic phrases like "across sectors mixed signals" in rationale.
- Labels must cover all ids exactly once from:
{json.dumps(evidence_ids, ensure_ascii=False)}

JSON SCHEMA (MUST FOLLOW)
{{
  "labels": [
    {{
      "id": "<evidence_id>",
      "sentiment": "bullish" | "bearish" | "neutral",
      "strength": 1 | 2 | 3,
      "rationale": "<=140 chars>"
    }}
  ],
  "notes": "<=200 chars>"
}}

NOW PRODUCE THE JSON.
"""
    return prompt


def _postprocess_industry_signals(signals, news_overlay_cfg):
    rows = [dict(x) for x in (signals or []) if isinstance(x, dict)]
    if not rows:
        return rows, {
            "status": "empty",
            "uniformity_triggered": False,
            "overweight_ratio": 0.0,
            "mean_risk_delta_before": 0.0,
            "mean_risk_delta_after": 0.0,
            "std_risk_delta_before": 0.0,
            "std_risk_delta_after": 0.0,
            "low_conf_neutralized": 0,
        }

    min_conf = float(np.clip(_safe_float(news_overlay_cfg.get("min_confidence", 0.55), 0.55), 0.0, 1.0))
    low_conf_neutralized = 0
    overrode_direction_count = 0
    overrode_risk_count = 0
    neutral_capped_count = 0

    def _risk_from_net(net_score):
        n = int(net_score)
        if n >= 4:
            return 0.20
        if n == 3:
            return 0.10
        if n == 2:
            return 0.05
        if n in (-1, 0, 1):
            return 0.0
        if n == -2:
            return -0.05
        if n == -3:
            return -0.10
        return -0.20

    for row in rows:
        pre_direction = _normalize_direction_text(row.get("direction"))
        pre_risk = float(np.clip(_safe_float(row.get("risk_delta", 0.0), 0.0), -0.30, 0.30))
        row["_pre_post_direction"] = pre_direction
        row["_pre_post_risk_delta"] = pre_risk
        conf = float(np.clip(_safe_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0))
        risk = float(np.clip(_safe_float(row.get("risk_delta", 0.0), 0.0), -0.30, 0.30))
        direction = _normalize_direction_text(row.get("direction"))
        evidence_count = 0
        row["status"] = str(row.get("status", "ok")).strip() or "ok"
        row["_neutral_confidence_capped"] = bool(row.get("_neutral_confidence_capped", False))
        row["postprocess_overrode_direction"] = False
        row["postprocess_overrode_risk_delta"] = False
        row["postprocess_downgrade_reason"] = ""
        row["postprocess_kept_direction"] = False
        row["postprocess_downgraded_to_neutral"] = False

        bull = int(_safe_int(row.get("bull_score", 0), 0))
        bear = int(_safe_int(row.get("bear_score", 0), 0))
        net = int(_safe_int(row.get("net_score", bull - bear), bull - bear))
        total_non_neutral = max(0, bull + bear)
        abs_net = abs(net)
        if isinstance(row.get("evidence"), list):
            evidence_count = len([x for x in row.get("evidence", []) if isinstance(x, dict)])

        strong_keep = (
            row.get("status") == "ok"
            and evidence_count > 0
            and abs_net >= 2
            and total_non_neutral >= 2
        )
        allow_downgrade = (
            row.get("status") != "ok"
            or evidence_count == 0
            or total_non_neutral == 0
            or abs_net <= 1
        )

        if strong_keep:
            # Keep deterministic direction/risk for strong signals.
            if abs(risk) < 1e-9:
                risk = _risk_from_net(net)
            if direction == "overweight":
                risk = abs(risk)
            elif direction == "underweight":
                risk = -abs(risk)
            row["risk_delta"] = float(np.clip(risk, -0.30, 0.30))
            row["direction"] = _direction_from_delta(row["risk_delta"])
            row["postprocess_kept_direction"] = True
            note = str(row.get("notes", "")).strip()
            keep_tag = "postprocess_kept_direction=True"
            if keep_tag not in note:
                row["notes"] = f"{note}; {keep_tag}".strip("; ").strip()
        elif allow_downgrade:
            reason = ""
            if row.get("status") != "ok":
                reason = "status_not_ok"
            elif evidence_count == 0:
                reason = "evidence_items_count_0"
            elif total_non_neutral == 0:
                reason = "bull_plus_bear_zero"
            elif abs_net <= 1:
                reason = "weak_net_score"
            row["risk_delta"] = 0.0
            row["direction"] = "neutral"
            row["postprocess_downgraded_to_neutral"] = True
            row["postprocess_downgrade_reason"] = reason
            low_conf_neutralized += 1
        else:
            # Keep direction but enforce sign coherence.
            if direction == "overweight":
                risk = abs(risk)
            elif direction == "underweight":
                risk = -abs(risk)
            row["risk_delta"] = float(np.clip(risk, -0.30, 0.30))
            row["direction"] = _direction_from_delta(row["risk_delta"])

        if conf < min_conf and strong_keep:
            # For strong net signals, bypass neutralization and keep direction.
            pass

    before_vals = np.array([float(np.clip(_safe_float(r.get("risk_delta", 0.0), 0.0), -0.30, 0.30)) for r in rows], dtype=float)
    overweight_ratio = float(np.mean(np.array([1.0 if _normalize_direction_text(r.get("direction")) == "overweight" else 0.0 for r in rows])))
    mean_before = float(np.mean(before_vals)) if len(before_vals) > 0 else 0.0
    std_before = float(np.std(before_vals)) if len(before_vals) > 0 else 0.0

    uniformity_triggered = bool(overweight_ratio > 0.80 and mean_before > 0.10)
    # Keep deterministic direction for strong signals; no uniformity recenter override here.
    for row in rows:
        conf = float(np.clip(_safe_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0))
        risk = float(np.clip(_safe_float(row.get("risk_delta", 0.0), 0.0), -0.30, 0.30))
        direction = _normalize_direction_text(row.get("direction"))
        should_cap = (direction == "neutral") or (abs(risk) < 1e-9)
        high_conf_zero = conf >= 0.65 and abs(risk) < 1e-9
        if should_cap or high_conf_zero:
            new_conf = min(conf, 0.45)
            if new_conf < conf - 1e-12:
                neutral_capped_count += 1
                row["_neutral_confidence_capped"] = True
                row["confidence"] = float(new_conf)
                note_tag = "confidence_capped_for_neutral"
                note = str(row.get("notes", "")).strip()
                if note_tag not in note:
                    row["notes"] = f"{note}; {note_tag}".strip("; ").strip()
                drivers = row.get("top_drivers", [])
                if not isinstance(drivers, list):
                    drivers = []
                if note_tag not in [str(x) for x in drivers]:
                    drivers = list(drivers) + [note_tag]
                row["top_drivers"] = [str(x) for x in drivers if str(x).strip()][:5]
            else:
                row["confidence"] = float(new_conf)

    allowed_risk = np.array([-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30], dtype=float)
    allowed_conf = np.array([0.35, 0.45, 0.55, 0.65, 0.75], dtype=float)

    for row in rows:
        pre_direction = _normalize_direction_text(row.get("_pre_post_direction", row.get("direction")))
        pre_risk = float(np.clip(_safe_float(row.get("_pre_post_risk_delta", row.get("risk_delta", 0.0)), 0.0), -0.30, 0.30))
        risk = float(np.clip(_safe_float(row.get("risk_delta", 0.0), 0.0), -0.30, 0.30))
        conf = float(np.clip(_safe_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0))
        snapped_risk = float(allowed_risk[int(np.argmin(np.abs(allowed_risk - risk)))])
        snapped_conf = float(allowed_conf[int(np.argmin(np.abs(allowed_conf - conf)))])
        row["risk_delta"] = snapped_risk
        row["confidence"] = snapped_conf
        row["direction"] = _direction_from_delta(snapped_risk)
        row["postprocess_overrode_direction"] = bool(row["direction"] != pre_direction)
        row["postprocess_overrode_risk_delta"] = bool(abs(snapped_risk - pre_risk) > 1e-9)
        if row["postprocess_overrode_direction"]:
            overrode_direction_count += 1
        if row["postprocess_overrode_risk_delta"]:
            overrode_risk_count += 1

    after_vals = np.array([float(np.clip(_safe_float(r.get("risk_delta", 0.0), 0.0), -0.30, 0.30)) for r in rows], dtype=float)
    return rows, {
        "status": "ok",
        "uniformity_triggered": uniformity_triggered,
        "overweight_ratio": overweight_ratio,
        "mean_risk_delta_before": mean_before,
        "mean_risk_delta_after": float(np.mean(after_vals)) if len(after_vals) > 0 else 0.0,
        "std_risk_delta_before": std_before,
        "std_risk_delta_after": float(np.std(after_vals)) if len(after_vals) > 0 else 0.0,
        "low_conf_neutralized": int(low_conf_neutralized),
        "neutral_confidence_capped": int(neutral_capped_count),
        "count_overrode_direction": int(overrode_direction_count),
        "count_overrode_risk_delta": int(overrode_risk_count),
    }


def _get_bucket_rate_sensitivity(bucket):
    key = str(bucket or "").strip().lower()
    mapping = {
        "utilities": "high",
        "real_estate": "high",
        "rates_and_gold": "high",
        "broad_market_etf": "med",
        "cash_equivalent": "high",
    }
    return mapping.get(key, "low")


def _get_macro_risk_off_prior_cfg(news_overlay_cfg):
    defaults = {
        "enabled": False,
        "strength": 1,
        "min_score": 2,
        "cooldown_minutes": 180,
        "buckets": [
            "utilities",
            "real_estate",
            "rates_and_gold",
            "broad_market_etf",
            "cash_equivalent",
        ],
        "mode": "add_bearish_label",
        "min_macro_confidence": 0.55,
    }
    cfg = dict(defaults)
    if isinstance(news_overlay_cfg, dict):
        raw = news_overlay_cfg.get("macro_risk_off_prior", {})
        if isinstance(raw, dict):
            cfg.update(raw)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["strength"] = int(np.clip(_safe_int(cfg.get("strength", 1), 1), 1, 3))
    cfg["min_score"] = int(np.clip(_safe_int(cfg.get("min_score", 2), 2), 0, 10))
    cfg["cooldown_minutes"] = int(np.clip(_safe_int(cfg.get("cooldown_minutes", 180), 180), 0, 1440 * 7))
    raw_buckets = cfg.get("buckets", defaults["buckets"])
    if not isinstance(raw_buckets, list):
        raw_buckets = defaults["buckets"]
    cfg["buckets"] = [str(x).strip().lower() for x in raw_buckets if str(x).strip()]
    cfg["mode"] = str(cfg.get("mode", "add_bearish_label")).strip().lower() or "add_bearish_label"
    cfg["min_macro_confidence"] = float(
        np.clip(_safe_float(cfg.get("min_macro_confidence", 0.55), 0.55), 0.0, 1.0)
    )
    return cfg


def _build_macro_context_for_industry_pipeline(cfg, mapped_news):
    context = {
        "risk_state": "",
        "trend_score": None,
        "headline_text": "",
        "macro_text": "",
    }
    text_parts = []

    for item in (mapped_news or [])[:80]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if title:
            text_parts.append(title)
        if summary:
            text_parts.append(summary)

    reporting_cfg = cfg.get("reporting", {}) if isinstance(cfg, dict) else {}
    snapshot_path = str(reporting_cfg.get("snapshot_live_path", "outputs/snapshot_live.json"))
    try:
        if os.path.exists(snapshot_path):
            snap = io_safe_read_json(snapshot_path, retries=3, sleep_ms=30)
            if snap is None:
                log_error(
                    f"_build_macro_context_for_industry_pipeline snapshot read warning: "
                    f"safe_read_json returned None for {snapshot_path}"
                )
            elif isinstance(snap, dict):
                risk_cfg = snap.get("risk_config", {})
                if isinstance(risk_cfg, dict):
                    context["risk_state"] = str(
                        risk_cfg.get("risk_state", snap.get("market_state", ""))
                    ).strip().upper()
                    ts = _optional_float(risk_cfg.get("regime_trend_score", None))
                    context["trend_score"] = float(ts) if ts is not None else None
                for k in ("macro_summary", "topic_summary", "last_macro"):
                    val = snap.get(k, "")
                    if isinstance(val, str) and val.strip():
                        text_parts.append(val.strip())
    except Exception as e:
        log_error(f"_build_macro_context_for_industry_pipeline snapshot read error: {e}")

    context["headline_text"] = " | ".join(text_parts[:120])
    context["macro_text"] = context["headline_text"]
    return context


def _is_macro_risk_off(macro_context):
    ctx = macro_context if isinstance(macro_context, dict) else {}
    risk_state = str(ctx.get("risk_state", "")).strip().upper()
    trend_score = _optional_float(ctx.get("trend_score", None))
    text_parts = []
    for key in ("macro_text", "headline_text", "summary", "text", "notes"):
        val = ctx.get(key, "")
        if isinstance(val, str) and val.strip():
            text_parts.append(val.strip().lower())
    text = " ".join(text_parts)

    keyword_map = {
        "hawkish": ["hawkish", "fed hawk", "hawk tone"],
        "rates up": ["rates up", "rate hike", "rates rising", "treasury yields", "yield jump", "yields rising"],
        "higher for longer": ["higher for longer"],
        "volatility spike": ["volatility spike", "vix spike", "vol spike"],
        "risk-off": ["risk-off", "risk off"],
        "credit spread widening": ["credit spread widening", "credit spreads widen", "credit spread widening"],
        "liquidity tightening": ["liquidity tightening", "tight liquidity", "liquidity drain"],
    }
    hits = []
    for canonical, patterns in keyword_map.items():
        if any(str(pat).lower() in text for pat in patterns):
            hits.append(canonical)

    if risk_state == "RISK_OFF":
        hits.append("risk_state_risk_off")
    if trend_score is not None and float(trend_score) <= 0.5:
        hits.append("trend_score_low")

    hit_set = sorted(set(hits))
    strong = {"risk_state_risk_off", "risk-off", "hawkish", "higher for longer", "credit spread widening", "liquidity tightening", "rates up"}
    weak = {"rates up", "volatility spike", "trend_score_low"}
    strong_hits = [h for h in hit_set if h in strong]
    weak_hits = [h for h in hit_set if h in weak]
    risk_off_score = len(hit_set)
    risk_off = bool(strong_hits) or (len(hit_set) >= 2) or (len(weak_hits) >= 2)

    confidence = 0.35 + 0.25 * len(strong_hits) + 0.12 * len(weak_hits)
    if "risk_state_risk_off" in hit_set:
        confidence += 0.20
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return bool(risk_off), hit_set, confidence, int(risk_off_score)


def _get_last_macro_prior_applied_age_minutes(bucket, collection_name, chroma_path):
    if not CHROMADB_IMPORTED:
        return None
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_or_create_collection(name=collection_name)
        rows = None
        # Preferred: metadata filter on macro_prior_applied==True.
        for where_clause in (
            {"$and": [{"L2": str(bucket)}, {"macro_prior_applied": True}]},
            {"L2": str(bucket), "macro_prior_applied": True},
        ):
            try:
                rows = coll.get(
                    where=where_clause,
                    include=["metadatas"],
                    limit=120,
                )
                break
            except Exception:
                rows = None
        if rows is None:
            # Fallback only if where filtering is unavailable.
            rows = coll.get(
                where={"L2": str(bucket)},
                include=["metadatas"],
                limit=120,
            )
        metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []
        now_utc = datetime.now(timezone.utc)
        best_age = None
        for idx, md in enumerate(metadatas):
            if not isinstance(md, dict):
                continue
            applied_meta = bool(md.get("macro_prior_applied", False))
            ts = str(md.get("timestamp", "")).strip()
            if not applied_meta:
                continue
            if not ts:
                continue
            dt = _parse_iso_or_none(ts)
            if dt is None:
                continue
            age_minutes = (now_utc - dt).total_seconds() / 60.0
            if age_minutes < 0:
                continue
            if best_age is None or age_minutes < best_age:
                best_age = age_minutes
        return float(best_age) if best_age is not None else None
    except Exception:
        return None


def _apply_macro_risk_off_prior(
    bucket,
    labels,
    macro_context,
    prior_cfg,
    *,
    collection_name="industry_signals",
    chroma_path="./memory_db",
):
    labels_out = [dict(x) for x in (labels or []) if isinstance(x, dict)]
    info = {
        "macro_prior_applied": False,
        "macro_prior_reasons": [],
        "macro_prior_strength": 0,
        "macro_prior_confidence": None,
        "macro_risk_off_score": 0,
        "macro_prior_skipped_due_to_score": False,
        "macro_prior_skipped_due_to_cooldown": False,
    }
    cfg = prior_cfg if isinstance(prior_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return labels_out, info
    bucket_key = str(bucket or "").strip().lower()
    allowed_buckets = set(str(x).strip().lower() for x in cfg.get("buckets", []) if str(x).strip())
    if bucket_key not in allowed_buckets:
        return labels_out, info
    if str(cfg.get("mode", "add_bearish_label")).strip().lower() != "add_bearish_label":
        return labels_out, info

    risk_off, reasons, macro_conf, risk_off_score = _is_macro_risk_off(macro_context)
    info["macro_risk_off_score"] = int(risk_off_score)
    if (not risk_off) or float(macro_conf) < float(cfg.get("min_macro_confidence", 0.55)):
        info["macro_prior_reasons"] = list(reasons)
        info["macro_prior_confidence"] = float(macro_conf)
        return labels_out, info
    min_score = int(np.clip(_safe_int(cfg.get("min_score", 2), 2), 0, 10))
    if int(risk_off_score) < min_score:
        info["macro_prior_reasons"] = list(reasons)
        info["macro_prior_confidence"] = float(macro_conf)
        info["macro_prior_skipped_due_to_score"] = True
        return labels_out, info
    cooldown_minutes = int(np.clip(_safe_int(cfg.get("cooldown_minutes", 180), 180), 0, 1440 * 7))
    if cooldown_minutes > 0:
        age_minutes = _get_last_macro_prior_applied_age_minutes(
            bucket_key,
            collection_name=collection_name,
            chroma_path=chroma_path,
        )
        if age_minutes is not None and age_minutes < float(cooldown_minutes):
            info["macro_prior_reasons"] = list(reasons)
            info["macro_prior_confidence"] = float(macro_conf)
            info["macro_prior_skipped_due_to_cooldown"] = True
            return labels_out, info

    strength = int(np.clip(_safe_int(cfg.get("strength", 1), 1), 1, 3))
    prior_reason = f"macro_prior: Macro risk-off prior ({', '.join(reasons[:3])})"
    labels_out.append(
        {
            "id": "__macro_prior__",
            "sentiment": "bearish",
            "strength": int(strength),
            "rationale": prior_reason[:140],
        }
    )
    info.update(
        {
            "macro_prior_applied": True,
            "macro_prior_reasons": list(reasons),
            "macro_prior_strength": int(strength),
            "macro_prior_confidence": float(macro_conf),
        }
    )
    return labels_out, info


def _validate_industry_label_payload(parsed_obj, evidence_items):
    if not isinstance(parsed_obj, dict):
        return False, "labels_payload_not_dict", []
    labels = parsed_obj.get("labels", [])
    if not isinstance(labels, list):
        return False, "labels_not_list", []
    if len(labels) != len(evidence_items):
        return False, f"labels_count_mismatch:{len(labels)}!={len(evidence_items)}", []

    allowed_ids = [
        str(x.get("id")).strip() for x in (evidence_items or [])
        if isinstance(x, dict) and str(x.get("id", "")).strip()
    ]
    allowed_id_set = set(allowed_ids)
    seen_ids = set()
    out = []
    for row in labels:
        if not isinstance(row, dict):
            return False, "label_row_not_dict", []
        ev_id = str(row.get("id", "")).strip()
        if not ev_id:
            return False, "label_missing_id", []
        if ev_id not in allowed_id_set:
            return False, f"label_unknown_id:{ev_id}", []
        if ev_id in seen_ids:
            return False, f"label_duplicate_id:{ev_id}", []
        seen_ids.add(ev_id)

        sentiment = str(row.get("sentiment", "")).strip().lower()
        if sentiment not in {"bullish", "bearish", "neutral"}:
            return False, f"invalid_sentiment:{sentiment}", []
        strength = _safe_int(row.get("strength", 1), 1)
        if sentiment == "neutral":
            strength = 1
        else:
            strength = int(np.clip(strength, 1, 3))
        rationale = str(row.get("rationale", "")).strip()
        if len(rationale) > 140:
            rationale = rationale[:140]
        out.append(
            {
                "id": ev_id,
                "sentiment": sentiment,
                "strength": int(strength),
                "rationale": rationale,
            }
        )

    if seen_ids != allowed_id_set:
        missing = sorted(list(allowed_id_set - seen_ids))
        return False, f"missing_ids:{','.join(missing)}", []

    return True, None, out


def _llm_label_industry_evidence(bucket, evidence_items, macro_context, model, llm_timeout_seconds=None):
    asof_utc = datetime.now(timezone.utc).isoformat()
    prompt = _build_industry_llm_prompt(bucket, evidence_items, asof_utc)
    try:
        raw_text = query_ollama(
            model,
            prompt,
            num_ctx=4096,
            temperature=0.1,
            timeout_seconds=llm_timeout_seconds,
        )
        timeout_hit = False
    except Exception as e:
        if "ollama_timeout" in str(e):
            fallback_labels = []
            for item in (evidence_items or []):
                if not isinstance(item, dict):
                    continue
                ev_id = str(item.get("id", "")).strip()
                if not ev_id:
                    continue
                fallback_labels.append(
                    {
                        "id": ev_id,
                        "sentiment": "neutral",
                        "strength": 1,
                        "rationale": "timeout_fallback_neutral",
                    }
                )
            return {
                "ok": True,
                "error": None,
                "raw_text": "",
                "parsed_obj": {"labels": fallback_labels, "notes": "ollama_timeout_fallback"},
                "labels": fallback_labels,
                "notes": "ollama_timeout_fallback",
                "timeout_hit": True,
            }
        raise
    parsed = robust_json_parse(raw_text, model, max_retries=1)
    if bool(parsed.get("_parse_error")):
        reason = str(parsed.get("_parse_error_reason", "parse_error"))
        return {
            "ok": False,
            "error": f"parse_error:{reason}",
            "raw_text": raw_text,
            "parsed_obj": parsed,
            "labels": [],
            "notes": "",
            "timeout_hit": False,
        }
    ok, reason, labels = _validate_industry_label_payload(parsed, evidence_items)
    if not ok:
        return {
            "ok": False,
            "error": f"label_schema_invalid:{reason}",
            "raw_text": raw_text,
            "parsed_obj": parsed,
            "labels": [],
            "notes": str(parsed.get("notes", ""))[:200] if isinstance(parsed, dict) else "",
            "timeout_hit": False,
        }
    return {
        "ok": True,
        "error": None,
        "raw_text": raw_text,
        "parsed_obj": parsed,
        "labels": labels,
        "notes": str(parsed.get("notes", ""))[:200],
        "timeout_hit": bool(timeout_hit),
    }


def _score_industry_from_labels(bucket, labels, rate_sensitivity, macro_context):
    bull = int(sum(int(x.get("strength", 0)) for x in labels if str(x.get("sentiment", "")).strip().lower() == "bullish"))
    bear = int(sum(int(x.get("strength", 0)) for x in labels if str(x.get("sentiment", "")).strip().lower() == "bearish"))
    net = int(bull - bear)

    if net >= 4:
        risk_delta = 0.20
    elif net == 3:
        risk_delta = 0.10
    elif net == 2:
        risk_delta = 0.05
    elif net in (-1, 0, 1):
        risk_delta = 0.0
    elif net == -2:
        risk_delta = -0.05
    elif net == -3:
        risk_delta = -0.10
    else:
        risk_delta = -0.20

    total_polar = bull + bear
    if total_polar <= 0:
        confidence = 0.35
    else:
        dominance = abs(net) / float(total_polar)
        confidence = 0.45 if dominance < 0.5 else 0.55
        if total_polar >= 6 and dominance >= 0.67:
            confidence = 0.65
        if total_polar >= 8 and dominance >= 0.75:
            confidence = 0.75

    allowed_conf = np.array([0.35, 0.45, 0.55, 0.65, 0.75], dtype=float)
    confidence = float(allowed_conf[int(np.argmin(np.abs(allowed_conf - float(confidence))))])
    direction = _direction_from_delta(risk_delta)

    ranked = sorted(
        [x for x in labels if isinstance(x, dict)],
        key=lambda x: int(x.get("strength", 0)),
        reverse=True,
    )
    if direction == "overweight":
        main_side = "bullish"
        counter_side = "bearish"
    elif direction == "underweight":
        main_side = "bearish"
        counter_side = "bullish"
    else:
        main_side = "neutral"
        counter_side = "bullish"

    top_drivers = [str(x.get("rationale", "")).strip() for x in ranked if str(x.get("sentiment", "")).strip().lower() == main_side and str(x.get("rationale", "")).strip()]
    if direction == "neutral" and not top_drivers:
        top_drivers = [str(x.get("rationale", "")).strip() for x in ranked if str(x.get("rationale", "")).strip()]
    top_drivers = top_drivers[:3]

    counterpoints = [str(x.get("rationale", "")).strip() for x in ranked if str(x.get("sentiment", "")).strip().lower() == counter_side and str(x.get("rationale", "")).strip()][:2]

    evidence_used = [
        str(x.get("id", "")).strip()
        for x in ranked
        if str(x.get("id", "")).strip() and str(x.get("id", "")).strip() != "__macro_prior__"
    ][:4]
    return {
        "direction": direction,
        "risk_delta": float(risk_delta),
        "confidence": float(confidence),
        "top_drivers": top_drivers,
        "counterpoints": counterpoints,
        "evidence_used": evidence_used,
        "rate_sensitivity": str(rate_sensitivity),
        "bull_score": bull,
        "bear_score": bear,
        "net_score": net,
        "score_mapping_reason": f"net={net} -> risk_delta={risk_delta:+.2f}",
    }


def _industry_schema_fallback(l2, asof_utc, items, reason):
    neutral = _neutral_industry_signal(l2, asof_utc, items, reason=f"schema_invalid:{reason}")
    neutral["direction"] = "neutral"
    neutral["risk_delta"] = 0.0
    neutral["confidence"] = 0.35
    neutral["status"] = "error"
    drivers = neutral.get("top_drivers", [])
    if not isinstance(drivers, list):
        drivers = []
    tag = f"schema_invalid:{reason}"
    if tag not in [str(x) for x in drivers]:
        drivers = [tag] + [str(x) for x in drivers if str(x).strip()]
    neutral["top_drivers"] = [str(x) for x in drivers if str(x).strip()][:5]
    neutral["counterpoints"] = []
    neutral["evidence_used"] = []
    neutral["bull_score"] = 0
    neutral["bear_score"] = 0
    neutral["net_score"] = 0
    neutral["labels_head"] = []
    neutral["score_mapping_reason"] = "fallback"
    neutral["rate_sensitivity"] = _get_bucket_rate_sensitivity(l2)
    return neutral


def _generate_industry_signal_with_llm(
    l2,
    items,
    model_name,
    return_debug=False,
    macro_context=None,
    macro_risk_off_prior_cfg=None,
    collection_name="industry_signals",
    chroma_path="./memory_db",
    llm_timeout_seconds=None,
):
    asof_utc = datetime.now(timezone.utc).isoformat()
    macro_ctx = {"asof_utc": asof_utc, "bucket": l2}
    if isinstance(macro_context, dict):
        macro_ctx.update(macro_context)
    prior_cfg = (
        macro_risk_off_prior_cfg
        if isinstance(macro_risk_off_prior_cfg, dict)
        else _get_macro_risk_off_prior_cfg({})
    )
    try:
        label_result = _llm_label_industry_evidence(
            l2,
            items,
            macro_ctx,
            model_name,
            llm_timeout_seconds=llm_timeout_seconds,
        )
        raw_text = str(label_result.get("raw_text", ""))
        parsed = label_result.get("parsed_obj", {})
        timeout_hit = bool(label_result.get("timeout_hit", False))
        if not bool(label_result.get("ok", False)):
            reason = str(label_result.get("error", "label_error"))
            neutral = _industry_schema_fallback(l2, asof_utc, items, reason=reason)
            neutral["macro_prior_applied"] = False
            neutral["macro_prior_reasons"] = []
            neutral["macro_prior_strength"] = 0
            neutral["macro_risk_off_score"] = 0
            neutral["macro_prior_skipped_due_to_score"] = False
            neutral["macro_prior_skipped_due_to_cooldown"] = False
            neutral["llm_timeout_hit"] = bool(timeout_hit)
            if return_debug:
                return neutral, raw_text, reason, parsed
            return neutral, raw_text, reason

        labels = label_result.get("labels", [])
        if not isinstance(labels, list):
            labels = []
        labels_scoring, prior_info = _apply_macro_risk_off_prior(
            l2,
            labels,
            macro_ctx,
            prior_cfg,
            collection_name=collection_name,
            chroma_path=chroma_path,
        )
        scored = _score_industry_from_labels(
            l2,
            labels_scoring,
            rate_sensitivity=_get_bucket_rate_sensitivity(l2),
            macro_context=macro_ctx,
        )
        normalized = _neutral_industry_signal(l2, asof_utc, items, reason="scored")
        normalized.update(scored)
        normalized["asof_utc"] = asof_utc
        normalized["scope"] = "industry"
        normalized["L2"] = str(l2)
        normalized["status"] = "ok"
        normalized["notes"] = str(label_result.get("notes", ""))[:240]
        normalized["labels_head"] = [dict(x) for x in labels_scoring[:4] if isinstance(x, dict)]
        normalized["macro_prior_applied"] = bool(prior_info.get("macro_prior_applied", False))
        normalized["macro_prior_reasons"] = list(prior_info.get("macro_prior_reasons", []))
        normalized["macro_prior_strength"] = int(prior_info.get("macro_prior_strength", 0))
        normalized["macro_prior_confidence"] = _optional_float(prior_info.get("macro_prior_confidence", None))
        normalized["macro_risk_off_score"] = int(_safe_int(prior_info.get("macro_risk_off_score", 0), 0))
        normalized["macro_prior_skipped_due_to_score"] = bool(prior_info.get("macro_prior_skipped_due_to_score", False))
        normalized["macro_prior_skipped_due_to_cooldown"] = bool(prior_info.get("macro_prior_skipped_due_to_cooldown", False))
        normalized["llm_timeout_hit"] = bool(timeout_hit)
        normalized["status"] = "ok"
        if return_debug:
            return normalized, raw_text, None, parsed
        return normalized, raw_text, None
    except Exception as e:
        neutral = _neutral_industry_signal(l2, asof_utc, items, reason=f"llm_error:{e}")
        neutral["status"] = "error"
        neutral["macro_risk_off_score"] = 0
        neutral["macro_prior_skipped_due_to_score"] = False
        neutral["macro_prior_skipped_due_to_cooldown"] = False
        neutral["llm_timeout_hit"] = False
        if return_debug:
            return neutral, "", str(e), {}
        return neutral, "", str(e)


def write_industry_signals_to_chroma(signals, collection_name="industry_signals", chroma_path="./memory_db", client_override=None):
    if not isinstance(signals, list) or not signals:
        return {"written": 0, "collection": collection_name}
    if not CHROMADB_IMPORTED:
        return {"written": 0, "collection": collection_name, "error": "chromadb_unavailable"}

    client = client_override
    if client is None:
        client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection(name=collection_name)

    ids = []
    docs = []
    metas = []
    row_details = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        l2 = str(signal.get("L2", "unknown"))
        asof = str(signal.get("asof_utc", datetime.now(timezone.utc).isoformat()))
        date_key = asof[:10]
        payload = json.dumps(signal, ensure_ascii=False)
        item_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        row_id = f"industry::{l2}::{date_key}::{item_hash}"
        row_status = str(signal.get("status", "ok")).strip() or "ok"
        row_meta = {
            "timestamp": asof,
            "status": row_status,
            "scope": "industry",
            "L2": l2,
            "direction": str(signal.get("direction", _direction_from_delta(_safe_float(signal.get("risk_delta", 0.0), 0.0)))),
            "confidence": float(np.clip(_safe_float(signal.get("confidence", 0.0), 0.0), 0.0, 1.0)),
            "risk_delta": float(np.clip(_safe_float(signal.get("risk_delta", 0.0), 0.0), -0.30, 0.30)),
            "horizon": str(signal.get("horizon", "1d")),
            "rate_sensitivity": str(signal.get("rate_sensitivity", "")),
            "source_count": len(signal.get("evidence", []) if isinstance(signal.get("evidence"), list) else []),
            "ticker_count": len(signal.get("impacted_tickers", []) if isinstance(signal.get("impacted_tickers"), list) else []),
            "version": "industry_news_v1",
            "macro_prior_applied": bool(signal.get("macro_prior_applied", False)),
        }
        ids.append(row_id)
        docs.append(payload)
        metas.append(row_meta)
        row_details.append(
            {
                "id": row_id,
                "L2": l2,
                "timestamp": asof,
                "status": row_status,
                "direction": row_meta["direction"],
                "risk_delta": row_meta["risk_delta"],
                "confidence": row_meta["confidence"],
            }
        )
    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
    return {"written": len(ids), "collection": collection_name, "rows": row_details}


def run_industry_news_pipeline(
    config,
    *,
    portfolio_tickers=None,
    candidate_tickers=None,
    synthetic_news=None,
    llm_stub=None,
    chroma_client_override=None,
    chroma_path_override=None,
    debug_trace=None,
):
    cfg = config if isinstance(config, dict) else {}
    news_overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    sources_cfg = _merge_cfg(_default_news_sources_cfg(), cfg.get("news_sources", {}))
    industry_taxonomy = cfg.get("industry_taxonomy", {})

    l2_to_tickers, _l3_to_tickers, ticker_to_tags = build_industry_membership(cfg)
    l2_list = list(l2_to_tickers.keys())
    context = {
        "portfolio_tickers": portfolio_tickers or [],
        "candidate_tickers": candidate_tickers or [],
        "industry_topics": l2_list,
    }

    all_items = []
    if isinstance(synthetic_news, list):
        all_items = [dict(x) for x in synthetic_news if isinstance(x, dict)]
    else:
        providers = []
        if bool(sources_cfg.get("market_rss_enabled", True)):
            providers.append(
                MarketRSSProvider(
                    sources_cfg.get("market_rss_feeds", {}),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                )
            )
        if bool(sources_cfg.get("ticker_rss_enabled", True)):
            providers.append(
                TickerYahooRSSProvider(
                    sources_cfg.get("ticker_rss_template", ""),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                    max_tickers=_safe_int(
                        max(
                            _safe_int(sources_cfg.get("max_portfolio_tickers", 20), 20),
                            _safe_int(sources_cfg.get("max_candidate_tickers", 20), 20),
                        ),
                        20,
                    ),
                )
            )
        if bool(sources_cfg.get("industry_rss_enabled", True)):
            providers.append(
                IndustryGoogleRSSProvider(
                    sources_cfg.get("industry_rss_template", ""),
                    sources_cfg.get("industry_topic_queries", {}),
                    timeout_seconds=_safe_int(sources_cfg.get("timeout_seconds", 8), 8),
                    retries=_safe_int(sources_cfg.get("retries", 1), 1),
                )
            )

        for provider in providers:
            try:
                fetched = provider.fetch(context)
                if isinstance(fetched, list):
                    all_items.extend(fetched)
            except Exception as e:
                log_error(f"[TAXONOMY_PREVIEW] provider {provider.provider_name} failed: {e}")

    max_age_hours = _safe_float(news_overlay_cfg.get("max_age_hours", 48), 48.0)
    age_filtered = []
    for item in all_items:
        row = dict(item)
        if not row.get("published_at"):
            row["published_at"] = datetime.now(timezone.utc).isoformat()
        if _within_hours(row.get("published_at"), max_age_hours):
            age_filtered.append(row)

    deduped = _dedup_news_sorted(age_filtered)
    mapped = map_news_items_to_taxonomy(
        deduped,
        ticker_to_tags=ticker_to_tags,
        industry_taxonomy=industry_taxonomy,
        industry_keyword_map=sources_cfg.get("industry_keyword_map", {}),
    )
    buckets = bucket_news_by_l2(
        mapped,
        l2_list=l2_list,
        max_per_l2=_safe_int(sources_cfg.get("max_per_l2", 8), 8),
        prefer_seed_primary=bool(sources_cfg.get("prefer_seed_primary", True)),
    )
    buckets = _apply_post_bucket_total_cap(buckets, sources_cfg.get("post_bucket_max_total"))
    macro_context_for_prior = _build_macro_context_for_industry_pipeline(cfg, mapped)
    macro_prior_cfg = _get_macro_risk_off_prior_cfg(news_overlay_cfg)
    collection_name = str(news_overlay_cfg.get("industry_collection", "industry_signals"))
    chroma_path = (
        str(chroma_path_override)
        if chroma_path_override
        else str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    )

    debug_bucket_entries = {}
    debug_enabled = isinstance(debug_trace, dict)
    if debug_enabled:
        for l2 in l2_list:
            items = buckets.get(l2, []) if isinstance(buckets, dict) else []
            source_counter = Counter(
                str(x.get("source", "")).strip() for x in items if isinstance(x, dict) and str(x.get("source", "")).strip()
            )
            debug_bucket_entries[str(l2)] = {
                "bucket": str(l2),
                "asof_utc": datetime.now(timezone.utc).isoformat(),
                "bucket_items_count": len(items),
                "evidence_source_counts": dict(source_counter),
                "llm_called": False,
                "llm_error": None,
                "parse_error": False,
                "parse_error_reason": None,
                "raw_text_len": 0,
                "raw_text_head_1200": "",
                "bull_score": 0,
                "bear_score": 0,
                "net_score": 0,
                "macro_prior_applied": False,
                "macro_prior_reasons": [],
                "macro_prior_strength": 0,
                "macro_risk_off_score": 0,
                "macro_prior_skipped_due_to_score": False,
                "macro_prior_skipped_due_to_cooldown": False,
                "labels_head": [],
                "score_mapping_reason": "",
                "parsed_obj": {},
                "normalized_obj": {},
                "final_to_write_obj": {
                    "direction": "neutral",
                    "risk_delta": 0.0,
                    "confidence": 0.0,
                    "status": "no_items",
                },
            }

    generated_signals = []
    llm_errors = []
    llm_model = str(
        cfg.get("news_overlay", {}).get(
            "llm_model",
            cfg.get("macro_integration", {}).get("llm_topic_model", LOCAL_MODEL),
        )
    )
    for l2, items in buckets.items():
        if debug_enabled and str(l2) in debug_bucket_entries:
            debug_bucket_entries[str(l2)]["bucket_items_count"] = len(items)
            debug_bucket_entries[str(l2)]["asof_utc"] = datetime.now(timezone.utc).isoformat()
        if not items:
            continue
        raw_text_for_debug = ""
        parsed_obj_for_debug = {}
        if callable(llm_stub):
            try:
                stub_obj = llm_stub(l2, items)
                if isinstance(stub_obj, str):
                    raw_text_for_debug = stub_obj
                    parsed = robust_json_parse(stub_obj, llm_model, max_retries=1)
                    parsed_obj_for_debug = parsed if isinstance(parsed, dict) else {}
                    normalized = _normalize_industry_signal(
                        parsed,
                        l2,
                        datetime.now(timezone.utc).isoformat(),
                        items,
                    )
                else:
                    parsed_obj_for_debug = stub_obj if isinstance(stub_obj, dict) else {}
                    raw_text_for_debug = json.dumps(parsed_obj_for_debug, ensure_ascii=False)
                    normalized = _normalize_industry_signal(
                        stub_obj,
                        l2,
                        datetime.now(timezone.utc).isoformat(),
                        items,
                    )
                err = None
            except Exception as e:
                normalized = _neutral_industry_signal(l2, datetime.now(timezone.utc).isoformat(), items, reason="stub_error")
                err = str(e)
        else:
            if debug_enabled:
                normalized, raw_text_for_debug, err, parsed_obj_for_debug = _generate_industry_signal_with_llm(
                    l2,
                    items,
                    llm_model,
                    return_debug=True,
                    macro_context=macro_context_for_prior,
                    macro_risk_off_prior_cfg=macro_prior_cfg,
                    collection_name=collection_name,
                    chroma_path=chroma_path,
                )
            else:
                normalized, _raw_text, err = _generate_industry_signal_with_llm(
                    l2,
                    items,
                    llm_model,
                    macro_context=macro_context_for_prior,
                    macro_risk_off_prior_cfg=macro_prior_cfg,
                    collection_name=collection_name,
                    chroma_path=chroma_path,
                )

        if err:
            llm_errors.append({"L2": l2, "error": err})
        generated_signals.append(normalized)

        if debug_enabled and str(l2) in debug_bucket_entries:
            debug_bucket_entries[str(l2)]["llm_called"] = True
            debug_bucket_entries[str(l2)]["llm_error"] = err
            debug_bucket_entries[str(l2)]["raw_text_len"] = len(str(raw_text_for_debug or ""))
            debug_bucket_entries[str(l2)]["raw_text_head_1200"] = str(raw_text_for_debug or "")[:1200]
            debug_bucket_entries[str(l2)]["parsed_obj"] = parsed_obj_for_debug if isinstance(parsed_obj_for_debug, dict) else {}
            debug_bucket_entries[str(l2)]["normalized_obj"] = dict(normalized) if isinstance(normalized, dict) else {}
            debug_bucket_entries[str(l2)]["macro_prior_applied"] = bool(normalized.get("macro_prior_applied", False)) if isinstance(normalized, dict) else False
            debug_bucket_entries[str(l2)]["macro_prior_reasons"] = list(normalized.get("macro_prior_reasons", [])) if isinstance(normalized, dict) else []
            debug_bucket_entries[str(l2)]["macro_prior_strength"] = int(_safe_int(normalized.get("macro_prior_strength", 0), 0)) if isinstance(normalized, dict) else 0
            debug_bucket_entries[str(l2)]["macro_risk_off_score"] = int(_safe_int(normalized.get("macro_risk_off_score", 0), 0)) if isinstance(normalized, dict) else 0
            debug_bucket_entries[str(l2)]["macro_prior_skipped_due_to_score"] = bool(normalized.get("macro_prior_skipped_due_to_score", False)) if isinstance(normalized, dict) else False
            debug_bucket_entries[str(l2)]["macro_prior_skipped_due_to_cooldown"] = bool(normalized.get("macro_prior_skipped_due_to_cooldown", False)) if isinstance(normalized, dict) else False
            if isinstance(parsed_obj_for_debug, dict) and bool(parsed_obj_for_debug.get("_parse_error")):
                debug_bucket_entries[str(l2)]["parse_error"] = True
                debug_bucket_entries[str(l2)]["parse_error_reason"] = str(
                    parsed_obj_for_debug.get("_parse_error_reason", "parse_error")
                )

    generated_signals, calibration_info = _postprocess_industry_signals(
        generated_signals, news_overlay_cfg=news_overlay_cfg
    )

    if debug_enabled:
        final_by_l2 = {}
        for sig in generated_signals:
            if isinstance(sig, dict):
                final_by_l2[str(sig.get("L2", "")).strip()] = sig
        for l2, entry in debug_bucket_entries.items():
            final_obj = final_by_l2.get(str(l2))
            if isinstance(final_obj, dict):
                final_with_status = dict(final_obj)
                final_with_status["status"] = str(final_with_status.get("status", "ok"))
                entry["final_to_write_obj"] = final_with_status
            else:
                entry["final_to_write_obj"] = {
                    "direction": "neutral",
                    "risk_delta": 0.0,
                    "confidence": 0.0,
                    "status": "no_items",
                }

    write_info = write_industry_signals_to_chroma(
        generated_signals,
        collection_name=collection_name,
        chroma_path=chroma_path,
        client_override=chroma_client_override,
    )

    if debug_enabled:
        debug_trace["bucket_entries"] = debug_bucket_entries

    return {
        "raw_count": len(all_items),
        "filtered_count": len(age_filtered),
        "dedup_count": len(deduped),
        "mapped_news": mapped,
        "buckets": buckets,
        "signals": generated_signals,
        "write_info": write_info,
        "ticker_to_tags": ticker_to_tags,
        "l2_to_tickers": {k: sorted(list(v)) for k, v in l2_to_tickers.items()},
        "llm_errors": llm_errors,
        "calibration_info": calibration_info,
        "config_used": {
            "news_overlay": news_overlay_cfg,
            "news_sources": sources_cfg,
        },
    }


def _build_synthetic_industry_news():
    now = datetime.now(timezone.utc)
    return [
        {
            "source": "Synthetic:Industry",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "title": "OPEC discusses deeper production cuts as oil demand rises",
            "summary": "Energy equities and integrated producers react to supply guidance.",
            "url": "https://synthetic.local/news/opec-energy-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Industry",
            "published_at": (now - timedelta(hours=1)).isoformat(),
            "title": "US FDA delays biotech review for obesity treatment",
            "summary": "Healthcare and biotech names face regulatory uncertainty.",
            "url": "https://synthetic.local/news/fda-healthcare-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Ticker",
            "published_at": (now - timedelta(minutes=50)).isoformat(),
            "title": "NVDA supply chain constraints raise semiconductor execution risk",
            "summary": "AI chip demand remains strong but near-term delivery risk increases.",
            "url": "https://synthetic.local/news/nvda-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Ticker",
            "published_at": (now - timedelta(minutes=45)).isoformat(),
            "title": "XLK ETF inflows continue amid cautious software outlook",
            "summary": "Investors rotate to large-cap technology while cloud guidance softens.",
            "url": "https://synthetic.local/news/xlk-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Market",
            "published_at": (now - timedelta(minutes=30)).isoformat(),
            "title": "Treasury yields jump after stronger inflation print",
            "summary": "Rates-sensitive assets and duration exposure reprice lower.",
            "url": "https://synthetic.local/news/rates-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Noise",
            "published_at": (now - timedelta(minutes=20)).isoformat(),
            "title": "Local sports team wins regional final",
            "summary": "No market relevance expected.",
            "url": "https://synthetic.local/news/noise-1",
            "raw_text": "",
        },
        {
            "source": "Synthetic:Duplicate",
            "published_at": (now - timedelta(minutes=18)).isoformat(),
            "title": "OPEC discusses deeper production cuts as oil demand rises",
            "summary": "Duplicate headline for de-dup check.",
            "url": "https://synthetic.local/news/opec-energy-1",
            "raw_text": "",
        },
    ]


def run_industry_news_dryrun(config_path="paper_config.json", outdir="outputs/gw_industry_dryrun"):
    os.makedirs(outdir, exist_ok=True)
    cfg = _load_json_config(config_path)
    checks = []

    def _check(name, condition, detail):
        passed = bool(condition)
        checks.append({"name": name, "pass": passed, "detail": detail})
        print(f"[DRYRUN] {'PASS' if passed else 'FAIL'} {name}: {detail}")
        return passed

    synthetic_news = _build_synthetic_industry_news()
    _check("SYN-A", len(synthetic_news) >= 6, "synthetic news count >= 6")

    attempt_state = {"bad_once": False}

    def _llm_stub(l2, items):
        l2_norm = str(l2).lower()
        if l2_norm == "utilities" and not attempt_state["bad_once"]:
            attempt_state["bad_once"] = True
            return "INVALID_JSON_OUTPUT"
        risk_map = {
            "energy": (-0.7, 0.82),
            "technology": (-0.5, 0.78),
            "healthcare": (-0.3, 0.74),
            "rates_and_gold": (-0.4, 0.71),
            "financials": (-0.2, 0.62),
            "utilities": (0.0, 0.35),
        }
        risk_delta, confidence = risk_map.get(l2_norm, (0.0, 0.35))
        evidence_rows = []
        for item in items[:5]:
            evidence_rows.append(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                }
            )
        impacted = []
        for ticker in sorted({t for row in items for t in row.get("matched_tickers", [])})[:6]:
            impacted.append(
                {
                    "ticker": ticker,
                    "direction": "down" if risk_delta < 0 else "neutral",
                    "magnitude": abs(risk_delta),
                    "reason": f"{l2} headline pressure",
                }
            )
        return {
            "asof_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "industry",
            "L2": l2,
            "L3_focus": [],
            "risk_delta": risk_delta,
            "confidence": confidence,
            "horizon": "1d",
            "top_drivers": [f"{l2} synthetic driver"],
            "impacted_tickers": impacted,
            "evidence": evidence_rows,
            "notes_speculative": False,
        }

    chroma_path = os.path.join(outdir, "memory_db")
    result = run_industry_news_pipeline(
        cfg,
        portfolio_tickers=["NVDA", "XOM", "TLT"],
        candidate_tickers=["XLK", "XLE", "MRK", "JPM"],
        synthetic_news=synthetic_news,
        llm_stub=_llm_stub,
        chroma_path_override=chroma_path,
    )

    mapped = result.get("mapped_news", [])
    buckets = result.get("buckets", {})
    signals = result.get("signals", [])
    write_info = result.get("write_info", {})

    _check("SYN-B", len(mapped) >= 6, f"mapped news count={len(mapped)}")
    non_empty_l2 = [l2 for l2, rows in buckets.items() if isinstance(rows, list) and len(rows) > 0]
    _check("SYN-C", len(non_empty_l2) >= 3, f"L2 buckets with news={len(non_empty_l2)}")

    schema_ok = True
    for sig in signals:
        if not isinstance(sig, dict):
            schema_ok = False
            break
        for key in ("asof_utc", "scope", "L2", "risk_delta", "confidence", "horizon", "evidence"):
            if key not in sig:
                schema_ok = False
                break
    _check("SYN-D", schema_ok, "signal schema validation")

    written = int(write_info.get("written", 0) or 0)
    _check("SYN-E", written >= 3, f"chroma writes={written}")

    overlay_ok = False
    overlay_tickers = []
    try:
        from paper_trading import PaperTradingEngine

        dryrun_cfg_path = os.path.join(outdir, "dryrun_news_overlay_config.json")
        cfg_copy = json.loads(json.dumps(cfg))
        cfg_copy.setdefault("news_overlay", {})
        cfg_copy["news_overlay"]["enabled"] = True
        cfg_copy["news_overlay"]["industry_collection"] = cfg_copy["news_overlay"].get("industry_collection", "industry_signals")
        cfg_copy["news_overlay"]["mode"] = "risk_only"
        cfg_copy["news_overlay"]["alpha"] = float(cfg_copy["news_overlay"].get("alpha", 0.08))
        cfg_copy["news_overlay"]["min_confidence"] = float(cfg_copy["news_overlay"].get("min_confidence", 0.55))
        cfg_copy.setdefault("macro_integration", {})
        cfg_copy["macro_integration"]["chroma_path"] = chroma_path
        io_atomic_write_json(dryrun_cfg_path, cfg_copy, indent=2)

        old_checkpoint_env = os.environ.get("GW_CHECKPOINT_ACTION")
        os.environ["GW_CHECKPOINT_ACTION"] = "fresh"
        engine = None
        try:
            engine = PaperTradingEngine(dryrun_cfg_path)
        finally:
            if old_checkpoint_env is None:
                os.environ.pop("GW_CHECKPOINT_ACTION", None)
            else:
                os.environ["GW_CHECKPOINT_ACTION"] = old_checkpoint_env

        _new_cash_target, overlay_info = engine.apply_news_overlay_to_cash_target(
            ["NVDA", "XLK", "XOM", "TLT"],
            cash_target=0.2,
        )
        ticker_deltas = overlay_info.get("ticker_deltas", {})
        overlay_tickers = [t for t, d in ticker_deltas.items() if _safe_float(d, 0.0) <= 0]
        overlay_ok = len(overlay_tickers) >= 2
    except Exception as e:
        overlay_ok = False
        log_error(f"[DRYRUN] paper overlay verification failed: {e}")
    _check("SYN-F", overlay_ok, f"paper overlay tickers with delta<=0: {overlay_tickers}")

    preview_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "outdir": outdir,
        "result_summary": {
            "raw_count": result.get("raw_count", 0),
            "filtered_count": result.get("filtered_count", 0),
            "dedup_count": result.get("dedup_count", 0),
            "l2_with_news": non_empty_l2,
            "signal_count": len(signals),
            "write_info": write_info,
        },
        "checks": checks,
        "signals": signals[:10],
    }
    out_json = os.path.join(outdir, "industry_news_dryrun_summary.json")
    io_atomic_write_json(out_json, preview_payload, indent=2)
    print(f"[TAXONOMY_PREVIEW] Dryrun summary written: {out_json}")

    pass_count = len([c for c in checks if c["pass"]])
    fail_count = len([c for c in checks if not c["pass"]])
    print(f"DRYRUN_SUMMARY pass={pass_count} fail={fail_count}")
    return 0 if fail_count == 0 else 1


def _run_industry_news_cli_if_requested():
    env_trigger = str(os.environ.get("GW_DEBUG_INDUSTRY_NEWS", "0")).strip() in ("1", "true", "TRUE", "yes")
    arg_trigger = "--debug-industry-news" in sys.argv
    if not env_trigger and not arg_trigger:
        return None

    parser = argparse.ArgumentParser(description="Industry news pipeline debug runner")
    parser.add_argument("--debug-industry-news", action="store_true")
    parser.add_argument("--config", default="paper_config.json")
    parser.add_argument("--debug-outdir", default="outputs/gw_industry_dryrun")
    args, _ = parser.parse_known_args()
    try:
        return run_industry_news_dryrun(config_path=args.config, outdir=args.debug_outdir)
    except Exception as e:
        print(f"[TAXONOMY_PREVIEW] ERROR industry dryrun failed: {e}")
        return 1


def _get_runtime_portfolio_tickers(snapshot_path="outputs/snapshot_live.json"):
    tickers = []
    try:
        if not os.path.exists(snapshot_path):
            return tickers
        with open(snapshot_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        positions = obj.get("positions_detail", {})
        if isinstance(positions, dict):
            for ticker, row in positions.items():
                t = str(ticker).strip().upper()
                if not t or t == "CASH":
                    continue
                qty = 0.0
                if isinstance(row, dict):
                    qty = _safe_float(row.get("quantity", 0.0), 0.0)
                if qty > 0 and t not in tickers:
                    tickers.append(t)
    except Exception as e:
        log_error(f"_get_runtime_portfolio_tickers error: {e}")
    return tickers


def _get_runtime_candidate_tickers(cfg, limit=20):
    out = []
    try:
        universe = cfg.get("universe", []) if isinstance(cfg, dict) else []
        for row in universe:
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker", "")).strip().upper()
            if not t or t == "CASH" or t in out:
                continue
            out.append(t)
            if len(out) >= max(1, _safe_int(limit, 20)):
                break
    except Exception as e:
        log_error(f"_get_runtime_candidate_tickers error: {e}")
    return out


def run_industry_news_pipeline_runtime(config_path="paper_config.json"):
    global _INDUSTRY_PIPELINE_LAST_RUN_TS
    try:
        cfg = _load_json_config(config_path)
    except Exception as e:
        log_error(f"[INDUSTRY_NEWS] config load failed: {e}")
        return {"status": "config_error", "error": str(e)}

    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    if not bool(overlay_cfg.get("enabled", False)):
        return {"status": "disabled"}

    min_interval = max(0, _safe_int(overlay_cfg.get("runtime_min_interval_seconds", 900), 900))
    now_ts = time.time()
    if min_interval > 0 and _INDUSTRY_PIPELINE_LAST_RUN_TS > 0:
        elapsed = now_ts - _INDUSTRY_PIPELINE_LAST_RUN_TS
        if elapsed < min_interval:
            remaining = max(0.0, min_interval - elapsed)
            return {
                "status": "throttled",
                "remaining_seconds": round(remaining, 1),
                "min_interval_seconds": int(min_interval),
            }

    portfolio_tickers = _get_runtime_portfolio_tickers(
        snapshot_path=cfg.get("reporting", {}).get("snapshot_live_path", "outputs/snapshot_live.json")
    )
    candidate_tickers = _get_runtime_candidate_tickers(
        cfg,
        limit=_safe_int(cfg.get("news_sources", {}).get("max_candidate_tickers", 20), 20),
    )

    result = run_industry_news_pipeline(
        cfg,
        portfolio_tickers=portfolio_tickers,
        candidate_tickers=candidate_tickers,
    )
    write_info = result.get("write_info", {})
    print(
        f"[INDUSTRY_NEWS] signals={len(result.get('signals', []))} "
        f"writes={int(write_info.get('written', 0) or 0)} "
        f"collection={write_info.get('collection', overlay_cfg.get('industry_collection', 'industry_signals'))}"
    )
    _INDUSTRY_PIPELINE_LAST_RUN_TS = now_ts
    return {"status": "ok", "result": result}


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
        io_atomic_write_json(TOPIC_MEMORY_PATH, payload, indent=2)
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


def query_ollama(model, prompt, num_ctx=8192, temperature=None, timeout_seconds=None):
    """
    Robust wrapper for ollama.chat.
    - Retries with backoff
    - Optional fallback models
    - Empty content is treated as hard failure
    """
    if temperature is None:
        temperature = TEMPERATURE

    primary = str(model or "").strip()
    model_candidates = []
    if primary:
        model_candidates.append(primary)
    for fallback in OLLAMA_FALLBACK_MODELS:
        fb = str(fallback or "").strip()
        if fb and fb not in model_candidates:
            model_candidates.append(fb)
    if not model_candidates:
        raise RuntimeError("ollama_model_missing")

    retries = max(0, int(DEFAULT_OLLAMA_RETRIES))
    backoff = max(0.2, float(DEFAULT_OLLAMA_RETRY_BACKOFF))
    last_error = None

    timeout_value = None
    if timeout_seconds is not None:
        timeout_value = max(1.0, float(_safe_float(timeout_seconds, 0.0)))

    for model_name in model_candidates:
        for attempt in range(retries + 1):
            try:
                if timeout_value is not None:
                    def _chat_once():
                        return ollama.chat(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            options={"num_ctx": int(num_ctx), "temperature": float(temperature)},
                            keep_alive=OLLAMA_KEEP_ALIVE,
                        )
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_chat_once)
                        try:
                            response = fut.result(timeout=timeout_value)
                        except FuturesTimeoutError:
                            fut.cancel()
                            raise RuntimeError("ollama_timeout")
                else:
                    response = ollama.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_ctx": int(num_ctx), "temperature": float(temperature)},
                        keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                content = str(response.get("message", {}).get("content", "") or "").strip()
                if not content:
                    raise RuntimeError("ollama_empty_content")
                return content
            except Exception as e:
                last_error = e
                if attempt < retries:
                    sleep_sec = backoff * float(attempt + 1)
                    time.sleep(sleep_sec)

    if str(last_error) == "ollama_empty_content":
        raise RuntimeError("ollama_empty_content")
    if str(last_error) == "ollama_timeout":
        raise RuntimeError("ollama_timeout")
    raise RuntimeError(f"ollama_query_failed:{last_error}")

def parse_deepseek_output(text):
    """Extract optional reasoning text and the JSON payload."""
    raw_text = str(text or "")
    thought_process = ""

    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        thought_process = str(think_match.group(1) or "").strip()
    else:
        alt_match = re.search(
            r"(?:THOUGHT_PROCESS|ANALYSIS)\s*:\s*(.*?)\s*(?:FINAL\s*:|$)",
            raw_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if alt_match:
            thought_process = str(alt_match.group(1) or "").strip()

    json_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    json_text = re.sub(r"```json", "", json_text, flags=re.IGNORECASE)
    json_text = re.sub(r"```", "", json_text).strip()

    return thought_process, json_text


def build_reasoning_summary(res):
    """Build a compact human-readable reasoning summary from parsed result fields."""
    data = res if isinstance(res, dict) else {}
    lines = []

    def _add(label, value):
        txt = str(value or "").strip()
        if txt:
            lines.append(f"- {label}: {txt}")

    _add("Event", data.get("event") or data.get("summary") or data.get("reason"))
    _add("Advice", data.get("advice") or data.get("key_risk"))

    key_drivers = data.get("key_drivers", [])
    if isinstance(key_drivers, list):
        for item in key_drivers[:3]:
            txt = str(item or "").strip()
            if txt:
                lines.append(f"- Driver: {txt}")

    counterpoints = data.get("counterpoints", [])
    if isinstance(counterpoints, list):
        for item in counterpoints[:2]:
            txt = str(item or "").strip()
            if txt:
                lines.append(f"- Counterpoint: {txt}")

    top_drivers = data.get("top_drivers", [])
    if isinstance(top_drivers, list):
        for item in top_drivers[:2]:
            txt = str(item or "").strip()
            if txt:
                lines.append(f"- Topic: {txt}")

    predictions = data.get("predictions", {})
    if isinstance(predictions, dict):
        for k, v in list(predictions.items())[:3]:
            k_txt = str(k or "").strip()
            v_txt = str(v or "").strip()
            if k_txt and v_txt:
                lines.append(f"- {k_txt}: {v_txt}")

    if not lines:
        lines.append("- No structured reasoning summary available.")

    return "\n".join(lines[:8])

def extract_json_from_text(text):
    """
     JSON 
    arkdown?
    
    Args:
        text: 
    Returns:
        json_str: ?JSON ?None
    """
    #  1:  { ... } ?
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
                # ?JSON 
                json_candidate = text[start_idx:i+1]
                try:
                    # ?JSON
                    json.loads(json_candidate)
                    return json_candidate
                except Exception as e:
                    # ?
                    start_idx = -1
                    continue
    
    #  2: ?
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
     JSON
    
    Args:
        raw_output: 
        model: 
    Returns:
        repaired_json_str:  JSON  None
    """
    raw_text = str(raw_output or "")
    if len(raw_text.strip()) < 20:
        return None

    repair_prompt = f"""
The following output contains a JSON object but may have extra text or formatting issues.
Please extract and output ONLY the valid JSON object, with NO explanations, NO markdown, NO extra text.

Original output:
{raw_text}

Output ONLY the JSON:
"""
    
    try:
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': repair_prompt}],
            options={"num_ctx": 4096, "temperature": 0}  # ?
        )
        repaired_text = response['message']['content'].strip()
        
        #  JSON
        json_str = extract_json_from_text(repaired_text)
        if json_str:
            # 
            json.loads(json_str)
            return json_str
    except Exception as e:
        pass
    
    return None

def robust_json_parse(raw_content, model, max_retries=1):
    """
     JSON ?+ ?+ 
    
    Args:
        raw_content: 
        model: ?
        max_retries: 
    Returns:
        dict:  JSON 
    """
    raw_text = str(raw_content or "")
    if len(raw_text.strip()) < 20:
        return {
            "_parse_error": True,
            "_parse_error_reason": "empty_raw_content",
            "_raw_head": raw_text[:2000],
            "status": "error",
            "evidence": [],
        }

    #  JSON
    json_str = extract_json_from_text(raw_text)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            pass  # ?
    
    # ?
    for attempt in range(max_retries):
        repaired_json_str = self_repair_json(raw_text, model)
        if repaired_json_str:
            try:
                return json.loads(repaired_json_str)
            except json.JSONDecodeError:
                continue
    
    # 
    return {
        "status": "error",
        "reason": "Failed to parse JSON after extraction and self-repair attempts",
        "raw_output": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
        "evidence": [],
        "_parse_error": True,
        "_parse_error_reason": "json_parse_failed",
        "_raw_head": raw_text[:2000],
    }

# ================= 2.  =================

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

# ================= 2.5. Signal Scoreboard  =================

def get_asset_ticker(asset_name):
    """
    ?ticker
    Args:
        asset_name: ?"CNY/CAD", "Oil", "NVDA"
    Returns:
        ticker: yfinance ticker ?None
    """
    # ?
    if '/' in asset_name:
        parts = asset_name.split('/')
        base, quote = parts[0].strip(), parts[1].strip()
        
        # ?ticker
        for name, info in ASSETS_DB.items():
            if base in name:
                return info['ticker']
        
        # ?
        if base != 'USD' and quote == 'USD':
            return f"{base}=X"
        elif base == 'USD' and quote != 'USD':
            return f"{quote}=X"
    
    # 
    if asset_name.lower() in ['oil', 'crude oil', 'crude']:
        return "CL=F"
    if asset_name.lower() in ['gold', 'xau']:
        return "GC=F"
    
    # 
    if asset_name.isupper() and len(asset_name) <= 5:
        return asset_name
    
    return None

def get_current_price(ticker):
    """
    
    Args:
        ticker: yfinance ticker
    Returns:
        price: float ?None
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
    
    Args:
        asset: 
        direction: Bullish/Bearish/Neutral
        confidence:  (0-10)
        predictions_dict: ?predictions 
        news_sources: 
    """
    try:
        signal_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        #  ticker ?
        ticker = get_asset_ticker(asset)
        current_price = get_current_price(ticker) if ticker else None
        
        # 
        theme = "UNKNOWN"
        if '/' in asset:
            theme = "FX"
        elif asset.upper() in ['OIL', 'GOLD', 'CRUDE']:
            theme = "MACRO"
        elif ticker and len(asset) <= 5 and asset.isupper():
            theme = "STOCK"
        
        # 
        sources = list(set([src for src in news_sources if src]))
        
        # 
        metadata = {
            "signal_id": signal_id,
            "timestamp": timestamp,
            "asset": asset,
            "ticker": ticker or "UNKNOWN",
            "direction": direction,
            "confidence": float(confidence),
            "theme": theme,
            "initial_price": float(current_price) if current_price else 0.0,
            "sources": ",".join(sources[:3]),  # ??
            "status": "PENDING",  # PENDING / VERIFIED
            # 
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
        
        # ?ChromaDB
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
    lang_instruction = "OUTPUT LANGUAGE: ENGLISH"
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
    
    ?PENDING ?
    """
    try:
        # ?PENDING 
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
            
            # ?
            time_diff = (now - signal_time).total_seconds() / 3600  # 
            
            updated = False
            
            #  1h
            if time_diff >= 1 and metadata['price_1h'] == 0.0:
                price_1h = get_historical_price(ticker, signal_time + timedelta(hours=1))
                if price_1h:
                    metadata['price_1h'] = price_1h
                    metadata['return_1h'] = (price_1h - initial_price) / initial_price * 100
                    metadata['correct_1h'] = check_direction(direction, metadata['return_1h'])
                    updated = True
            
            #  4h
            if time_diff >= 4 and metadata['price_4h'] == 0.0:
                price_4h = get_historical_price(ticker, signal_time + timedelta(hours=4))
                if price_4h:
                    metadata['price_4h'] = price_4h
                    metadata['return_4h'] = (price_4h - initial_price) / initial_price * 100
                    metadata['correct_4h'] = check_direction(direction, metadata['return_4h'])
                    updated = True
            
            #  1d
            if time_diff >= 24 and metadata['price_1d'] == 0.0:
                price_1d = get_historical_price(ticker, signal_time + timedelta(days=1))
                if price_1d:
                    metadata['price_1d'] = price_1d
                    metadata['return_1d'] = (price_1d - initial_price) / initial_price * 100
                    metadata['correct_1d'] = check_direction(direction, metadata['return_1d'])
                    updated = True
            
            #  1w
            if time_diff >= 168 and metadata['price_1w'] == 0.0:
                price_1w = get_historical_price(ticker, signal_time + timedelta(weeks=1))
                if price_1w:
                    metadata['price_1w'] = price_1w
                    metadata['return_1w'] = (price_1w - initial_price) / initial_price * 100
                    metadata['correct_1w'] = check_direction(direction, metadata['return_1w'])
                    metadata['status'] = "VERIFIED"  # 
                    updated = True
            
            # ?
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
    ?
    """
    try:
        t = yf.Ticker(ticker)
        # 1
        start = target_time - timedelta(days=1)
        end = target_time + timedelta(days=1)
        hist = t.history(start=start, end=end, interval="1h")
        
        if not hist.empty:
            # ?
            closest_idx = (hist.index - target_time).abs().argmin()
            return float(hist['Close'].iloc[closest_idx])
    except Exception as e:
        log_error(f"get_historical_price error for {ticker}: {str(e)}")
        pass
    return None

def check_direction(predicted_direction, actual_return):
    """
    ?
    Args:
        predicted_direction: Bullish/Bearish/Neutral
        actual_return: ?(%)
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

# ================= 2.6. Early-Warning  =================

def calculate_rsi(ticker, period=14):
    """ RSI """
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
    """Calculate moving average."""
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
    """ ATR (Average True Range)"""
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
    """Get price change percentage."""
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
    """"""
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
    """Get current volume."""
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
    """Get average volume."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        if hist.empty or len(hist) < period:
            return 1  # 
        return int(hist['Volume'].rolling(window=period).mean().iloc[-1])
    except Exception as e:
        log_error(f"get_avg_volume error for {ticker}: {str(e)}")
        return 1

def count_keyword_mentions(keywords, news_list):
    """"""
    count = 0
    for news_item in news_list:
        title = news_item.get('title', '').lower()
        for keyword in keywords:
            if keyword.lower() in title:
                count += 1
                break  # ?
    return count

def calculate_macro_chain_score(asset_name, asset_config, recent_news):
    """
    ?(0-25)
    ?
    """
    score = 0
    evidence = []
    
    try:
        # 1.  (0-10?
        dxy_change = get_price_change("DX-Y.NYB", period="1w")  # 
        dxy_correlation = asset_config['correlations'].get('DXY', 0)
        
        if abs(dxy_correlation) > 0.5:  # ?
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
        
        # 2.  (0-10?
        tnx_change = get_price_change("^TNX", period="1w")  # 10?
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
        
        # 3.  (0-5?
        macro_keywords = ["Fed", "interest rate", "dollar", "USD", "inflation", "", ""]
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
     (0-25)
     RSI?
    """
    score = 0
    evidence = []
    
    try:
        # 1. RSI / (0-12?
        rsi = calculate_rsi(ticker, period=14)
        
        if rsi is not None:
            if rsi > 70:  # 
                score += min((rsi - 70) / 3, 12)
                evidence.append({
                    "type": "price",
                    "indicator": "RSI(14)",
                    "value": f"{rsi:.1f}",
                    "interpretation": "Overbought - potential reversal risk"
                })
            elif rsi < 30:  # 
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
        
        # 2.  (0-10?
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
        
        # 3. ?(0-3?
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
     (0-25)
    
    """
    score = 0
    evidence = []
    
    try:
        # 1. ?(0-12?
        current_atr = calculate_atr(ticker, period=14)
        avg_atr = calculate_atr(ticker, period=50)
        
        if current_atr and avg_atr and avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            
            if atr_ratio > 1.5:  # ?0%+
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
        
        # 2.  (0-8?
        gap = calculate_gap(ticker)
        
        if abs(gap) > 2:  # 2%
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
        
        # 3. ?(0-5?
        current_volume = get_current_volume(ticker)
        avg_volume = get_avg_volume(ticker, period=20)
        
        if current_volume > 0 and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 2:  # ?
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
    ?(0-25)
    ?
    """
    score = 0
    evidence = []
    
    try:
        # ?
        event_keywords = {
            "central_bank": ["Fed", "ECB", "", "interest rate", "monetary policy", "Powell", "Yellen"],
            "policy": ["tariff", "sanction", "regulation", "policy", "law", "trade war", ""],
            "geopolitical": ["war", "conflict", "election", "crisis", "tension", "", ""]
        }
        
        # 1.  (0-10?
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
        
        # 2.  (0-8?
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
        
        # 3.  (0-7?
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
     Early-Warning 
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
    
    # ?
    macro_score, macro_evidence = calculate_macro_chain_score(asset_name, asset_config, recent_news)
    crowding_score, crowding_evidence = calculate_crowding_score(ticker)
    micro_score, micro_evidence = calculate_microstructure_score(ticker)
    event_score, event_evidence = calculate_event_risk_score(recent_news)
    
    # 
    total_score = macro_score + crowding_score + micro_score + event_score
    
    # 
    if total_score >= 76:
        risk_level = "CRITICAL"
    elif total_score >= 51:
        risk_level = "HIGH"
    elif total_score >= 26:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # ?
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
    
    # 
    if risk_level == "CRITICAL":
        recommendation = " CRITICAL: Multiple risk factors at extreme levels. Consider reducing exposure significantly or hedging."
    elif risk_level == "HIGH":
        recommendation = " CAUTION: Elevated risk across multiple dimensions. Monitor closely and consider risk management."
    elif risk_level == "MEDIUM":
        recommendation = " WATCH: Some risk factors elevated. Stay alert to developments."
    else:
        recommendation = " NORMAL: Risk levels within normal range. Continue monitoring."
    
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
     - 
    Args:
        theme:  (FX/MACRO/STOCK/None)
        asset:  (None )
        timeframe:  (1h/4h/1d/1w)
    Returns:
        dict: ?
    """
    try:
        # ?
        where_clause = {}
        if theme:
            where_clause["theme"] = theme
        if asset:
            where_clause["asset"] = asset
        
        # 
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
        
        # ?
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
            
            # 
            cumulative_return = sum(returns)
            
            # ?
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
            
            # 
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
            
            # 
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
            "returns_list": returns,  # ?
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
    rading-Grade Performance Classification?
    
     real-money execution 
    
    Args:
        stats_dict: ?get_signal_statistics?
        theme: ?
        asset: ?
        transaction_cost: ?0.1%?
        max_dd_threshold:  15%?
    
    Returns:
        dict: {
            "classification_v2": str,  # 
            "classification_v1": str,  # 
            "decision_allowed": bool,  # 
            "reason_summary": str,     # ?
            "risk_warnings": list,     # 
            "net_expected_value": float,  # ?
            "multi_timeframe_validated": bool  # ?
        }
    """
    
    # 
    trades_count = stats_dict.get('sample_size', 0)
    accuracy = stats_dict.get('accuracy', 0.0)
    avg_return = stats_dict.get('avg_return', 0.0)
    cumulative_return = stats_dict.get('cumulative_return', 0.0)
    max_drawdown = stats_dict.get('max_drawdown', 0.0)
    volatility = stats_dict.get('volatility', 0.0)
    win_rate = stats_dict.get('win_rate', 0.0)
    profit_factor = stats_dict.get('profit_factor', 0.0)
    
    # ?
    net_expected_value = avg_return - transaction_cost
    
    # ?
    classification_v2 = ""
    classification_v1 = ""
    decision_allowed = False
    reason_summary = ""
    risk_warnings = []
    multi_timeframe_validated = False
    
    # ========== V1 ==========
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
    
    # ==========  ==========
    if trades_count < 30:
        classification_v2 = " Insufficient Data"
        decision_allowed = False
        reason_summary = f"Insufficient verified sample size ({trades_count}/30). Need at least 30 verified signals for reliable evaluation."
        risk_warnings.append(" ")
        risk_warnings.append(" ")
        
        return {
            "classification_v2": classification_v2,
            "classification_v1": classification_v1,
            "decision_allowed": decision_allowed,
            "reason_summary": reason_summary,
            "risk_warnings": risk_warnings,
            "net_expected_value": net_expected_value,
            "multi_timeframe_validated": False
        }
    
    # ========== ?==========
    # ?2  net_expected_value 
    if theme or asset:
        timeframes_to_check = ["1h", "4h", "1d", "1w"]
        positive_timeframes = []
        
        for tf in timeframes_to_check:
            tf_stats = get_signal_statistics(theme=theme, asset=asset, timeframe=tf)
            tf_net_ev = tf_stats.get('avg_return', 0.0) - transaction_cost
            tf_sample = tf_stats.get('sample_size', 0)
            
            if tf_sample >= 10 and tf_net_ev > 0:  #  10 
                positive_timeframes.append(tf)
        
        multi_timeframe_validated = len(positive_timeframes) >= 2
    else:
        # ?theme/asset?
        multi_timeframe_validated = False
    
    # ========== ?==========
    
    #  Tradable Edge
    if (trades_count >= 50 and 
        net_expected_value > 0 and 
        max_drawdown <= max_dd_threshold and
        multi_timeframe_validated):
        
        classification_v2 = " Tradable Edge"
        decision_allowed = True
        reason_summary = (
            f"All tradable criteria passed:\n"
            f"- Sample size sufficient ({trades_count} >= 50)\n"
            f"- Net expected value positive ({net_expected_value:.2f}% > 0)\n"
            f"- Drawdown controlled ({max_drawdown:.2f}% <= {max_dd_threshold}%)\n"
            f"- Multi-timeframe validation passed\n"
            f"=> Eligible for live trading"
        )
        
        # ?
        if profit_factor < 1.5:
            risk_warnings.append(f"Profit factor is modest ({profit_factor:.2f}); use conservative sizing.")
        if volatility > 2.0:
            risk_warnings.append(f"Return volatility is elevated ({volatility:.2f}%); tighten risk controls.")
    
    #  Directional Signal
    elif (accuracy >= 58 and abs(net_expected_value) < 0.2):
        classification_v2 = " Directional Signal"
        decision_allowed = False
        reason_summary = (
            f"Directional accuracy is decent ({accuracy:.1f}%), but net edge is near zero ({net_expected_value:.2f}%).\n"
            f"Transaction costs consume most of the gross edge.\n"
            f"=> Use only for position adjustment, risk confirmation, and multi-signal filtering.\n"
            f"=> Do not use as a standalone trading trigger."
        )
        risk_warnings.append("Do not use this signal as a standalone live-trading trigger.")
        risk_warnings.append("Can be used as an auxiliary confirmation signal.")
    
    elif (avg_return > 0 and net_expected_value <= 0):
        classification_v2 = " Directional Signal"
        decision_allowed = False
        reason_summary = (
            f"Average return is positive ({avg_return:.2f}%), but transaction costs dominate.\n"
            f"Net expected value: {net_expected_value:.2f}%\n"
            f"=> Use only as auxiliary decision support, not standalone execution."
        )
        risk_warnings.append("Transaction costs are too high relative to edge.")
        risk_warnings.append("Do not use this signal as a standalone live-trading trigger.")
    
    #  Unstable / Regime-Dependent
    elif (net_expected_value > 0 and 
          (trades_count < 50 or max_drawdown > max_dd_threshold or not multi_timeframe_validated)):
        
        classification_v2 = " Unstable / Regime-Dependent"
        decision_allowed = False
        
        reasons = []
        if trades_count < 50:
            reasons.append(f"Sample size is near the minimum ({trades_count}/50)")
        if max_drawdown > max_dd_threshold:
            reasons.append(f"Drawdown too high ({max_drawdown:.2f}% > {max_dd_threshold}%)")
        if not multi_timeframe_validated:
            reasons.append("Multi-timeframe validation failed")

        reason_summary = (
            f"Net expected value is positive ({net_expected_value:.2f}%), but key stability issues remain:\n" +
            "\n".join(f"- {r}" for r in reasons) +
            f"\n=> Marked as observation-only\n"
            f"=> Auto trading is not allowed\n"
            f"=> Wait for more data or regime re-validation"
        )
        risk_warnings.append("System stability is insufficient; live trading is not allowed.")
        risk_warnings.append("Continue monitoring and accumulate more samples.")

    # Handle Lucky Streak (revised)
    elif (accuracy <= 55 and avg_return > 0):
        avg_win = stats_dict.get('avg_win', 0.0)
        avg_loss = stats_dict.get('avg_loss', 0.0)

        has_fat_tail = avg_win > 2 * abs(avg_loss) if avg_loss != 0 else False

        if has_fat_tail and profit_factor > 2.0:
            classification_v2 = "Unstable / Regime-Dependent"
            reason_summary = (
                f"Accuracy is low ({accuracy:.1f}%), but payoff looks asymmetric:\n"
                f"- Avg win: {avg_win:.2f}%\n"
                f"- Avg loss: {avg_loss:.2f}%\n"
                f"- Profit factor: {profit_factor:.2f}\n"
                f"=> Possible fat-tail payoff profile\n"
                f"=> More validation data required before trading"
            )
            risk_warnings.append("Low hit-rate with high payoff ratio needs further validation.")
        else:
            classification_v2 = "No Edge"
            reason_summary = (
                f"Accuracy is low ({accuracy:.1f}%), positive returns likely from noise.\n"
                f"Likely a lucky streak without structural edge.\n"
                f"=> Not tradable"
            )
            risk_warnings.append("Likely luck-driven performance; trading is disallowed.")

        decision_allowed = False
        decision_allowed = False
    
    #  No Edge?
    else:
        classification_v2 = " No Edge"
        decision_allowed = False
        
        reasons = []
        if net_expected_value <= 0:
            reasons.append(f"Net expected value is non-positive ({net_expected_value:.2f}%)")
        if max_drawdown > max_dd_threshold * 1.5:  # 
            reasons.append(f"Drawdown exceeds severe threshold ({max_drawdown:.2f}%)")
        if accuracy < 45 and avg_return <= 0:
            reasons.append("Accuracy and returns do not show statistical edge")
        
        reason_summary = (
            "No tradeable edge detected:\n" +
            "\n".join(f"- {r}" for r in reasons) +
            f"\n=> Trading should remain disabled unless strategy logic materially changes"
        )
        risk_warnings.append("No tradeable edge. Keep trading disabled.")
    
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
    """Normalize title for de-duplication."""
    import string
    # ?
    normalized = title.lower()
    # 
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    # 
    normalized = ' '.join(normalized.split())
    return normalized

def get_rss_news():
    """
    ?
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
                if src_count >= 2:  # ??
                    break
                
                # 
                title = e.get('title', '').strip()
                link = e.get('link', '').strip()
                
                if not title or not link:
                    continue
                
                #  1: 
                if link in seen_links:
                    continue
                
                #  2: 
                normalized_title = normalize_title(title)
                if normalized_title in seen_titles:
                    continue
                
                # 
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
                
                # ?
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
    
    return news[:8]  # 


def build_keywords_for_assets(selected_asset_names, selected_pairs):
    """Build lightweight keyword list for targeted RSS filtering in interactive mode."""
    keyword_map = {
        "usd": ["usd", "dollar", "fed", "treasury"],
        "cad": ["cad", "canadian", "canada"],
        "cny": ["cny", "yuan", "renminbi", "china"],
        "jpy": ["jpy", "yen", "boj", "japan"],
        "gbp": ["gbp", "pound", "uk", "boe"],
        "gold": ["gold", "bullion", "xau"],
        "crude": ["oil", "crude", "wti", "brent", "opec"],
        "bitcoin": ["bitcoin", "btc", "crypto"],
        "btc": ["bitcoin", "btc", "crypto"],
    }

    def _asset_token(name):
        txt = str(name or "").strip().lower()
        if not txt:
            return ""
        if "usd" in txt or "dollar" in txt:
            return "usd"
        if "cad" in txt or "canadian" in txt:
            return "cad"
        if "cny" in txt or "yuan" in txt or "renminbi" in txt or "china" in txt:
            return "cny"
        if "jpy" in txt or "yen" in txt or "boj" in txt or "japan" in txt:
            return "jpy"
        if "gbp" in txt or "pound" in txt or "boe" in txt or "british" in txt:
            return "gbp"
        if "gold" in txt or "xau" in txt:
            return "gold"
        if "crude" in txt or "oil" in txt or "wti" in txt or "brent" in txt:
            return "crude"
        if "bitcoin" in txt or txt == "btc":
            return "bitcoin"
        return re.split(r"[^a-z0-9]+", txt)[0] if txt else ""

    out = []
    seen = set()

    for asset_name in (selected_asset_names or []):
        token = _asset_token(asset_name)
        for kw in keyword_map.get(token, [token] if token else []):
            k = str(kw).strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)

    for pair in (selected_pairs or []):
        pair_txt = str(pair or "").strip()
        if not pair_txt:
            continue
        for part in pair_txt.split("/"):
            token = _asset_token(part)
            for kw in keyword_map.get(token, [token] if token else []):
                k = str(kw).strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    out.append(k)

    return out


def filter_news_by_keywords(news, keywords, *, min_keep=4, max_keep=12):
    """Keep targeted headlines first; fallback to mixed list when hits are too few."""
    rows = [x for x in (news or []) if isinstance(x, dict)]
    kws = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not kws:
        return rows[: max(1, int(max_keep))]

    hits = []
    non_hits = []
    for item in rows:
        title_l = str(item.get("title", "")).strip().lower()
        if any(kw in title_l for kw in kws):
            hits.append(item)
        else:
            non_hits.append(item)

    if len(hits) >= int(min_keep):
        return hits[: max(1, int(max_keep))]

    merged = hits + non_hits
    return merged[: max(1, int(max_keep))]

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

# ================= 3. Evidence  =================

def validate_evidence(evidence_list, input_news):
    """
     AI ?evidence 
    Args:
        evidence_list: AI ?evidence 
        input_news: ?List[Dict] with keys: source, title, published, link
    Returns:
        validated_evidence:  evidence  _invalid?
        valid_count: 
    """
    validated = []
    valid_count = 0
    
    for ev in evidence_list:
        headline = ev.get('headline', '').strip()
        is_valid = False
        
        # ?headline  title ?
        for news_item in input_news:
            news_title = news_item.get('title', '')
            # 
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

# ================= 4. AI  (DeepSeek Logic with Evidence) =================

def _get_latest_industry_signals_summary(config_path="paper_config.json", limit=12):
    try:
        cfg = _load_json_config(config_path)
    except Exception:
        cfg = {}

    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    collection_name = str(overlay_cfg.get("industry_collection", "industry_signals"))
    chroma_path = str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    out = []
    if not CHROMADB_IMPORTED:
        return out
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        coll = client.get_or_create_collection(name=collection_name)
        rows = coll.get(limit=max(1, int(limit)), include=["metadatas"])
        metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            out.append(
                {
                    "L2": str(meta.get("L2", "")),
                    "risk_delta": float(np.clip(_safe_float(meta.get("risk_delta", 0.0), 0.0), -1.0, 1.0)),
                    "confidence": float(np.clip(_safe_float(meta.get("confidence", 0.0), 0.0), 0.0, 1.0)),
                    "timestamp": str(meta.get("timestamp", "")),
                    "horizon": str(meta.get("horizon", "")),
                }
            )
    except Exception as e:
        log_error(f"_get_latest_industry_signals_summary error: {e}")
    return out


def _build_interactive_prompt(
    selected_scope,
    selected_targets,
    user_question,
    news_items,
    macro_context,
    latest_industry_signals,
    request_think=False,
):
    compact_news = []
    for item in (news_items or [])[:8]:
        if not isinstance(item, dict):
            continue
        compact_news.append(
            {
                "source": str(item.get("source", "")),
                "title": str(item.get("title", ""))[:220],
                "summary": str(item.get("summary", ""))[:260],
                "published_at": item.get("published") or item.get("published_at"),
                "url": item.get("link") or item.get("url"),
            }
        )
    think_instruction = ""
    if bool(request_think):
        think_instruction = (
            "If supported, output your reasoning inside <think>...</think> FIRST, "
            "then output ONLY the required JSON.\n"
        )
    prompt = f"""
You are GlobalWatch Interactive Analyst.
{think_instruction}

GOAL
- Return a fast, high-signal analysis for ONLY the user-selected target(s).
- Prioritize responsiveness: finish within ~90 seconds and keep output concise.
- Do NOT run broad industry bucket analysis unless explicitly asked.

SCOPE
- mode: INTERACTIVE
- selected_scope: {selected_scope}
- selected_targets: {json.dumps(selected_targets, ensure_ascii=False)}
- user_question: {user_question}

INPUTS
1) Recent news items (already fetched; do NOT fetch new sources):
{json.dumps(compact_news, ensure_ascii=False)}

2) Market context snapshot (already computed):
{json.dumps(macro_context, ensure_ascii=False)}

3) Optional existing background signals (read-only; do NOT recompute heavy pipelines):
- latest_industry_signals_summary (may be empty):
{json.dumps(latest_industry_signals, ensure_ascii=False)}

OUTPUT REQUIREMENTS (STRICT)
- Output MUST be valid JSON only. No markdown. No extra commentary.
- Keep it short: <= 600 output tokens.
- If uncertain or data is insufficient, return status="no_update" with a short explanation.

DECISION RULES
- Focus only on selected_targets; ignore unrelated sectors/buckets/tickers.
- Provide a single event narrative: what happened and why it matters.
- Convert the narrative into actionable "risk parameters" in small bounded deltas.
- Use conservative deltas for INTERACTIVE mode:
  - cash_target_delta in [-0.05, +0.05]
  - risk_on_score in [0, 10]
- Avoid long chain-of-thought. Provide only brief reasoning bullets.

JSON SCHEMA (MUST FOLLOW)
{{
  "mode": "INTERACTIVE",
  "selected_scope": "...",
  "selected_targets": [...],
  "status": "alert" | "watch" | "neutral" | "no_update" | "error",
  "score": 0-10,
  "event": "1-2 sentences summary",
  "key_drivers": ["max 5 short bullets"],
  "impact": {{
    "assets": [
      {{
        "symbol": "e.g. CAD/CNY",
        "direction": "bullish" | "bearish" | "mixed",
        "horizon": "1d" | "1w",
        "confidence": 0.0-1.0,
        "rationale": "1 short sentence"
      }}
    ],
    "risk_params": {{
      "cash_target_delta": -0.05 to +0.05,
      "max_position_weight_delta": -0.05 to +0.05,
      "notes": "short"
    }}
  }},
  "news_used": [
    {{"title": "...", "source": "..."}}
  ],
  "actions": [
    {{"type": "no_action|reduce_risk|increase_risk|hedge", "details": "short"}}
  ],
  "limits": {{"time_budget_sec": 90, "llm_profile": "interactive"}}
}}

NOW PRODUCE THE JSON.
"""
    return prompt


def analyze_all(
    news,
    user_pairs,
    macro_data,
    lang_mode,
    selected_scope="FX_PAIR",
    selected_targets=None,
    user_question="",
    latest_industry_signals_summary=None,
    request_think=False,
    *,
    run_topic_signals=None,
    run_industry_runtime=None,
    industry_config_path=None,
):
    if run_topic_signals is None:
        run_topic_signals = bool(RUNTIME_SETTINGS.get("enable_llm_topic_signals", True))
    if run_industry_runtime is None:
        run_industry_runtime = True
    if industry_config_path is None:
        industry_config_path = os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")

    if not news:
        return {
            "mode": "INTERACTIVE",
            "selected_scope": str(selected_scope),
            "selected_targets": selected_targets or [],
            "status": "no_update",
            "score": 0,
            "event": "No recent news available.",
            "impact_score": 0,
            "summary": "No recent news available.",
            "predictions": {},
            "advice": "no_action",
        }

    selected_targets = selected_targets if isinstance(selected_targets, list) and selected_targets else list(user_pairs or [])
    if not selected_targets:
        selected_targets = ["US_MACRO"]
    if latest_industry_signals_summary is None:
        latest_industry_signals_summary = _get_latest_industry_signals_summary(
            config_path=os.environ.get("PAPER_CONFIG_PATH", "paper_config.json"),
            limit=12,
        )
    prompt = _build_interactive_prompt(
        selected_scope=str(selected_scope),
        selected_targets=selected_targets,
        user_question=str(user_question or ""),
        news_items=news,
        macro_context=macro_data or {},
        latest_industry_signals=latest_industry_signals_summary or [],
        request_think=bool(request_think),
    )

    try:
        raw_content = query_ollama(LOCAL_MODEL, prompt, num_ctx=4096, temperature=0.15)
        thought, json_text = parse_deepseek_output(raw_content)
        parsed = robust_json_parse(json_text, LOCAL_MODEL, max_retries=1)

        if parsed.get("_parse_error"):
            parsed["thought_process"] = thought
            return parsed

        status = str(parsed.get("status", "no_update")).strip().lower()
        if status not in {"alert", "watch", "neutral", "no_update", "error"}:
            status = "no_update"
        score = float(np.clip(_safe_float(parsed.get("score", parsed.get("impact_score", 0.0)), 0.0), 0.0, 10.0))
        key_drivers = parsed.get("key_drivers", [])
        if not isinstance(key_drivers, list):
            key_drivers = []
        key_drivers = [str(x) for x in key_drivers if str(x).strip()][:5]

        impact_obj = parsed.get("impact", {}) if isinstance(parsed.get("impact"), dict) else {}
        assets = impact_obj.get("assets", [])
        if not isinstance(assets, list):
            assets = []
        normalized_assets = []
        selected_upper = {str(x).strip().upper() for x in selected_targets}
        for item in assets[:12]:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip()
            if not symbol:
                continue
            if selected_upper and symbol.upper() not in selected_upper:
                continue
            direction = str(item.get("direction", "mixed")).strip().lower()
            if direction not in {"bullish", "bearish", "mixed"}:
                direction = "mixed"
            horizon = str(item.get("horizon", "1d")).strip().lower()
            if horizon not in {"1d", "1w"}:
                horizon = "1d"
            normalized_assets.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "horizon": horizon,
                    "confidence": float(np.clip(_safe_float(item.get("confidence", 0.0), 0.0), 0.0, 1.0)),
                    "rationale": str(item.get("rationale", ""))[:220],
                }
            )
        if not normalized_assets:
            for tgt in selected_targets[:3]:
                normalized_assets.append(
                    {
                        "symbol": str(tgt),
                        "direction": "mixed",
                        "horizon": "1d",
                        "confidence": 0.3,
                        "rationale": "Insufficient direct evidence for a directional call.",
                    }
                )

        risk_params = impact_obj.get("risk_params", {}) if isinstance(impact_obj.get("risk_params"), dict) else {}
        risk_params_norm = {
            "cash_target_delta": float(np.clip(_safe_float(risk_params.get("cash_target_delta", 0.0), 0.0), -0.05, 0.05)),
            "max_position_weight_delta": float(np.clip(_safe_float(risk_params.get("max_position_weight_delta", 0.0), 0.0), -0.05, 0.05)),
            "notes": str(risk_params.get("notes", ""))[:220],
        }

        news_used = parsed.get("news_used", [])
        if not isinstance(news_used, list):
            news_used = []
        news_used_norm = []
        for item in news_used[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            source = str(item.get("source", "")).strip()
            if title:
                news_used_norm.append({"title": title, "source": source})
        if not news_used_norm:
            for item in news[:5]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                if title:
                    news_used_norm.append({"title": title, "source": str(item.get("source", ""))})

        actions = parsed.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        actions_norm = []
        for item in actions[:3]:
            if not isinstance(item, dict):
                continue
            act_type = str(item.get("type", "no_action")).strip().lower()
            if act_type not in {"no_action", "reduce_risk", "increase_risk", "hedge"}:
                act_type = "no_action"
            actions_norm.append({"type": act_type, "details": str(item.get("details", ""))[:220]})
        if not actions_norm:
            actions_norm.append({"type": "no_action", "details": "Hold current stance until clearer evidence."})

        event_text = str(parsed.get("event", parsed.get("summary", ""))).strip()[:420]
        if not event_text:
            event_text = "No clear actionable update from selected targets."

        res = {
            "mode": "INTERACTIVE",
            "selected_scope": str(parsed.get("selected_scope", selected_scope)),
            "selected_targets": selected_targets,
            "status": status,
            "score": score,
            "event": event_text,
            "key_drivers": key_drivers,
            "impact": {
                "assets": normalized_assets[:5],
                "risk_params": risk_params_norm,
            },
            "news_used": news_used_norm[:5],
            "actions": actions_norm,
            "limits": {"time_budget_sec": 90, "llm_profile": "interactive"},
            "thought_process": thought,
        }

        # Backward-compatible fields for existing UI components.
        res["impact_score"] = score
        res["summary"] = event_text
        pred = {}
        for item in normalized_assets:
            pred[item["symbol"]] = f"{item['direction']} ({item['horizon']}, conf={item['confidence']:.2f})"
        res["predictions"] = pred
        res["advice"] = actions_norm[0].get("details", "") if actions_norm else risk_params_norm.get("notes", "")
        res["evidence"] = [
            {
                "source": str(item.get("source", "")),
                "headline": str(item.get("title", "")),
                "why_it_matters": event_text,
            }
            for item in news_used_norm[:5]
        ]
        res["_valid_evidence_count"] = len(res["evidence"])

        news_sources = [item.get("source") for item in news if isinstance(item, dict)]
        if run_topic_signals:
            try:
                topic_payload = extract_llm_topic_sentiment(news, macro_data, lang_mode)
                res["_topic_signals_recorded"] = int(record_llm_topic_signals(topic_payload, news_sources) if topic_payload else 0)
            except Exception as e:
                log_error(f"[TOPIC_SIGNALS] interactive record failed: {e}")
                res["_topic_signals_recorded"] = 0
        else:
            res["_topic_signals_recorded"] = 0

        if res.get("status") == "alert" and res.get("predictions"):
            for asset, prediction_text in res.get("predictions", {}).items():
                direction = "Neutral"
                txt = str(prediction_text).lower()
                if "bullish" in txt:
                    direction = "Bullish"
                elif "bearish" in txt:
                    direction = "Bearish"
                record_signal(
                    asset=asset,
                    direction=direction,
                    confidence=score,
                    predictions_dict=res.get("predictions", {}),
                    news_sources=news_sources,
                )
        if res.get("status") == "alert":
            save_to_memory(res.get("summary"), res.get("impact_score", 0), res.get("advice"))

        # Interactive plane should not trigger broad batch pipeline unless explicitly requested.
        if run_industry_runtime:
            try:
                industry_runtime = run_industry_news_pipeline_runtime(
                    config_path=industry_config_path
                )
                if isinstance(industry_runtime, dict):
                    res["_industry_news_status"] = industry_runtime.get("status")
                    if industry_runtime.get("status") == "ok":
                        result_obj = industry_runtime.get("result", {})
                        write_info = result_obj.get("write_info", {}) if isinstance(result_obj, dict) else {}
                        res["_industry_news_written"] = int(write_info.get("written", 0) or 0)
                    else:
                        res["_industry_news_written"] = 0
            except Exception as e:
                log_error(f"[INDUSTRY_NEWS] runtime pipeline failed: {e}")
                res["_industry_news_status"] = "error"
                res["_industry_news_written"] = 0
        else:
            res["_industry_news_status"] = "skipped"
            res["_industry_news_written"] = 0

        return res
    except Exception as e:
        return {
            "mode": "INTERACTIVE",
            "selected_scope": str(selected_scope),
            "selected_targets": selected_targets,
            "status": "error",
            "reason": f"Unexpected error: {str(e)}",
            "raw_output": "",
            "evidence": [],
            "_parse_error": True,
        }

def analyze_single_stock(ticker, news, lang_mode, request_think=False):
    lang_instruction = "OUTPUT LANGUAGE: ENGLISH"
    news_str = " ".join(news)
    
    think_instruction = ""
    if bool(request_think):
        think_instruction = (
            "If supported, output your reasoning inside <think>...</think> FIRST, "
            "then output ONLY the required JSON.\n"
        )
    prompt = f"""
    You are a Wall Street Analyst. {lang_instruction}
    {think_instruction}
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
        
        # ?robust_json_parse
        res = robust_json_parse(json_text, LOCAL_MODEL, max_retries=1)
        
        # ?
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


def _build_trading_loop_prompt(
    cycle_id,
    timestamp_local,
    portfolio_state,
    macro_snapshot,
    topic_signals,
    latest_industry_signals,
):
    return f"""
You are GlobalWatch Trading Risk Parameter Composer.

GOAL
- Convert existing signals (macro + industry) into conservative, executable trading parameters
  that a paper trading engine can apply next cycle.
- DO NOT fetch news, DO NOT run deep analysis, DO NOT regenerate industry buckets.

MODE
- mode: TRADING_LOOP
- cycle_id: {cycle_id}
- timestamp_local: {timestamp_local}

INPUTS
1) Portfolio state:
{json.dumps(portfolio_state or {{}}, ensure_ascii=False)}

2) Current macro regime & trend snapshot:
{json.dumps(macro_snapshot or {{}}, ensure_ascii=False)}

3) Latest confirmed topic signals (optional):
{json.dumps(topic_signals or {{}}, ensure_ascii=False)}

4) Latest industry_signals summary (already computed by batch job):
{json.dumps(latest_industry_signals or [], ensure_ascii=False)}

CONSTRAINTS (MUST OBEY)
- cash_target must be within [min_cash, max_cash] from portfolio constraints
- max_weight_per_asset within [0.05, 0.40] unless constraints specify otherwise
- if regime == RISK_OFF:
  - do NOT propose offensive tilts
  - prefer defensive assets/tags
- keep deltas small and stable (avoid thrashing):
  - cash_target_delta magnitude <= 0.10 per cycle
  - per-bucket tilt delta magnitude <= 0.05

OUTPUT REQUIREMENTS (STRICT)
- Output MUST be valid JSON only. No markdown.
- Keep it short: <= 400 output tokens.
- If signals are stale or missing, return a safe default (neutral parameters).

JSON SCHEMA (MUST FOLLOW)
{{
  "mode": "TRADING_LOOP",
  "cycle_id": {cycle_id},
  "regime": "RISK_ON" | "RISK_OFF" | "RISK_MIXED",
  "cash_target": 0.0-1.0,
  "max_weight_per_asset": 0.0-1.0,
  "tilts": [
    {{"tag": "energy|technology|consumer|...", "delta": -0.05 to +0.05, "reason": "short"}}
  ],
  "risk_limits": {{
    "turnover_cap": 0.0-1.0,
    "notes": "short"
  }},
  "explain": {{"summary": "1-2 sentences", "drivers": ["max 4 bullets"]}},
  "limits": {{"llm_profile": "trading_loop"}}
}}

NOW PRODUCE THE JSON.
"""


def compose_trading_loop_risk_parameters(
    cycle_id,
    portfolio_state,
    macro_snapshot,
    topic_signals=None,
    latest_industry_signals=None,
):
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = _build_trading_loop_prompt(
        cycle_id=cycle_id,
        timestamp_local=now_local,
        portfolio_state=portfolio_state,
        macro_snapshot=macro_snapshot,
        topic_signals=topic_signals or {},
        latest_industry_signals=latest_industry_signals or [],
    )
    safe_default = {
        "mode": "TRADING_LOOP",
        "cycle_id": cycle_id,
        "regime": "RISK_MIXED",
        "cash_target": float(np.clip(_safe_float((portfolio_state or {}).get("cash_target", 0.2), 0.2), 0.0, 1.0)),
        "max_weight_per_asset": float(np.clip(_safe_float((portfolio_state or {}).get("max_weight_per_asset", 0.2), 0.2), 0.05, 0.40)),
        "tilts": [],
        "risk_limits": {"turnover_cap": float(np.clip(_safe_float((portfolio_state or {}).get("turnover_cap", 0.2), 0.2), 0.0, 1.0)), "notes": "safe default"},
        "explain": {"summary": "Insufficient fresh signals; keep neutral risk parameters.", "drivers": []},
        "limits": {"llm_profile": "trading_loop"},
    }
    try:
        raw = query_ollama(LOCAL_MODEL, prompt, num_ctx=3072, temperature=0.1)
        parsed = robust_json_parse(raw, LOCAL_MODEL, max_retries=1)
        if parsed.get("_parse_error"):
            return safe_default

        constraints = {}
        if isinstance(portfolio_state, dict):
            constraints = portfolio_state.get("constraints", {})
        min_cash = float(np.clip(_safe_float(constraints.get("min_cash", 0.0), 0.0), 0.0, 1.0))
        max_cash = float(np.clip(_safe_float(constraints.get("max_cash", 1.0), 1.0), 0.0, 1.0))
        if max_cash < min_cash:
            max_cash = min_cash

        regime = str(parsed.get("regime", "RISK_MIXED")).strip().upper()
        if regime not in {"RISK_ON", "RISK_OFF", "RISK_MIXED"}:
            regime = "RISK_MIXED"
        cash_target = float(np.clip(_safe_float(parsed.get("cash_target", safe_default["cash_target"]), safe_default["cash_target"]), min_cash, max_cash))
        max_weight = float(np.clip(_safe_float(parsed.get("max_weight_per_asset", safe_default["max_weight_per_asset"]), safe_default["max_weight_per_asset"]), 0.05, 0.40))

        tilts_raw = parsed.get("tilts", [])
        tilts = []
        if isinstance(tilts_raw, list):
            for item in tilts_raw[:12]:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("tag", "")).strip()
                if not tag:
                    continue
                delta = float(np.clip(_safe_float(item.get("delta", 0.0), 0.0), -0.05, 0.05))
                if regime == "RISK_OFF" and delta > 0:
                    continue
                tilts.append({"tag": tag, "delta": delta, "reason": str(item.get("reason", ""))[:180]})

        risk_limits = parsed.get("risk_limits", {})
        if not isinstance(risk_limits, dict):
            risk_limits = {}
        turnover_cap = float(np.clip(_safe_float(risk_limits.get("turnover_cap", safe_default["risk_limits"]["turnover_cap"]), safe_default["risk_limits"]["turnover_cap"]), 0.0, 1.0))

        explain = parsed.get("explain", {})
        if not isinstance(explain, dict):
            explain = {}
        drivers = explain.get("drivers", [])
        if not isinstance(drivers, list):
            drivers = []

        return {
            "mode": "TRADING_LOOP",
            "cycle_id": cycle_id,
            "regime": regime,
            "cash_target": cash_target,
            "max_weight_per_asset": max_weight,
            "tilts": tilts,
            "risk_limits": {
                "turnover_cap": turnover_cap,
                "notes": str(risk_limits.get("notes", ""))[:180],
            },
            "explain": {
                "summary": str(explain.get("summary", ""))[:240],
                "drivers": [str(x)[:180] for x in drivers[:4]],
            },
            "limits": {"llm_profile": "trading_loop"},
        }
    except Exception as e:
        log_error(f"compose_trading_loop_risk_parameters error: {e}")
        return safe_default

# ================= 4. UI  =================



def _safe_read_json(path):
    try:
        obj = io_safe_read_json(path, retries=2, sleep_ms=15)
        return obj if isinstance(obj, dict) else {}
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


@st.cache_data(ttl=5, show_spinner=False)
def read_jsonl_tail(path, max_lines=5000):
    rows = deque(maxlen=max(1, int(max_lines)))
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = str(line).strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return list(rows)
    except Exception as e:
        log_error(f"read_jsonl_tail error for {path}: {str(e)}")
        return []


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

    # Parse mixed timestamp formats robustly:
    # - old rows may be naive ISO strings
    # - newer rows are timezone-aware ISO strings (e.g. +00:00)
    # Parse row-by-row first to avoid pandas mixed-format inference dropping aware rows.
    def _parse_mixed_time(value):
        if value is None:
            return pd.NaT
        text = str(value).strip()
        if not text:
            return pd.NaT
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return pd.Timestamp(dt)
        except Exception:
            try:
                return pd.to_datetime(text, errors="coerce", utc=True)
            except Exception:
                return pd.NaT

    parsed_points = [_parse_mixed_time(v) for v in df["time"].tolist()]
    time_utc = pd.to_datetime(pd.Series(parsed_points), errors="coerce", utc=True)
    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is not None:
        df["time"] = time_utc.dt.tz_convert(local_tz).dt.tz_localize(None)
    else:
        df["time"] = time_utc.dt.tz_localize(None)
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
    st.caption("Live paper trading monitor")

    snapshot_path = "outputs/snapshot_live.json"
    summary_path = "outputs/paper_summary_live.txt"
    trades_path = "outputs/trade_history.jsonl"

    defaults = {
        "pm_auto_refresh_data": True,
        "pm_auto_refresh_interval": 60,
        "pm_refresh_nonce": 0,
        "pm_window_hours": 48,
        "pm_resample_rule": "15min",
        "pm_x_tick_mode": "auto",
        "pm_x_tick_minutes": 15,
        "pm_y_mode": "auto",
        "pm_y_metric": "equity",
        "pm_y_min": 0.0,
        "pm_y_max": 1.0,
        "pm_y_dtick": 0.0,
        "pm_y_bounds_initialized": False,
        "pm_hide_off_hours": True,
        "pm_telemetry_rows": 20,
        "pm_telemetry_show_events": False,
        "pm_telemetry_level": "ALL",
        "pm_telemetry_cycle_filter": False,
        "pm_telemetry_cycle_min": 0,
        "pm_telemetry_cycle_max": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    top_col1, top_col2, top_col3 = st.columns([1, 1, 1])
    refresh_clicked = top_col1.button("Refresh data", key="pm_refresh_button")
    top_col2.toggle("Auto refresh data", key="pm_auto_refresh_data")

    interval_options = [10, 30, 60, 300, 1200]
    interval_value = int(st.session_state.get("pm_auto_refresh_interval", 60))
    if interval_value not in interval_options:
        interval_value = 60
        st.session_state["pm_auto_refresh_interval"] = 60

    top_col3.selectbox(
        "Auto refresh interval (sec)",
        interval_options,
        index=interval_options.index(interval_value),
        key="pm_auto_refresh_interval",
    )

    if refresh_clicked:
        st.session_state["pm_refresh_nonce"] = int(st.session_state.get("pm_refresh_nonce", 0)) + 1

    auto_state = "ON" if st.session_state.get("pm_auto_refresh_data") else "OFF"
    st.caption(
        f"Data refresh: {auto_state} | interval: {int(st.session_state.get('pm_auto_refresh_interval', 60))}s"
    )

    def load_snapshot(path, interval_seconds, refresh_nonce):
        ttl_seconds = 1
        try:
            stat = os.stat(path)
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            file_size = int(stat.st_size)
        except Exception:
            mtime_ns = 0
            file_size = 0

        @st.cache_data(ttl=ttl_seconds, show_spinner=False)
        def _cached_snapshot(snapshot_file, nonce, mtime_ns_key, size_key):
            return _safe_read_json(snapshot_file)

        try:
            return _cached_snapshot(path, int(refresh_nonce), int(mtime_ns), int(file_size))
        except KeyError:
            return _safe_read_json(path)
        except Exception:
            return _safe_read_json(path)

    def load_summary(path, interval_seconds, refresh_nonce):
        ttl_seconds = 1
        try:
            stat = os.stat(path)
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            file_size = int(stat.st_size)
        except Exception:
            mtime_ns = 0
            file_size = 0

        @st.cache_data(ttl=ttl_seconds, show_spinner=False)
        def _cached_summary(summary_file, nonce, mtime_ns_key, size_key):
            return _safe_read_text(summary_file)

        try:
            return _cached_summary(path, int(refresh_nonce), int(mtime_ns), int(file_size))
        except KeyError:
            return _safe_read_text(path)
        except Exception:
            return _safe_read_text(path)

    def load_trade_history(path, interval_seconds, refresh_nonce):
        ttl_seconds = 1
        try:
            stat = os.stat(path)
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            file_size = int(stat.st_size)
        except Exception:
            mtime_ns = 0
            file_size = 0

        @st.cache_data(ttl=ttl_seconds, show_spinner=False)
        def _cached_trades(trades_file, nonce, mtime_ns_key, size_key):
            return _safe_read_jsonl(trades_file)

        try:
            return _cached_trades(path, int(refresh_nonce), int(mtime_ns), int(file_size))
        except KeyError:
            return _safe_read_jsonl(path)
        except Exception:
            return _safe_read_jsonl(path)

    def load_daily_reports_index(path, interval_seconds, refresh_nonce):
        ttl_seconds = max(1, int(interval_seconds))

        @st.cache_data(ttl=ttl_seconds, show_spinner=False)
        def _cached_index(index_file, nonce):
            payload = _safe_read_json(index_file)
            reports = payload.get("reports", []) if isinstance(payload, dict) else []
            if not isinstance(reports, list):
                reports = []
            reports_clean = [r for r in reports if isinstance(r, dict)]
            reports_clean.sort(key=lambda x: str(x.get("date", "")), reverse=True)
            return {"reports": reports_clean}

        try:
            return _cached_index(path, int(refresh_nonce))
        except KeyError:
            payload = _safe_read_json(path)
            reports = payload.get("reports", []) if isinstance(payload, dict) else []
            if not isinstance(reports, list):
                reports = []
            reports_clean = [r for r in reports if isinstance(r, dict)]
            reports_clean.sort(key=lambda x: str(x.get("date", "")), reverse=True)
            return {"reports": reports_clean}
        except Exception:
            payload = _safe_read_json(path)
            reports = payload.get("reports", []) if isinstance(payload, dict) else []
            if not isinstance(reports, list):
                reports = []
            reports_clean = [r for r in reports if isinstance(r, dict)]
            reports_clean.sort(key=lambda x: str(x.get("date", "")), reverse=True)
            return {"reports": reports_clean}

    def load_daily_reports_by_paths(paths, interval_seconds, refresh_nonce):
        ttl_seconds = max(1, int(interval_seconds))
        path_payload = json.dumps(list(paths or []), ensure_ascii=False)

        @st.cache_data(ttl=ttl_seconds, show_spinner=False)
        def _cached_reports(paths_json, nonce):
            try:
                path_list = json.loads(paths_json)
            except Exception:
                path_list = []
            if not isinstance(path_list, list):
                path_list = []

            reports = []
            for p in path_list:
                if not isinstance(p, str) or not p.strip():
                    continue
                payload = _safe_read_json(p)
                if isinstance(payload, dict) and payload.get("date"):
                    reports.append(payload)
            reports.sort(key=lambda x: str(x.get("date", "")))
            return reports

        try:
            return _cached_reports(path_payload, int(refresh_nonce))
        except KeyError:
            reports = []
            for p in list(paths or []):
                if not isinstance(p, str) or not p.strip():
                    continue
                payload = _safe_read_json(p)
                if isinstance(payload, dict) and payload.get("date"):
                    reports.append(payload)
            reports.sort(key=lambda x: str(x.get("date", "")))
            return reports
        except Exception:
            reports = []
            for p in list(paths or []):
                if not isinstance(p, str) or not p.strip():
                    continue
                payload = _safe_read_json(p)
                if isinstance(payload, dict) and payload.get("date"):
                    reports.append(payload)
            reports.sort(key=lambda x: str(x.get("date", "")))
            return reports

    def _render_data_panel():
        interval_seconds = int(st.session_state.get("pm_auto_refresh_interval", 60))
        refresh_nonce = int(st.session_state.get("pm_refresh_nonce", 0))

        snapshot = load_snapshot(snapshot_path, interval_seconds, refresh_nonce)
        summary_text = load_summary(summary_path, interval_seconds, refresh_nonce)
        trades = load_trade_history(trades_path, interval_seconds, refresh_nonce)
        daily_reports_index = load_daily_reports_index(
            os.path.join("outputs", "Daily Report", "daily_reports_index.json"),
            interval_seconds,
            refresh_nonce,
        )
        daily_report_entries = daily_reports_index.get("reports", []) if isinstance(daily_reports_index, dict) else []

        if not isinstance(snapshot, dict):
            snapshot = {}
        if not isinstance(trades, list):
            trades = []
        if not isinstance(daily_report_entries, list):
            daily_report_entries = []

        # Create container inside the render call so fragment-owned widgets
        # are not written into a container created outside fragment context.
        with st.container():
            total_equity = _to_float(snapshot.get("total_equity", snapshot.get("equity", 0.0)))
            cash = _to_float(snapshot.get("cash", 0.0))
            positions_value = _to_float(snapshot.get("positions_value", max(0.0, total_equity - cash)))
            drawdown = _to_float(snapshot.get("drawdown", 0.0))

            run_id_raw = snapshot.get("run_id") or snapshot.get("session_id")
            run_id_text = str(run_id_raw).strip() if run_id_raw is not None else ""
            run_id_short = run_id_text[:8] if run_id_text else "-"
            schema_raw = snapshot.get("schema_version")
            schema_text = str(schema_raw).strip() if schema_raw is not None and str(schema_raw).strip() else "-"
            cycle_raw = snapshot.get("cycle_id")
            if cycle_raw in (None, ""):
                cycle_raw = snapshot.get("cycle")
            cycle_text = str(cycle_raw).strip() if cycle_raw is not None and str(cycle_raw).strip() else "-"

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Equity", f"${total_equity:,.2f}")
            metric_col2.metric("Cash", f"${cash:,.2f}")
            metric_col3.metric("Positions", f"${positions_value:,.2f}")
            snap_ts = snapshot.get("timestamp")
            if snap_ts:
                st.caption(f"Snapshot timestamp: {snap_ts}")
            st.caption(f"Run: {run_id_short} | schema: {schema_text} | cycle: {cycle_text}")

            with st.expander("Diagnostics / Telemetry", expanded=False):
                diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
                diag_col1.slider("Recent N", min_value=10, max_value=200, key="pm_telemetry_rows")
                diag_col2.checkbox("Show events", key="pm_telemetry_show_events")
                diag_col3.selectbox("Level", ["ALL", "INFO", "WARN", "ERROR"], key="pm_telemetry_level")
                diag_col4.checkbox("Cycle range", key="pm_telemetry_cycle_filter")

                config_path = os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")
                cfg_obj = _safe_read_json(config_path)
                reporting_cfg = cfg_obj.get("reporting", {}) if isinstance(cfg_obj, dict) else {}
                if not isinstance(reporting_cfg, dict):
                    reporting_cfg = {}
                out_dir = str(reporting_cfg.get("out_dir", "outputs")).strip() or "outputs"
                metrics_path = os.path.join(out_dir, "telemetry", "metrics.jsonl")
                events_path = os.path.join(out_dir, "telemetry", "events.jsonl")
                rows_limit = int(st.session_state.get("pm_telemetry_rows", 20))
                level_filter = str(st.session_state.get("pm_telemetry_level", "ALL")).upper()

                def _safe_int(value, default=None):
                    try:
                        if value is None or value == "":
                            return default
                        return int(value)
                    except Exception:
                        return default

                def _extract_cycle_id(row):
                    if not isinstance(row, dict):
                        return None
                    cyc = _safe_int(row.get("cycle_id"), None)
                    if cyc is not None:
                        return cyc
                    payload = row.get("payload")
                    if isinstance(payload, dict):
                        cyc = _safe_int(payload.get("cycle_id"), None)
                        if cyc is not None:
                            return cyc
                    return _safe_int(row.get("cycle"), None)

                def _extract_payload(row):
                    payload = row.get("payload")
                    if isinstance(payload, dict):
                        return payload
                    return {}

                def _apply_common_filters(rows):
                    filtered = [r for r in rows if isinstance(r, dict)]
                    if run_id_text:
                        filtered = [r for r in filtered if str(r.get("run_id", "")).strip() == run_id_text]
                    if level_filter != "ALL":
                        filtered = [r for r in filtered if str(r.get("level", "INFO")).upper() == level_filter]
                    if bool(st.session_state.get("pm_telemetry_cycle_filter", False)):
                        cycles = [c for c in (_extract_cycle_id(r) for r in filtered) if c is not None]
                        if cycles:
                            min_cycle = int(min(cycles))
                            max_cycle = int(max(cycles))
                            if st.session_state.get("pm_telemetry_cycle_min", 0) < min_cycle:
                                st.session_state["pm_telemetry_cycle_min"] = min_cycle
                            if st.session_state.get("pm_telemetry_cycle_max", 0) < min_cycle:
                                st.session_state["pm_telemetry_cycle_max"] = max_cycle
                            cycle_col1, cycle_col2 = st.columns(2)
                            cycle_col1.number_input(
                                "Cycle min",
                                min_value=min_cycle,
                                max_value=max_cycle,
                                step=1,
                                key="pm_telemetry_cycle_min",
                            )
                            cycle_col2.number_input(
                                "Cycle max",
                                min_value=min_cycle,
                                max_value=max_cycle,
                                step=1,
                                key="pm_telemetry_cycle_max",
                            )
                            cycle_min = int(st.session_state.get("pm_telemetry_cycle_min", min_cycle))
                            cycle_max = int(st.session_state.get("pm_telemetry_cycle_max", max_cycle))
                            if cycle_max < cycle_min:
                                cycle_min, cycle_max = cycle_max, cycle_min
                            filtered = [
                                r for r in filtered
                                if (lambda c: c is not None and cycle_min <= c <= cycle_max)(_extract_cycle_id(r))
                            ]
                    return filtered

                metrics_rows = read_jsonl_tail(metrics_path, max_lines=5000)
                metrics_rows = _apply_common_filters(metrics_rows)
                if not run_id_text:
                    st.caption("Current snapshot has no run_id; showing all telemetry rows.")

                if not metrics_rows:
                    st.info("No telemetry yet.")
                else:
                    metrics_rows = sorted(
                        metrics_rows,
                        key=lambda r: (
                            str(r.get("ts_utc", "")),
                            _extract_cycle_id(r) if _extract_cycle_id(r) is not None else -1,
                        ),
                    )
                    metrics_rows = list(reversed(metrics_rows[-rows_limit:]))

                    flat_rows = []
                    for row in metrics_rows:
                        payload = _extract_payload(row)
                        price_stats = payload.get("price_fetch_stats", {}) if isinstance(payload.get("price_fetch_stats"), dict) else {}
                        price_diag_stats = payload.get("price_diagnostics_summary", {}) if isinstance(payload.get("price_diagnostics_summary"), dict) else {}
                        overlay_stats = payload.get("news_overlay_stats", {}) if isinstance(payload.get("news_overlay_stats"), dict) else {}
                        hits = _to_float(price_stats.get("cache_hits"), 0.0)
                        misses = _to_float(price_stats.get("cache_misses"), 0.0)
                        denom = hits + misses
                        hit_rate = (hits / denom * 100.0) if denom > 0 else None
                        overlay_cash_delta = overlay_stats.get("cash_delta")
                        if overlay_cash_delta in (None, ""):
                            overlay_cash_delta = overlay_stats.get("applied_delta")
                        if overlay_cash_delta in (None, ""):
                            overlay_cash_delta = overlay_stats.get("applied")

                        flat_rows.append(
                            {
                                "ts_utc": str(row.get("ts_utc", ""))[:19],
                                "cycle_id": _extract_cycle_id(row),
                                "total_equity": _to_float(payload.get("total_equity", row.get("total_equity", None)), None),
                                "cash_pct": _to_float(payload.get("cash_pct", row.get("cash_pct", None)), None),
                                "drawdown": _to_float(payload.get("drawdown", row.get("drawdown", None)), None),
                                "macro_risk_score": _to_float(payload.get("macro_risk_score", row.get("macro_risk_score", None)), None),
                                "price_batch_calls": _safe_int(price_stats.get("batch_calls"), None),
                                "price_hit_rate": hit_rate,
                                "price_ms": _safe_int(price_stats.get("elapsed_ms"), None),
                                "price_quality": str(price_diag_stats.get("data_quality_level", "")),
                                "price_p95_age_s": _to_float(price_diag_stats.get("p95_age_seconds", None), None),
                                "overlay_cash_delta": _to_float(overlay_cash_delta, None),
                            }
                        )

                    st.dataframe(pd.DataFrame(flat_rows), width="stretch", hide_index=True)

                price_diag = snapshot.get("price_diagnostics_summary", {})
                if isinstance(price_diag, dict) and price_diag:
                    st.markdown("**Price Diagnostics (Current Snapshot)**")
                    freshness_counts = price_diag.get("freshness_counts", {})
                    source_counts = price_diag.get("source_counts", {})
                    if not isinstance(freshness_counts, dict):
                        freshness_counts = {}
                    if not isinstance(source_counts, dict):
                        source_counts = {}
                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("LIVE", int(_to_float(freshness_counts.get("LIVE", 0), 0.0)))
                    q2.metric("RECENT", int(_to_float(freshness_counts.get("RECENT", 0), 0.0)))
                    q3.metric("STALE", int(_to_float(freshness_counts.get("STALE", 0), 0.0)))
                    q4.metric("MISSING", int(_to_float(freshness_counts.get("MISSING", 0), 0.0)))
                    data_quality_level = str(price_diag.get("data_quality_level", "OK") or "OK").upper()
                    p95_age_seconds = _to_float(price_diag.get("p95_age_seconds", None), None)
                    st.caption(
                        f"n_tickers={int(_to_float(price_diag.get('n_tickers', 0), 0.0))} | "
                        f"quality={data_quality_level} | "
                        f"tz_ok_false={int(_to_float(price_diag.get('tz_ok_false', 0), 0.0))} | "
                        f"p95_age_seconds={p95_age_seconds} | "
                        f"max_age_seconds={_to_float(price_diag.get('max_age_seconds', None), None)}"
                    )
                    if source_counts:
                        source_df = pd.DataFrame(
                            [{"source": str(k), "count": int(_to_float(v, 0.0))} for k, v in source_counts.items()]
                        ).sort_values("count", ascending=False)
                        st.dataframe(source_df, width="stretch", hide_index=True)
                    stale_list = price_diag.get("stale_by_age", [])
                    if not isinstance(stale_list, list) or not stale_list:
                        stale_list = price_diag.get("stale_tickers", [])
                    if isinstance(stale_list, list) and stale_list:
                        st.caption("stale_by_age (top 10)")
                        st.dataframe(pd.DataFrame(stale_list), width="stretch", hide_index=True)
                    missing_list = price_diag.get("missing_tickers", [])
                    if isinstance(missing_list, list) and missing_list:
                        st.caption("missing_tickers (top 10)")
                        st.dataframe(pd.DataFrame(missing_list), width="stretch", hide_index=True)
                    tz_bad_list = price_diag.get("tz_bad_tickers", [])
                    if isinstance(tz_bad_list, list) and tz_bad_list:
                        st.caption("tz_bad_tickers (top 10)")
                        st.dataframe(pd.DataFrame({"ticker": tz_bad_list[:10]}), width="stretch", hide_index=True)

                if bool(st.session_state.get("pm_telemetry_show_events", False)):
                    events_rows = read_jsonl_tail(events_path, max_lines=5000)
                    events_rows = _apply_common_filters(events_rows)
                    if not events_rows:
                        st.info("No events telemetry yet.")
                    else:
                        events_rows = sorted(
                            events_rows,
                            key=lambda r: (
                                str(r.get("ts_utc", "")),
                                _extract_cycle_id(r) if _extract_cycle_id(r) is not None else -1,
                            ),
                        )
                        events_rows = list(reversed(events_rows[-max(rows_limit, 50):]))

                        event_table_rows = []
                        for row in events_rows:
                            event_table_rows.append(
                                {
                                    "ts_utc": str(row.get("ts_utc", ""))[:19],
                                    "cycle_id": _extract_cycle_id(row),
                                    "event": str(row.get("event", "")),
                                    "status": str(row.get("status", "")),
                                    "duration_ms": _to_float(row.get("duration_ms", None), None),
                                    "level": str(row.get("level", "")),
                                    "message": str(row.get("message", "")),
                                }
                            )
                        st.dataframe(pd.DataFrame(event_table_rows), width="stretch", hide_index=True)

                        with st.expander("Event payload details", expanded=False):
                            for row in events_rows[:3]:
                                payload = _extract_payload(row)
                                if not payload:
                                    continue
                                st.markdown(
                                    f"**{row.get('event', '-') or '-'} | cycle={_extract_cycle_id(row)} | ts={row.get('ts_utc', '-') or '-'}**"
                                )
                                st.json(payload)

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

            theme_slot = st.container()
            equity_slot = st.container()
            trade_history_slot = st.container()

            with trade_history_slot:
                st.subheader("Trade History")
                st.caption("Updates on app rerun. Enable Auto refresh or click Refresh data for near real-time updates.")
                if trades:
                    trade_rows = []
                    for row_idx, trade in enumerate(trades):
                        timestamp = trade.get("timestamp") or trade.get("time") or trade.get("datetime") or ""
                        ticker = str(trade.get("ticker", ""))
                        side = str(trade.get("side", trade.get("direction", "")))
                        amount = _to_float(
                            trade.get("cost", trade.get("notional", trade.get("amount", trade.get("desired_trade_value", 0.0))))
                        )
                        weight_change = None
                        raw_weight_change = trade.get("weight_change")
                        if raw_weight_change not in (None, ""):
                            weight_change = _to_float(raw_weight_change)
                        else:
                            old_weight_raw = trade.get(
                                "old_target_weight",
                                trade.get("current_weight", trade.get("old_weight", None)),
                            )
                            new_weight_raw = trade.get(
                                "new_target_weight",
                                trade.get("target_weight", trade.get("new_weight", None)),
                            )
                            if old_weight_raw not in (None, "") or new_weight_raw not in (None, ""):
                                old_weight = _to_float(old_weight_raw)
                                new_weight = _to_float(new_weight_raw)
                                if abs(old_weight) > 1e-12 or abs(new_weight) > 1e-12:
                                    weight_change = new_weight - old_weight

                        if weight_change is None:
                            equity_reference = _to_float(
                                trade.get("equity_reference", trade.get("equity_before_trade", 0.0))
                            )
                            if equity_reference <= 0 and total_equity > 0:
                                equity_reference = total_equity
                            notional = _to_float(
                                trade.get("notional", trade.get("amount", trade.get("desired_trade_value", 0.0)))
                            )
                            side_upper = side.upper()
                            if equity_reference > 0 and abs(notional) > 0 and side_upper in ("BUY", "SELL"):
                                signed_notional = abs(notional) if side_upper == "BUY" else -abs(notional)
                                weight_change = signed_notional / equity_reference

                        trade_rows.append(
                            {
                                "time": str(timestamp),
                                "ticker": ticker,
                                "side": side,
                                "amount": amount,
                                "weight_change": weight_change if weight_change is not None else "N/A",
                                "_row_order": int(row_idx),
                            }
                        )

                    trade_df = pd.DataFrame(trade_rows)
                    if not trade_df.empty:
                        time_raw = trade_df["time"].astype(str).str.strip()
                        try:
                            # pandas>=2 supports mixed-format parsing and handles timezone/no-timezone rows together.
                            trade_df["time_sort"] = pd.to_datetime(
                                time_raw, errors="coerce", utc=True, format="mixed"
                            )
                        except Exception:
                            trade_df["time_sort"] = pd.to_datetime(time_raw, errors="coerce", utc=True)

                        # Row-wise fallback for entries that still fail vector parsing (e.g. mixed tz edge cases).
                        if trade_df["time_sort"].isna().any():
                            def _parse_trade_ts_single(value):
                                try:
                                    text = str(value).strip()
                                    if not text:
                                        return pd.NaT
                                    ts = pd.Timestamp(text.replace("Z", "+00:00"))
                                    if ts.tzinfo is None:
                                        return ts.tz_localize("UTC")
                                    return ts.tz_convert("UTC")
                                except Exception:
                                    return pd.NaT

                            fallback_ts = time_raw.map(_parse_trade_ts_single)
                            trade_df["time_sort"] = trade_df["time_sort"].fillna(fallback_ts)
                        trade_df["time_str"] = trade_df["time"].astype(str)
                        if trade_df["time_sort"].notna().any():
                            trade_df = trade_df.sort_values(
                                ["time_sort", "time_str", "_row_order"],
                                ascending=[False, False, False],
                                na_position="last",
                                kind="mergesort",
                            )
                        else:
                            # If timestamps are not parseable, fall back to ISO string + read-order newest-first.
                            trade_df = trade_df.sort_values(
                                ["time_str", "_row_order"],
                                ascending=[False, False],
                                kind="mergesort",
                            )
                        trade_df = trade_df.drop(columns=["time_sort", "time_str", "_row_order"]).reset_index(drop=True)
                        st.dataframe(trade_df, width="stretch", hide_index=True)
                else:
                    st.info("No trade history found in outputs/trade_history.jsonl")

            def _render_reports_statistics_section():
                st.subheader("Reports / Statistics")
                window_defs = [
                    ("Today (1D)", 1),
                    ("3 Days (3D)", 3),
                    ("1 Week (7D)", 7),
                    ("1 Month (30D)", 30),
                    ("3 Months (90D)", 90),
                    ("Half Year (180D)", 180),
                    ("1 Year (365D)", 365),
                ]
                if not daily_report_entries:
                    st.info("No Daily Report index found in outputs/Daily Report/daily_reports_index.json")
                    return

                tabs = st.tabs([x[0] for x in window_defs])
                for tab, (_label, window_days) in zip(tabs, window_defs):
                    with tab:
                        if len(daily_report_entries) < window_days:
                            st.info("Time insufficient")
                            continue

                        selected_entries = daily_report_entries[:window_days]
                        selected_paths = []
                        for entry in selected_entries:
                            path = str(entry.get("path", "")).strip()
                            if path:
                                selected_paths.append(path)
                        reports = load_daily_reports_by_paths(selected_paths, interval_seconds, refresh_nonce)
                        if len(reports) < window_days:
                            st.info("Time insufficient")
                            continue

                        if daily_reporter is None:
                            st.info("Time insufficient")
                            continue

                        try:
                            agg = daily_reporter.aggregate_reports(reports, window_days)
                        except Exception:
                            agg = {"status": "insufficient"}

                        if not isinstance(agg, dict) or agg.get("status") != "ok":
                            st.info("Time insufficient")
                            continue

                        agg_quality = str(agg.get("data_quality", "ok")).strip().lower()
                        if agg_quality == "inconsistent":
                            st.warning("Data inconsistent")
                            issues = agg.get("issues", [])
                            if isinstance(issues, list) and issues:
                                for issue in issues[:3]:
                                    st.caption(f"- {issue}")
                            continue

                        metrics = agg.get("metrics", {}) if isinstance(agg.get("metrics"), dict) else {}
                        buy_notional = _to_float(metrics.get("buy_notional"), 0.0)
                        sell_notional = _to_float(metrics.get("sell_notional"), 0.0)
                        net_flow = _to_float(metrics.get("net_flow"), 0.0)
                        trades_count = int(_to_float(metrics.get("trade_count"), 0.0))
                        pnl_value = metrics.get("pnl")
                        pnl_pct_value = metrics.get("pnl_pct")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Buy Notional", f"${buy_notional:,.2f}")
                        m2.metric("Sell Notional", f"${sell_notional:,.2f}")
                        m3.metric("Net Flow", f"${net_flow:,.2f}")
                        m4.metric("Trade Count", f"{trades_count}")

                        pnl_text = "N/A" if pnl_value is None else f"${_to_float(pnl_value):,.2f}"
                        pnl_pct_text = "N/A" if pnl_pct_value is None else f"{_to_float(pnl_pct_value):.2f}%"
                        p1, p2, p3 = st.columns([1, 1, 2])
                        p1.metric("PnL", pnl_text)
                        p2.metric("PnL %", pnl_pct_text)
                        p3.caption(
                            f"Range: {agg.get('from', 'N/A')} -> {agg.get('to', 'N/A')} | "
                            f"Reports: {int(_to_float(agg.get('report_count'), 0.0))}"
                        )

                        st.markdown("**Top Risky Tickers**")
                        risky_rows = []
                        risky_list = agg.get("top_risky_tickers", []) if isinstance(agg.get("top_risky_tickers"), list) else []
                        for item in risky_list:
                            if not isinstance(item, dict):
                                continue
                            risky_rows.append(
                                {
                                    "ticker": str(item.get("ticker", "")),
                                    "count": int(_to_float(item.get("count"), 0.0)),
                                    "avg_score": _to_float(item.get("avg_score"), 0.0),
                                }
                            )
                        if risky_rows:
                            st.dataframe(pd.DataFrame(risky_rows), width="stretch", hide_index=True)
                        else:
                            st.caption("No risky ticker statistics.")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Long-term Conviction**")
                            long_rows = agg.get("long_term_stats", []) if isinstance(agg.get("long_term_stats"), list) else []
                            if long_rows:
                                for row in long_rows[:5]:
                                    if not isinstance(row, dict):
                                        continue
                                    st.markdown(
                                        f"- `{row.get('ticker', '')}` x{int(_to_float(row.get('count'), 0.0))}: "
                                        f"{row.get('last_why', '')}"
                                    )
                            else:
                                st.caption("No long-term conviction entries.")
                        with c2:
                            st.markdown("**Short-term Conviction**")
                            short_rows = agg.get("short_term_stats", []) if isinstance(agg.get("short_term_stats"), list) else []
                            if short_rows:
                                for row in short_rows[:5]:
                                    if not isinstance(row, dict):
                                        continue
                                    st.markdown(
                                        f"- `{row.get('ticker', '')}` x{int(_to_float(row.get('count'), 0.0))}: "
                                        f"{row.get('last_why', '')}"
                                    )
                            else:
                                st.caption("No short-term conviction entries.")
            with theme_slot:
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
    
            with equity_slot:
                st.subheader("Equity Curve")
                equity_history = snapshot.get("equity_history")
                if not isinstance(equity_history, list) or not equity_history:
                    st.info("No equity_history found in outputs/snapshot_live.json")
                else:
                    window_options = [6, 12, 24, 48, 168]
                    resample_options = ["raw", "1min", "5min", "15min", "1h", "1d"]
                    x_tick_mode_options = ["auto", "fixed"]
                    x_tick_options = [5, 15, 30, 60]
                    y_mode_options = ["auto", "manual"]
                    y_metric_options = ["equity", "pnl_from_initial"]
    
                    if int(st.session_state.get("pm_window_hours", 48)) not in window_options:
                        st.session_state["pm_window_hours"] = 48
                    if str(st.session_state.get("pm_resample_rule", "15min")) not in resample_options:
                        st.session_state["pm_resample_rule"] = "15min"
                    if str(st.session_state.get("pm_x_tick_mode", "auto")) not in x_tick_mode_options:
                        st.session_state["pm_x_tick_mode"] = "auto"
                    if int(st.session_state.get("pm_x_tick_minutes", 15)) not in x_tick_options:
                        st.session_state["pm_x_tick_minutes"] = 15
                    if str(st.session_state.get("pm_y_mode", "auto")) not in y_mode_options:
                        st.session_state["pm_y_mode"] = "auto"
                    if str(st.session_state.get("pm_y_metric", "equity")) not in y_metric_options:
                        st.session_state["pm_y_metric"] = "equity"
    
                    control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns(5)
                    control_col1.selectbox(
                        "Window (hours)",
                        window_options,
                        index=window_options.index(int(st.session_state.get("pm_window_hours", 48))),
                        key="pm_window_hours",
                    )
                    control_col2.selectbox(
                        "Resample",
                        resample_options,
                        index=resample_options.index(str(st.session_state.get("pm_resample_rule", "15min"))),
                        key="pm_resample_rule",
                    )
                    control_col3.selectbox(
                        "X Tick Mode",
                        x_tick_mode_options,
                        index=x_tick_mode_options.index(str(st.session_state.get("pm_x_tick_mode", "auto"))),
                        key="pm_x_tick_mode",
                    )
                    control_col4.selectbox(
                        "X Tick (min)",
                        x_tick_options,
                        index=x_tick_options.index(int(st.session_state.get("pm_x_tick_minutes", 15))),
                        key="pm_x_tick_minutes",
                    )
                    control_col5.checkbox(
                        "Trading hours only (06:00–13:00)",
                        key="pm_hide_off_hours",
                    )
    
                    y_ctrl_col1, y_ctrl_col2, y_ctrl_col3, y_ctrl_col4 = st.columns(4)
                    y_ctrl_col1.selectbox(
                        "Y Mode",
                        y_mode_options,
                        index=y_mode_options.index(str(st.session_state.get("pm_y_mode", "auto"))),
                        key="pm_y_mode",
                    )
                    y_ctrl_col2.selectbox(
                        "Y Metric",
                        y_metric_options,
                        index=y_metric_options.index(str(st.session_state.get("pm_y_metric", "equity"))),
                        key="pm_y_metric",
                    )
    
                    history_payload = json.dumps(equity_history, ensure_ascii=False, sort_keys=True)
                    equity_df = _prepare_equity_curve_cached(
                        history_payload,
                        int(st.session_state.get("pm_window_hours", 48)),
                        str(st.session_state.get("pm_resample_rule", "15min")),
                    )
    
                    if equity_df.empty:
                        st.info("No valid equity points in selected window.")
                    else:
                        first_equity = _to_float(equity_df["equity"].iloc[0], 0.0)
                        initial_equity = _infer_initial_equity(snapshot, summary_text, first_equity)
                        y_metric = str(st.session_state.get("pm_y_metric", "equity"))
                        y_series = equity_df["equity"] if y_metric == "equity" else (equity_df["equity"] - initial_equity)
                        plot_df = pd.DataFrame({"time": equity_df["time"], "y": y_series})
    
                        y_min_default = float(plot_df["y"].min())
                        y_max_default = float(plot_df["y"].max())
                        if y_max_default <= y_min_default:
                            y_max_default = y_min_default + 1.0
    
                        if not st.session_state.get("pm_y_bounds_initialized", False):
                            st.session_state["pm_y_min"] = y_min_default
                            st.session_state["pm_y_max"] = y_max_default
                            st.session_state["pm_y_bounds_initialized"] = True
    
                        manual_inputs_disabled = str(st.session_state.get("pm_y_mode", "auto")) != "manual"
                        y_min = y_ctrl_col3.number_input(
                            "Y Min",
                            value=float(st.session_state.get("pm_y_min", y_min_default)),
                            step=1.0,
                            disabled=manual_inputs_disabled,
                            key="pm_y_min",
                        )
                        y_max = y_ctrl_col4.number_input(
                            "Y Max",
                            value=float(st.session_state.get("pm_y_max", y_max_default)),
                            step=1.0,
                            disabled=manual_inputs_disabled,
                            key="pm_y_max",
                        )
    
                        y_opt_col1, _ = st.columns(2)
                        y_dtick = y_opt_col1.number_input(
                            "Y dtick (0=auto)",
                            value=float(st.session_state.get("pm_y_dtick", 0.0)),
                            step=1.0,
                            disabled=manual_inputs_disabled,
                            key="pm_y_dtick",
                        )
    
                        y_title = "PnL ($)" if y_metric == "pnl_from_initial" else "Equity ($)"
                        x_tick_mode = str(st.session_state.get("pm_x_tick_mode", "auto"))
                        x_tick_minutes = int(st.session_state.get("pm_x_tick_minutes", 15))
                        pm_window_hours = int(st.session_state.get("pm_window_hours", 48))
                        pm_resample_rule = str(st.session_state.get("pm_resample_rule", "15min"))
                        pm_hide_off_hours = bool(st.session_state.get("pm_hide_off_hours", True))
                        pm_y_mode = str(st.session_state.get("pm_y_mode", "auto"))
                        pm_y_metric = str(st.session_state.get("pm_y_metric", "equity"))
                        pm_uirev = (
                            f"pm:{pm_window_hours}:{pm_resample_rule}:{x_tick_mode}:{x_tick_minutes}:"
                            f"{int(pm_hide_off_hours)}:{pm_y_mode}:{pm_y_metric}"
                        )
                        if pm_y_mode == "manual":
                            pm_uirev += (
                                f":{float(st.session_state.get('pm_y_min', y_min)):.6f}:"
                                f"{float(st.session_state.get('pm_y_max', y_max)):.6f}:"
                                f"{float(st.session_state.get('pm_y_dtick', y_dtick)):.6f}"
                            )
                        x_tick_label = "auto" if x_tick_mode == "auto" else f"{x_tick_minutes}min"
                        y_mode_label = "auto"
                        if str(st.session_state.get("pm_y_mode", "auto")) == "manual":
                            y_mode_label = f"manual [{y_min:.2f}, {y_max:.2f}]"
                        st.caption(
                            "X: last "
                            f"{pm_window_hours}h @ {pm_resample_rule}"
                            f" | X tick: {x_tick_label} | Y: {y_metric} {y_mode_label}"
                        )
    
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
                                dtick_ms = x_tick_minutes * 60 * 1000
                                time_span = (plot_df["time"].max() - plot_df["time"].min()).total_seconds()
                                tick_fmt = "%m-%d %H:%M" if time_span <= 7 * 24 * 3600 else "%m-%d"
                                xaxis_config.update(
                                    {
                                        "tickmode": "linear",
                                        "dtick": dtick_ms,
                                        "tickformat": tick_fmt,
                                    }
                                )
                            if bool(st.session_state.get("pm_hide_off_hours", True)):
                                xaxis_config["rangebreaks"] = [{"bounds": [13, 6], "pattern": "hour"}]
    
                            yaxis_config = {"title": y_title}
                            if str(st.session_state.get("pm_y_mode", "auto")) == "manual":
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
                                uirevision=pm_uirev,
                            )
                            st.plotly_chart(fig, width="stretch", key="pm_equity_curve")
                            plot_rendered = True
                        except Exception:
                            plot_rendered = False
    
                        if not plot_rendered:
                            try:
                                import altair as alt
    
                                y_scale = alt.Scale()
                                if str(st.session_state.get("pm_y_mode", "auto")) == "manual":
                                    y_scale = alt.Scale(domain=[float(min(y_min, y_max)), float(max(y_min, y_max))])
    
                                x_axis = alt.Axis(title="Time")
                                if x_tick_mode == "fixed":
                                    tick_count = max(
                                        2,
                                        int(
                                            (int(st.session_state.get("pm_window_hours", 48)) * 60)
                                            / max(1, x_tick_minutes)
                                        ),
                                    )
                                    x_axis = alt.Axis(title="Time", tickCount=tick_count)
    
                                y_axis = alt.Axis(title=y_title)
                                if str(st.session_state.get("pm_y_mode", "auto")) == "manual" and y_dtick and y_dtick > 0:
                                    y_axis = alt.Axis(title=y_title, tickMinStep=float(y_dtick))
    
                                chart = (
                                    alt.Chart(plot_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("time:T", axis=x_axis),
                                        y=alt.Y("y:Q", axis=y_axis, scale=y_scale),
                                        tooltip=[
                                            alt.Tooltip("time:T", title="Time"),
                                            alt.Tooltip("y:Q", title=y_title, format=",.2f"),
                                        ],
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

            _render_reports_statistics_section()

    auto_refresh = bool(st.session_state.get("pm_auto_refresh_data", False))
    interval_seconds = int(st.session_state.get("pm_auto_refresh_interval", 60))

    if auto_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{interval_seconds}s")
        def _portfolio_refresh_fragment():
            _render_data_panel()

        _portfolio_refresh_fragment()
    else:
        if auto_refresh and not hasattr(st, "fragment"):
            st.info("Auto refresh requires Streamlit fragment support. Use Refresh data if unavailable.")
        _render_data_panel()
_industry_cli_code = _run_industry_news_cli_if_requested()
if _industry_cli_code is not None:
    raise SystemExit(_industry_cli_code)

_taxonomy_cli_code = _run_taxonomy_cli_if_requested()
if _taxonomy_cli_code is not None:
    raise SystemExit(_taxonomy_cli_code)

_sanity_cli_code = _run_industry_sanity_cli_if_requested()
if _sanity_cli_code is not None:
    raise SystemExit(_sanity_cli_code)

_runtime_once_cli_code = _run_industry_runtime_once_cli_if_requested()
if _runtime_once_cli_code is not None:
    raise SystemExit(_runtime_once_cli_code)

_runtime_once_debug_cli_code = _run_industry_runtime_once_debug_cli_if_requested()
if _runtime_once_debug_cli_code is not None:
    raise SystemExit(_runtime_once_debug_cli_code)

_debug_one_bucket_cli_code = _run_debug_industry_one_bucket_cli_if_requested()
if _debug_one_bucket_cli_code is not None:
    raise SystemExit(_debug_one_bucket_cli_code)

st.set_page_config(page_title="GlobalWatch DeepSeek Edition", layout="wide", page_icon=":satellite:")

st.sidebar.header("Settings")
st.sidebar.caption(f"Brain: {LOCAL_MODEL}")

page_options = ["\U0001F4E1 Global Macro Signals", "\U0001F4BC Portfolio Monitor"]
if "page" not in st.session_state or st.session_state.get("page") not in page_options:
    st.session_state["page"] = page_options[0]
page_choice = st.sidebar.selectbox(
    "Page",
    page_options,
    index=page_options.index(st.session_state.get("page", page_options[0])),
    key="page",
)

# Macro rules reference
with st.sidebar.expander("Macro Rules Library"):
    st.text(MACRO_LOGIC_KNOWLEDGE)

lang_mode = st.sidebar.radio("Language", ["English"], index=0)
refresh_label = st.sidebar.selectbox("Refresh Rate", list(REFRESH_OPTIONS.keys()), index=0)
refresh_sec = REFRESH_OPTIONS[refresh_label]
enable_toast = st.sidebar.checkbox("Desktop Notify", value=True)
auto_run = st.sidebar.checkbox("Auto Run", value=True)

if 'last_run' not in st.session_state: st.session_state['last_run'] = datetime.now() - timedelta(days=1)

if page_choice == "\U0001F4BC Portfolio Monitor":
    render_portfolio_monitor()
    st.stop()

st.title("GlobalWatch: DeepSeek-R1 Reasoning Edition")
st.caption("Powered by Chain-of-Thought Reasoning")
st.divider()

tab_macro, tab_stock, tab_scoreboard, tab_warning = st.tabs(["Macro / FX", "US Stocks", "Signal Scoreboard", "Early-Warning"])

# === TAB 1: Macro / FX ===
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
                if b1 != "USD (US Dollar)": plot_candle_chart(ASSETS_DB[b1]['ticker'], b1)
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

    available_pairs = list(dict.fromkeys(user_pairs))
    current_selected_pairs = st.session_state.get("macro_selected_pairs", [])
    if not isinstance(current_selected_pairs, list):
        current_selected_pairs = []
    current_selected_pairs = [p for p in current_selected_pairs if p in available_pairs]
    if not current_selected_pairs:
        current_selected_pairs = available_pairs[:]
    st.session_state["macro_selected_pairs"] = current_selected_pairs

    selected_pairs = st.multiselect(
        "Pairs to Analyze (Only selected pairs will be reasoned)",
        options=available_pairs,
        key="macro_selected_pairs",
    )
    selected_scope = "FX_PAIR" if selected_pairs else "MACRO_OVERVIEW"
    selected_targets = selected_pairs[:] if selected_pairs else ["US_MACRO"]
    user_question = st.text_input(
        "Question for selected target(s) (optional)",
        value="",
        key="interactive_user_question",
        help="Interactive analysis focuses only on selected targets and avoids broad bucket recompute.",
    )
    use_targeted_news = st.checkbox(
        "Targeted RSS filter (recommended)",
        value=True,
        key="macro_targeted_news",
    )
    run_topic_signals_now = st.checkbox(
        "Update Topic Signals (extra ~10-20s)",
        value=False,
        key="macro_run_topic_signals",
    )
    request_think_macro = st.checkbox(
        "Request <think> reasoning block (model-dependent)",
        value=False,
        key="req_think_macro",
    )
    update_industry_buckets_now = st.checkbox(
        "Include Industry Buckets (slow)",
        value=False,
        key="interactive_run_industry_batch",
        help="Disabled by default to keep interactive analysis fast. Batch updates can run in background.",
    )

    delta = (datetime.now() - st.session_state['last_run']).total_seconds()
    remain = max(0, refresh_sec - delta) if refresh_sec > 0 else 0
    
    run_trigger = st.button("Run Deep Reason Analysis") or (refresh_sec > 0 and remain == 0 and auto_run)
    if run_trigger:
        if not selected_pairs:
            st.warning("Please select at least 1 pair.")
        else:
            with st.status("DeepSeek is thinking...", expanded=True) as s:
                t0 = time.perf_counter()
                s.write("step=1 start get_rss_news")
                news_raw = get_rss_news()
                t1 = time.perf_counter()

                s.write("step=2 start targeted_filter")
                if use_targeted_news:
                    keywords = build_keywords_for_assets([b1, q1, b2, q2], selected_pairs)
                    news = filter_news_by_keywords(news_raw, keywords, min_keep=4, max_keep=12)
                else:
                    keywords = []
                    news = list(news_raw)
                t2 = time.perf_counter()

                s.write(f"[UI] rss_raw={len(news_raw)} filtered={len(news)} pairs={selected_pairs}")
                if use_targeted_news:
                    s.write(f"[UI] targeted_keywords={keywords[:12]}")

                s.write("step=3 start analyze_all")
                res = analyze_all(
                    news,
                    selected_pairs,
                    macro,
                    lang_mode,
                    selected_scope=selected_scope,
                    selected_targets=selected_targets,
                    user_question=user_question,
                    latest_industry_signals_summary=_get_latest_industry_signals_summary(
                        config_path=os.environ.get("PAPER_CONFIG_PATH", "paper_config.json"),
                        limit=12,
                    ),
                    request_think=bool(request_think_macro),
                    run_topic_signals=bool(run_topic_signals_now),
                    run_industry_runtime=bool(update_industry_buckets_now),
                    industry_config_path=os.environ.get("PAPER_CONFIG_PATH", "paper_config.json"),
                )
                t3 = time.perf_counter()
                s.write(
                    f"[UI] elapsed_fetch={t1 - t0:.2f}s "
                    f"elapsed_filter={t2 - t1:.2f}s "
                    f"elapsed_llm={t3 - t2:.2f}s"
                )
                s.write(
                    f"[UI] topic_recorded={int(res.get('_topic_signals_recorded', 0) or 0)} "
                    f"industry_status={res.get('_industry_news_status', 'skipped')} "
                    f"industry_written={int(res.get('_industry_news_written', 0) or 0)}"
                )
                if enable_toast and res.get("status") == "alert" and res.get("impact_score", 0) >= 7:
                    send_notification("Market Alert", res.get("summary"))

                st.session_state['last_run'] = datetime.now()
                st.session_state['res'] = res
                st.session_state['news'] = news
                s.update(label="Reasoning Complete", state="complete", expanded=False)
                st.rerun()

    if 'res' in st.session_state:
        res = st.session_state['res']
        
        # Parse error handling
        if res.get('_parse_error'):
            st.error("AI Output Parsing Error")
            st.markdown(f"**Reason**: {res.get('reason', 'Unknown error')}")
            
            with st.expander("Raw Output (Debug)", expanded=False):
                st.code(res.get('raw_output', 'No output available'), language="text")
            
            st.warning("The AI failed to generate valid JSON output. This may be due to:")
            st.markdown("""
            - Model output format issues
            - Context length exceeded
            - Unexpected model behavior
            
            **Suggested actions**:
            - Try again with a different model
            - Reduce the number of news items
            - Check Ollama logs for errors
            """)
            
            thought_text = str(res.get('thought_process', '') or '').strip()
            if thought_text:
                with st.expander("DeepSeek Thought Process (Click to expand)", expanded=False):
                    st.markdown(thought_text)
            else:
                with st.expander("Reasoning Summary", expanded=False):
                    st.markdown(build_reasoning_summary(res))
        # ================================
        
        # V3 thought process rendering
        elif res.get("status") != "error":
            thought_text = str(res.get('thought_process', '') or '').strip()
            if thought_text:
                with st.expander("DeepSeek Thought Process (Click to expand)", expanded=False):
                    st.markdown(thought_text)
            else:
                with st.expander("Reasoning Summary", expanded=False):
                    st.markdown(build_reasoning_summary(res))
        # ==========================

        if res.get("status") == "alert":
            st.error(f"ALERT (Score: {res.get('impact_score')})")
            st.markdown(f"**Event**: {res.get('summary')}")
            
            # Evidence chain
            evidence = res.get('evidence', [])
            valid_count = res.get('_valid_evidence_count', 0)
            
            if evidence:
                with st.expander(f"Evidence Chain ({valid_count}/{len(evidence)} valid)", expanded=True):
                    for idx, ev in enumerate(evidence, 1):
                        is_invalid = ev.get('_invalid', False)
                        icon = "WARN" if is_invalid else "OK"
                        
                        st.markdown(f"**{icon} Evidence {idx}**")
                        st.markdown(f"- **Source**: {ev.get('source', 'Unknown')}")
                        st.markdown(f"- **Headline**: _{ev.get('headline', 'N/A')}_")
                        st.markdown(f"- **Why it matters**: {ev.get('why_it_matters', 'N/A')}")
                        
                        if is_invalid:
                            st.warning(ev.get('_warning', 'Invalid evidence'))
                        st.divider()
            
            if res.get('_evidence_warning'):
                st.warning("No valid evidence found. AI predictions may be unreliable.")
            # ================================
            
            col_p, col_a = st.columns(2)
            col_p.write(res.get("predictions"))
            col_a.warning(res.get("advice"))
        else:
            st.success("Market is Stable")
            st.caption(res.get("advice"))
        
        with st.expander("News Sources"):
            news_list = st.session_state.get('news', [])
            if news_list:
                for idx, news_item in enumerate(news_list, 1):
                    # Structured news rendering
                    source = news_item.get('source', 'Unknown')
                    title = news_item.get('title', 'N/A')
                    published = news_item.get('published', None)
                    link = news_item.get('link', '')
                    
                    # Timestamp rendering
                    time_str = ""
                    if published:
                        try:
                            # Human-readable UTC time
                            from datetime import datetime
                            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                            time_str = f" {dt.strftime('%Y-%m-%d %H:%M UTC')}"
                        except Exception as e:
                            time_str = f" {published}"
                    
                    # Render one news item
                    st.markdown(f"**{idx}. [{source}]** {title}")
                    if time_str:
                        st.caption(time_str)
                    if link:
                        st.markdown(f"[Read More]({link})")
                    st.divider()
            else:
                st.caption("No news available")

        with st.expander("Industry Signal Sanity Check", expanded=False):
            config_path = os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")
            try:
                sanity_report = build_industry_sanity_check_report(config_path=config_path)
                sanity_df = sanity_report.get("df", pd.DataFrame())
                sanity_summary = sanity_report.get("summary", {})

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Available Buckets", int(_safe_float(sanity_summary.get("available_buckets", 0), 0)))
                s2.metric("Missing Buckets", int(_safe_float(sanity_summary.get("missing_buckets", 0), 0)))
                s3.metric("Overweight Ratio", f"{100.0 * _safe_float(sanity_summary.get('pct_overweight', 0.0), 0.0):.1f}%")
                s4.metric("Risk Delta Std", f"{_safe_float(sanity_summary.get('std_local_risk_delta', 0.0), 0.0):.4f}")

                st.caption(
                    f"Uniformity={bool(sanity_summary.get('FLAG_UNIFORMITY', False))} | "
                    f"RateSensitiveMismatch={bool(sanity_summary.get('FLAG_RATE_SENSITIVE_MISMATCH', False))} | "
                    f"ConsumerMismatch={bool(sanity_summary.get('FLAG_CONSUMER_MISMATCH', False))} | "
                    f"Corr(local,baseline)={sanity_summary.get('correlation_local_vs_baseline')}"
                )

                if isinstance(sanity_df, pd.DataFrame) and not sanity_df.empty:
                    view_cols = [
                        "bucket",
                        "local_direction",
                        "local_risk_delta",
                        "local_confidence",
                        "local_timestamp",
                        "baseline_direction",
                        "baseline_risk_delta",
                        "baseline_confidence",
                        "diff_risk_delta",
                        "diff_confidence",
                        "direction_match",
                        "effective_cash_impact",
                        "flags",
                    ]
                    view_cols = [c for c in view_cols if c in sanity_df.columns]
                    st.dataframe(sanity_df[view_cols], width="stretch")
                else:
                    st.info("No industry signal rows available.")
            except Exception as e:
                st.error(f"[SANITY] failed to build report: {e}")

# === TAB 2: Stock Analysis ===
with tab_stock:
    st.header("US Stock Deep Dive")
    c_in, c_go = st.columns([3, 1])
    ticker = c_in.text_input("Ticker", value="NVDA").upper()
    request_think_stock = st.checkbox(
        "Request <think> reasoning block (model-dependent)",
        value=False,
        key="req_think_stock",
    )
    
    if c_go.button("Analyze"):
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
                    
                    analysis = analyze_single_stock(
                        ticker,
                        stock_news,
                        lang_mode,
                        request_think=bool(request_think_stock),
                    )
                    thought_text_stock = str(analysis.get('thought_process', '') or '').strip()
                    if thought_text_stock:
                        with st.expander("DeepSeek Thought Process (Stock)", expanded=False):
                            st.markdown(thought_text_stock)
                    else:
                        with st.expander("Reasoning Summary", expanded=False):
                            st.markdown(build_reasoning_summary(analysis))
                    
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
    st.header("Signal Scoreboard - Performance Tracking")
    st.caption("Track the accuracy and profitability of AI predictions over time")
    
    # 
    col_refresh, col_info = st.columns([1, 3])
    if col_refresh.button(" Update Results"):
        with st.spinner("Backfilling signal results..."):
            updated = backfill_signal_results()
            if updated:
                st.success(f"Updated {updated} signals")
            else:
                st.info("No signals to update")
            st.rerun()
    
    col_info.caption("Click to check and update signal results based on actual market movements")
    
    st.divider()
    
    # ?
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
    
    # 
    stats = get_signal_statistics(theme=theme, timeframe=timeframe)
    
    # 
    st.subheader(" Key Metrics")
    
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
    
    # ?
    accuracy = stats['accuracy']
    accuracy_delta = accuracy - 50  # ?
    col3.metric(
        "Accuracy",
        f"{accuracy:.1f}%",
        f"{accuracy_delta:+.1f}% vs random",
        delta_color="normal" if accuracy_delta > 0 else "inverse"
    )
    
    # 
    avg_return = stats['avg_return']
    col4.metric(
        "Avg Return",
        f"{avg_return:+.2f}%",
        "per signal",
        delta_color="normal" if avg_return > 0 else "inverse"
    )
    
    st.divider()
    
    # ==========  ==========
    st.subheader(" Trading-Grade Performance Classification")
    st.caption("  real-money execution ")
    
    # ?
    classification = classify_trading_performance(
        stats, 
        theme=theme, 
        asset=None,  # 
        transaction_cost=0.1,  # 0.1% 
        max_dd_threshold=15.0  # 15% ?
    )
    
    # 
    col_class, col_decision = st.columns([2, 1])
    
    with col_class:
        # 
        class_v2 = classification['classification_v2']
        
        # 
        if "" in class_v2:
            st.success(f"### {class_v2}")
        elif "" in class_v2:
            st.warning(f"### {class_v2}")
        elif "" in class_v2:
            st.warning(f"### {class_v2}")
        elif "" in class_v2:
            st.error(f"### {class_v2}")
        else:
            st.info(f"### {class_v2}")
    
    with col_decision:
        # 
        if classification['decision_allowed']:
            st.success("### TRADABLE")
            st.caption("")
        else:
            st.error("###  NOT TRADABLE")
            st.caption("")
    
    # 
    with st.expander(" Classification Details", expanded=True):
        st.markdown("**?*")
        st.info(classification['reason_summary'])
        
        # 
        if classification['risk_warnings']:
            st.markdown("**?*")
            for warning in classification['risk_warnings']:
                st.markdown(f"- {warning}")
        
        # 
        st.markdown("**?*")
        col_i1, col_i2, col_i3 = st.columns(3)
        
        col_i1.metric(
            "Net Expected Value",
            f"{classification['net_expected_value']:.2f}%",
            help=" - "
        )
        
        col_i2.metric(
            "Max Drawdown",
            f"{stats['max_drawdown']:.2f}%",
            help="Maximum historical drawdown."
        )
        
        col_i3.metric(
            "Multi-TF Validated",
            "Yes" if classification['multi_timeframe_validated'] else "No",
            help="Whether multi-timeframe validation passed."
        )
    
    # V1 
    with st.expander("V1 Classification (Reference Only)", expanded=False):
        st.caption("Reference-only diagnostic labels. Do not use for standalone trading decisions.")
        st.markdown(f"**V1 Classification**: {classification['classification_v1']}")
        
        st.markdown("**V1 classification guide**")
        if "Positive Edge" in classification['classification_v1']:
            st.success("Positive Edge (V1): high accuracy with positive returns.")
        elif "High Accuracy" in classification['classification_v1']:
            st.warning("High Accuracy, Low Returns (V1): directionally right but low payoff.")
        elif "Lucky Streak" in classification['classification_v1']:
            st.info("Lucky Streak (V1): low accuracy with positive return, likely noise.")
        elif "No Edge" in classification['classification_v1']:
            st.error("No Edge (V1): low accuracy with negative returns.")
        else:
            st.info("No data")
    
    st.divider()
    
    # ?
    st.subheader("Enhanced Statistics")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    col_s1.metric(
        "Cumulative Return",
        f"{stats['cumulative_return']:+.2f}%",
        help="Cumulative return across evaluated signals."
    )
    
    col_s2.metric(
        "Win Rate",
        f"{stats['win_rate']:.1f}%",
        help="Percentage of profitable signals."
    )
    
    col_s3.metric(
        "Profit Factor",
        f"{stats['profit_factor']:.2f}",
        help="Total gains divided by total losses."
    )
    
    col_s4.metric(
        "Volatility",
        f"{stats['volatility']:.2f}%",
        help="Standard deviation of signal returns."
    )
    
    # 
    col_w, col_l = st.columns(2)
    
    with col_w:
        st.metric(
            "Avg Win",
            f"{stats['avg_win']:+.2f}%",
            help=""
        )
    
    with col_l:
        st.metric(
            "Avg Loss",
            f"{stats['avg_loss']:+.2f}%",
            help=""
        )
    
    st.divider()
    
    # ?
    if not stats['statistical_significance']:
        st.warning(f"""
         **Statistical Significance Warning**
        
        Sample size: {stats['sample_size']} (minimum 30 required)
        
        The current sample size is too small to draw reliable conclusions. 
        Continue running analyses to build a larger dataset.
        """)
    else:
        st.success(f"Sample size: {stats['sample_size']} - Statistically significant")
    
    st.divider()
    
    # ?
    st.subheader(" Recent Signals")
    
    try:
        # ?0?
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
    
    with st.expander("How to Use Signal Scoreboard (V2 - Trading-Grade)"):
        st.markdown("""
        ### Signal Tracking Overview

        **Automatic Recording**
        - Each analysis run records predicted direction and confidence.
        - Entry price is stored at signal generation time.

        **Result Backfill**
        - Click **Update Results** to backfill 1h / 4h / 1d / 1w outcomes.
        - The system fetches realized prices and computes return accuracy.

        ### Trading-Grade Classification (V2)

        **Tradable Edge**
        - Sufficient sample size
        - Positive net expected value after transaction cost
        - Controlled drawdown
        - Multi-timeframe consistency

        **Directional Signal**
        - Useful directional bias, but not robust enough for standalone execution.

        **Unstable / Regime-Dependent**
        - Performance depends on market regime or has weak consistency.

        **No Edge**
        - No stable predictive advantage.

        **Insufficient Data**
        - Sample size is too small for reliable conclusions.

        ### Practical Guidance
        - Use V2 classification as the execution gate.
        - Re-check results regularly as market regime changes.
        - Combine with risk controls and portfolio-level constraints.
        - Past performance does not guarantee future results.
        """)


# === TAB 4: Early-Warning ===
with tab_warning:
    st.header("Early-Warning Risk Monitor")
    st.caption("Universal risk scoring system for monitored assets")
    
    st.divider()
    
    # 
    st.subheader("Watchlist")
    
    col_select, col_analyze = st.columns([3, 1])
    
    selected_asset = col_select.selectbox(
        "Select Asset to Analyze",
        list(WATCHLIST.keys()),
        index=0
    )
    
    if col_analyze.button("Calculate Risk Score"):
        with st.spinner(f"Analyzing risk for {selected_asset}..."):
            # ?
            recent_news = st.session_state.get('news', [])
            if not recent_news:
                recent_news = get_rss_news()
            
            # 
            risk_result = calculate_early_warning_score(selected_asset, recent_news)
            
            # ?session state
            st.session_state['risk_result'] = risk_result
            st.rerun()
    
    st.divider()
    
    # 
    if 'risk_result' in st.session_state:
        risk = st.session_state['risk_result']
        
        if 'error' in risk:
            st.error(f"Error: {risk['error']}")
        else:
            # 
            st.subheader(f"Risk Assessment: {risk['asset']}")
            
            # 
            total_score = risk['total_risk_score']
            risk_level = risk['risk_level']
            
            # 
            level_colors = {
                "LOW": "green",
                "MEDIUM": "yellow",
                "HIGH": "orange",
                "CRITICAL": "red"
            }
            level_color = level_colors.get(risk_level, "gray")
            
            # 
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
            
            # ?
            st.subheader("Risk Breakdown")
            
            sub_scores = risk['sub_scores']
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Macro Chain",
                f"{sub_scores['macro_chain']['score']}/25",
                help="USD/rates/macro environment impact"
            )
            
            col2.metric(
                "Crowding",
                f"{sub_scores['crowding']['score']}/25",
                help="Technical overbought/oversold levels"
            )
            
            col3.metric(
                "Microstructure",
                f"{sub_scores['microstructure']['score']}/25",
                help="Volatility/gaps/volume anomalies"
            )
            
            col4.metric(
                "Event Risk",
                f"{sub_scores['event_risk']['score']}/25",
                help="Central bank/policy/geopolitical events"
            )
            
            st.divider()
            
            # ?
            st.subheader("Risk Radar")
            
            categories = ['Macro Chain', 'Crowding', 'Microstructure', 'Event Risk']
            values = [
                sub_scores['macro_chain']['score'],
                sub_scores['crowding']['score'],
                sub_scores['microstructure']['score'],
                sub_scores['event_risk']['score']
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # 
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
            
            # ?
            st.subheader("Evidence Chain")
            
            # Macro Chain Evidence
            with st.expander("Macro Chain Evidence", expanded=True):
                macro_evidence = sub_scores['macro_chain']['evidence']
                if macro_evidence:
                    for idx, ev in enumerate(macro_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"{ev.get('message')}")
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
            with st.expander("Crowding Evidence"):
                crowding_evidence = sub_scores['crowding']['evidence']
                if crowding_evidence:
                    for idx, ev in enumerate(crowding_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"{ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Indicator**: {ev.get('indicator', 'N/A')}")
                            st.markdown(f"- **Value**: {ev.get('value', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No crowding risks detected")
            
            # Microstructure Evidence
            with st.expander("Microstructure Evidence"):
                micro_evidence = sub_scores['microstructure']['evidence']
                if micro_evidence:
                    for idx, ev in enumerate(micro_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"{ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Indicator**: {ev.get('indicator', 'N/A')}")
                            st.markdown(f"- **Value**: {ev.get('value', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No microstructure risks detected")
            
            # Event Risk Evidence
            with st.expander("Event Risk Evidence"):
                event_evidence = sub_scores['event_risk']['evidence']
                if event_evidence:
                    for idx, ev in enumerate(event_evidence, 1):
                        if ev.get('type') == 'error':
                            st.error(f"{ev.get('message')}")
                        else:
                            st.markdown(f"**Evidence {idx}**")
                            st.markdown(f"- **Category**: {ev.get('category', 'N/A')}")
                            st.markdown(f"- **Count**: {ev.get('count', 'N/A')}")
                            st.markdown(f"- **Interpretation**: {ev.get('interpretation', 'N/A')}")
                            st.divider()
                else:
                    st.info("No event risks detected")
            
            st.divider()
            
            # ?
            if risk['alert_triggers']:
                st.subheader("Alert Triggers")
                for trigger in risk['alert_triggers']:
                    st.warning(f"- {trigger}")
            
            # 
            st.subheader("Recommendation")
            st.info(risk['recommendation'])
            
            # ?
            st.caption(f"Analysis Time: {risk['timestamp'][:19]}")
    
    else:
        st.info("Select an asset and click 'Calculate Risk Score' to begin analysis")
    
    st.divider()
    
    # 
    with st.expander("How to Use Early-Warning System"):
        st.markdown("""
        ### Early-Warning Risk Scoring System
        
        **Purpose**:
        - Detect elevated risk BEFORE major moves
        - Provide evidence-based risk assessment
        - Help answer: "Should I reduce exposure now?"
        
        **Four Risk Dimensions**:
        
        1. **Macro Chain (0-25)**:
           - USD strength/weakness impact
           - Interest rate movements
           - Macro news flow
           - Based on correlations with DXY, 10Y yields
        
        2. **Crowding (0-25)**:
           - RSI overbought/oversold levels
           - Price deviation from moving averages
           - Volume spikes indicating crowding
        
        3. **Microstructure (0-25)**:
           - Volatility surges (ATR ratio)
           - Price gaps
           - Volume anomalies
        
        4. **Event Risk (0-25)**:
           - Central bank events (Fed, ECB)
           - Policy changes (tariffs, regulations)
           - Geopolitical tensions
        
        **Risk Levels**:
        - **LOW (0-25)**: Normal market environment
        - **MEDIUM (26-50)**: Some factors elevated, monitor
        - **HIGH (51-75)**: Multiple risks, consider caution
        - **CRITICAL (76-100)**: Extreme risk, reduce exposure
        
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
