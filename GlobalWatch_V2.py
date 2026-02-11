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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import urllib.parse
import urllib.request
import os

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
LOCAL_MODEL = "gemma3:12" 
# Temperature range: 0.1 ~ 0.3 (configured in ollama.chat calls)
TEMPERATURE = 0.2  # Default temperature for model calls

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

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
    }


def _default_news_sources_cfg():
    return {
        "market_rss_enabled": True,
        "ticker_rss_enabled": True,
        "industry_rss_enabled": True,
        "timeout_seconds": 8,
        "retries": 1,
        "max_per_l2": 8,
        "max_total": 60,
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


def bucket_news_by_l2(mapped_news, l2_list, max_per_l2=8):
    buckets = {str(l2): [] for l2 in l2_list}
    for item in mapped_news:
        l2_tags = item.get("matched_L2", [])
        if not isinstance(l2_tags, list):
            continue
        for l2 in l2_tags:
            key = str(l2)
            if key not in buckets:
                continue
            if len(buckets[key]) >= max(1, _safe_int(max_per_l2, 8)):
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
    normalized["L2"] = str(signal.get("L2") or l2)
    normalized["scope"] = "industry"
    normalized["L3_focus"] = [str(x) for x in signal.get("L3_focus", []) if str(x).strip()][:8]
    normalized["risk_delta"] = float(np.clip(_safe_float(signal.get("risk_delta", 0.0), 0.0), -1.0, 1.0))
    normalized["confidence"] = float(np.clip(_safe_float(signal.get("confidence", 0.0), 0.0), 0.0, 1.0))
    normalized["horizon"] = str(signal.get("horizon", "1d"))
    drivers = signal.get("top_drivers", [])
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
    raw_evidence = signal.get("evidence", [])
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
    normalized["notes_speculative"] = bool(signal.get("notes_speculative", False))
    normalized["asof_utc"] = str(signal.get("asof_utc", asof_utc))
    if normalized["confidence"] < 0.4:
        normalized["risk_delta"] = 0.0
    return normalized


def _build_industry_llm_prompt(l2, news_items, asof_utc):
    compact_rows = []
    for item in news_items[:8]:
        compact_rows.append(
            {
                "id": item.get("id"),
                "source": item.get("source"),
                "title": str(item.get("title", ""))[:180],
                "summary": str(item.get("summary", ""))[:220],
                "url": item.get("url"),
                "matched_tickers": item.get("matched_tickers", []),
                "matched_L2": item.get("matched_L2", []),
                "matched_L3": item.get("matched_L3", []),
            }
        )
    schema = {
        "asof_utc": asof_utc,
        "scope": "industry",
        "L2": l2,
        "L3_focus": ["string"],
        "risk_delta": "float in [-1,1]",
        "confidence": "float in [0,1]",
        "horizon": "1d|1w|1m",
        "top_drivers": ["string <=5"],
        "impacted_tickers": [{"ticker": "str", "direction": "up|down|neutral", "magnitude": "float[-1,1]", "reason": "str"}],
        "evidence": [{"id": "string", "source": "string", "title": "string", "url": "string"}],
        "notes_speculative": False,
    }
    prompt = (
        "You are an industry risk signal parser. Return STRICT JSON only.\n"
        "Do not output markdown, comments, or extra text.\n"
        "Rules:\n"
        "1) Use only facts from evidence items.\n"
        "2) If evidence is weak or mixed, return risk_delta=0 and confidence<=0.4.\n"
        "3) Keep top_drivers <=5 and impacted_tickers <=8.\n\n"
        f"Industry L2: {l2}\n"
        f"Asof UTC: {asof_utc}\n"
        f"Evidence items:\n{json.dumps(compact_rows, ensure_ascii=False)}\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return prompt


def _generate_industry_signal_with_llm(l2, items, model_name):
    asof_utc = datetime.now(timezone.utc).isoformat()
    prompt = _build_industry_llm_prompt(l2, items, asof_utc)
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        raw_text = (
            resp.get("message", {}).get("content")
            if isinstance(resp, dict)
            else ""
        )
        parsed = robust_json_parse(raw_text, model_name, max_retries=1)
        return _normalize_industry_signal(parsed, l2, asof_utc, items), raw_text, None
    except Exception as e:
        neutral = _neutral_industry_signal(l2, asof_utc, items, reason=f"llm_error:{e}")
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
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        l2 = str(signal.get("L2", "unknown"))
        asof = str(signal.get("asof_utc", datetime.now(timezone.utc).isoformat()))
        date_key = asof[:10]
        payload = json.dumps(signal, ensure_ascii=False)
        item_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        row_id = f"industry::{l2}::{date_key}::{item_hash}"
        ids.append(row_id)
        docs.append(payload)
        metas.append(
            {
                "timestamp": asof,
                "status": "PENDING",
                "scope": "industry",
                "L2": l2,
                "confidence": float(np.clip(_safe_float(signal.get("confidence", 0.0), 0.0), 0.0, 1.0)),
                "risk_delta": float(np.clip(_safe_float(signal.get("risk_delta", 0.0), 0.0), -1.0, 1.0)),
                "horizon": str(signal.get("horizon", "1d")),
                "source_count": len(signal.get("evidence", []) if isinstance(signal.get("evidence"), list) else []),
                "ticker_count": len(signal.get("impacted_tickers", []) if isinstance(signal.get("impacted_tickers"), list) else []),
                "version": "industry_news_v1",
            }
        )
    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
    return {"written": len(ids), "collection": collection_name}


def run_industry_news_pipeline(
    config,
    *,
    portfolio_tickers=None,
    candidate_tickers=None,
    synthetic_news=None,
    llm_stub=None,
    chroma_client_override=None,
    chroma_path_override=None,
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

    deduped = _dedup_and_limit_news(age_filtered, max_total=_safe_int(sources_cfg.get("max_total", 60), 60))
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
    )

    generated_signals = []
    llm_errors = []
    llm_model = str(
        cfg.get("news_overlay", {}).get(
            "llm_model",
            cfg.get("macro_integration", {}).get("llm_topic_model", LOCAL_MODEL),
        )
    )
    for l2, items in buckets.items():
        if not items:
            continue
        if callable(llm_stub):
            try:
                stub_obj = llm_stub(l2, items)
                if isinstance(stub_obj, str):
                    parsed = robust_json_parse(stub_obj, llm_model, max_retries=1)
                    normalized = _normalize_industry_signal(
                        parsed,
                        l2,
                        datetime.now(timezone.utc).isoformat(),
                        items,
                    )
                else:
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
            normalized, _raw_text, err = _generate_industry_signal_with_llm(l2, items, llm_model)

        if err:
            llm_errors.append({"L2": l2, "error": err})
        generated_signals.append(normalized)

    collection_name = str(news_overlay_cfg.get("industry_collection", "industry_signals"))
    chroma_path = (
        str(chroma_path_override)
        if chroma_path_override
        else str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    )
    write_info = write_industry_signals_to_chroma(
        generated_signals,
        collection_name=collection_name,
        chroma_path=chroma_path,
        client_override=chroma_client_override,
    )

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
        with open(dryrun_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_copy, f, ensure_ascii=False, indent=2)

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
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(preview_payload, f, ensure_ascii=False, indent=2)
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
    try:
        cfg = _load_json_config(config_path)
    except Exception as e:
        log_error(f"[INDUSTRY_NEWS] config load failed: {e}")
        return {"status": "config_error", "error": str(e)}

    overlay_cfg = _merge_cfg(_default_news_overlay_cfg(), cfg.get("news_overlay", {}))
    if not bool(overlay_cfg.get("enabled", False)):
        return {"status": "disabled"}

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
     DeepSeek-R1 ?
    : (? SON)
    """
    # 1.  <think>...</think> ?
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thought_process = think_match.group(1).strip() if think_match else "No internal thought process detected (Direct Output)."
    
    # 2.  <think> ?JSON 
    json_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    #  Markdown ?
    json_text = re.sub(r'```json', '', json_text)
    json_text = re.sub(r'```', '', json_text).strip()
    
    return thought_process, json_text

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
    #  JSON
    json_str = extract_json_from_text(raw_content)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            pass  # ?
    
    # ?
    for attempt in range(max_retries):
        repaired_json_str = self_repair_json(raw_content, model)
        if repaired_json_str:
            try:
                return json.loads(repaired_json_str)
            except json.JSONDecodeError:
                continue
    
    # 
    return {
        "status": "error",
        "reason": "Failed to parse JSON after extraction and self-repair attempts",
        "raw_output": raw_content[:500] + "..." if len(raw_content) > 500 else raw_content,
        "evidence": [],
        "_parse_error": True
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

def analyze_all(news, user_pairs, macro_data, lang_mode):
    if not news: return {"status": "no_update"}
    
    # ?prompt
    headlines = " ".join([f"[{item['source']}] {item['title']}" for item in news])
    history = recall_history(headlines)
    lang_instruction = "OUTPUT LANGUAGE: ENGLISH"

    # ?MACRO_LOGIC_KNOWLEDGE +  evidence 
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
        #  num_ctx 
        response = ollama.chat(model=LOCAL_MODEL, messages=[{'role': 'user', 'content': prompt}], options={"num_ctx": 8192})
        raw_content = response['message']['content']
        
        # ?robust_json_parse  json.loads
        thought, json_text = parse_deepseek_output(raw_content)
        
        # 
        res = robust_json_parse(json_text, LOCAL_MODEL, max_retries=1)
        
        # ?
        if res.get('_parse_error'):
            res['thought_process'] = thought
            return res
        
        # ?
        res['thought_process'] = thought
        
        # ?evidence ?
        evidence = res.get('evidence', [])
        validated_evidence, valid_count = validate_evidence(evidence, news)
        res['evidence'] = validated_evidence
        res['_valid_evidence_count'] = valid_count
        
        # ?
        if valid_count == 0 and res.get('status') == 'alert':
            res['_evidence_warning'] = True
            original_advice = res.get('advice', '')
            res['advice'] = f"{original_advice}\n\n WARNING: No valid evidence found. Predictions may be unreliable. Please verify independently."

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

        # Optional industry-news pipeline (separate collection; does not alter macro pipeline behavior).
        try:
            industry_runtime = run_industry_news_pipeline_runtime(
                config_path=os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")
            )
            if isinstance(industry_runtime, dict):
                res['_industry_news_status'] = industry_runtime.get("status")
                if industry_runtime.get("status") == "ok":
                    result_obj = industry_runtime.get("result", {})
                    write_info = result_obj.get("write_info", {}) if isinstance(result_obj, dict) else {}
                    res['_industry_news_written'] = int(write_info.get("written", 0) or 0)
                else:
                    res['_industry_news_written'] = 0
        except Exception as e:
            log_error(f"[INDUSTRY_NEWS] runtime pipeline failed: {e}")
            res['_industry_news_status'] = "error"
            res['_industry_news_written'] = 0
        
        # ?
        if res.get("status") == "alert" and res.get("predictions"):
            predictions = res.get("predictions", {})
            impact_score = res.get("impact_score", 0)
            
            # ?
            for asset, prediction_text in predictions.items():
                # 
                direction = "Neutral"
                if "Bullish" in prediction_text or "bullish" in prediction_text:
                    direction = "Bullish"
                elif "Bearish" in prediction_text or "bearish" in prediction_text:
                    direction = "Bearish"
                
                # 
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
        # 
        return {
            "status": "error",
            "reason": f"Unexpected error: {str(e)}",
            "raw_output": "",
            "evidence": [],
            "_parse_error": True
        }

def analyze_single_stock(ticker, news, lang_mode):
    lang_instruction = "OUTPUT LANGUAGE: ENGLISH"
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

# ================= 4. UI  =================



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

        return _cached_snapshot(path, int(refresh_nonce), int(mtime_ns), int(file_size))

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

        return _cached_summary(path, int(refresh_nonce), int(mtime_ns), int(file_size))

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

        return _cached_trades(path, int(refresh_nonce), int(mtime_ns), int(file_size))

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

        return _cached_index(path, int(refresh_nonce))

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

        return _cached_reports(path_payload, int(refresh_nonce))

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

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Equity", f"${total_equity:,.2f}")
            metric_col2.metric("Cash", f"${cash:,.2f}")
            metric_col3.metric("Positions", f"${positions_value:,.2f}")
            snap_ts = snapshot.get("timestamp")
            if snap_ts:
                st.caption(f"Snapshot timestamp: {snap_ts}")

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
                    for trade in trades:
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
                            }
                        )

                    trade_df = pd.DataFrame(trade_rows)
                    if not trade_df.empty:
                        trade_df["time_sort"] = pd.to_datetime(trade_df["time"], errors="coerce", utc=True)
                        if trade_df["time_sort"].notna().any():
                            trade_df = trade_df.sort_values("time_sort", ascending=False)
                        else:
                            # If timestamps are not parseable, keep UI newest-first by reversing read order.
                            trade_df = trade_df.iloc[::-1]
                        trade_df = trade_df.drop(columns=["time_sort"])
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
    
                    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
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
                        x_tick_label = "auto" if x_tick_mode == "auto" else f"{x_tick_minutes}min"
                        y_mode_label = "auto"
                        if str(st.session_state.get("pm_y_mode", "auto")) == "manual":
                            y_mode_label = f"manual [{y_min:.2f}, {y_max:.2f}]"
                        st.caption(
                            "X: last "
                            f"{int(st.session_state.get('pm_window_hours', 48))}h @ {str(st.session_state.get('pm_resample_rule', '15min'))}"
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
                            )
                            st.plotly_chart(fig, width="stretch")
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

    delta = (datetime.now() - st.session_state['last_run']).total_seconds()
    remain = max(0, refresh_sec - delta) if refresh_sec > 0 else 0
    
    if st.button("Run Deep Reason Analysis") or (refresh_sec > 0 and remain == 0 and auto_run):
        with st.status("DeepSeek is thinking...", expanded=True) as s:
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
            
            # Optional thought process rendering
            if res.get('thought_process'):
                with st.expander("DeepSeek Thought Process (Click to expand)", expanded=False):
                    st.markdown(res.get('thought_process', 'No thoughts recorded.'))
        # ================================
        
        # V3 thought process rendering
        elif res.get("status") != "error":
            with st.expander("DeepSeek Thought Process (Click to expand)", expanded=False):
                st.markdown(res.get('thought_process', 'No thoughts recorded.'))
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

# === TAB 2: Stock Analysis ===
with tab_stock:
    st.header("US Stock Deep Dive")
    c_in, c_go = st.columns([3, 1])
    ticker = c_in.text_input("Ticker", value="NVDA").upper()
    
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
                    
                    analysis = analyze_single_stock(ticker, stock_news, lang_mode)
                    
                    # V3 thought process rendering
                    with st.expander(" AI Thought Process (Stock)", expanded=True):
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





