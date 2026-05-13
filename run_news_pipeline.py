"""
run_news_pipeline.py — Standalone GlobalWatch news + LLM pipeline
─────────────────────────────────────────────────────────────────
Runs independently of the Streamlit UI. Every INTERVAL_MINUTES:
  1. Fetches RSS headlines (Reuters / CNBC / BBC)
  2. Calls local Ollama LLM to extract macro topic signals
  3. Writes results to ChromaDB (./memory_db)
     → paper_trading.py reads these via MacroSignalAdapter

Usage:
    python3 run_news_pipeline.py              # every 30 min
    python3 run_news_pipeline.py --interval 15
    python3 run_news_pipeline.py --once       # run once and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.environ.get("PAPER_CONFIG_PATH", "paper_config.json")
DEFAULT_INTERVAL_MINUTES = 30
LOCK_STALE_SECONDS = 35 * 60  # 35 minutes


# ── Process lock helpers ──────────────────────────────────────────────────────
def _lock_path() -> str:
    cfg = _load_config()
    chroma_path = str(cfg.get("macro_integration", {}).get("chroma_path", "./memory_db"))
    os.makedirs(chroma_path, exist_ok=True)
    return os.path.join(chroma_path, ".pipeline.lock")


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_lock() -> bool:
    lock_file = _lock_path()
    now_ts = time.time()

    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                data = json.load(f)
            pid = int(data.get("pid", 0))
            written_at = float(data.get("ts", 0))
            age = now_ts - written_at
            if age < LOCK_STALE_SECONDS and _is_pid_alive(pid):
                print(f"[LOCK] Another pipeline instance is running (PID={pid}, age={age:.0f}s). Exiting.")
                return False
            print(f"[LOCK] Stale lock found (PID={pid}, age={age:.0f}s) — removing.")
        except Exception as e:
            print(f"[LOCK] Could not read lock file: {e} — removing.")

    try:
        with open(lock_file, "w") as f:
            json.dump({"pid": os.getpid(), "ts": now_ts}, f)
        return True
    except Exception as e:
        print(f"[LOCK] Could not write lock file: {e} — proceeding without lock.")
        return True


def _release_lock() -> None:
    try:
        lp = _lock_path()
        if os.path.exists(lp):
            with open(lp, "r") as f:
                data = json.load(f)
            if int(data.get("pid", 0)) == os.getpid():
                os.remove(lp)
    except Exception:
        pass


RSS_FEEDS = {
    "Reuters":  "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "CNBC":     "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "BBC":      "http://feeds.bbci.co.uk/news/business/rss.xml",
    "GoogleFin":"https://news.google.com/rss/search?q=stock+market+economy&hl=en-US&gl=US&ceid=US:en",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("GW_LOCAL_MODEL", "qwen2.5:32b")


# ── Config loader ─────────────────────────────────────────────────────────────
def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {CONFIG_PATH}: {e}")
        return {}


# ── Step 1: RSS news fetch ────────────────────────────────────────────────────
def fetch_news() -> list[dict]:
    try:
        import feedparser  # type: ignore
    except ImportError:
        print("[ERROR] feedparser not installed. Run: pip3 install feedparser")
        return []

    news = []
    seen = set()
    for src, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                link  = (entry.get("link") or "").strip()
                if not title or link in seen:
                    continue
                seen.add(link)
                news.append({"source": src, "title": title, "link": link})
                count += 1
                if count >= 5:
                    break
        except Exception as e:
            print(f"[WARN] RSS fetch failed for {src}: {e}")
    return news


# ── Step 2: Ollama LLM call ───────────────────────────────────────────────────
def call_ollama(prompt: str, timeout: int = 120) -> str | None:
    try:
        import urllib.request as _req
        import urllib.error as _err
    except ImportError:
        return None

    body = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }).encode()

    req = _req.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _req.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except _err.URLError as e:
        print(f"[ERROR] Ollama not reachable at {OLLAMA_URL}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        return None


def extract_topic_signals(news: list[dict]) -> dict | None:
    if not news:
        return None

    headlines = "\n".join(
        f"- [{item['source']}] {item['title']}" for item in news[:20]
    )
    prompt = f"""You are a macro-news signal parser. OUTPUT LANGUAGE: ENGLISH.
Read the headlines and infer sector/theme momentum.

Headlines:
{headlines}

Return ONLY valid JSON with this exact schema (no markdown, no explanation):
{{
  "timestamp": "{datetime.now(timezone.utc).isoformat()}",
  "source": "llm+news",
  "confidence": 0.70,
  "summary": "1-2 sentence macro summary",
  "topic_signals": [
    {{"topic": "AI chip demand", "direction": "bullish", "strength": 2, "confidence": 0.75}},
    {{"topic": "interest rates", "direction": "bearish", "strength": 1, "confidence": 0.60}}
  ],
  "sector_signals": [
    {{"sector": "technology", "direction": "bullish", "confidence": 0.70}},
    {{"sector": "energy", "direction": "neutral", "confidence": 0.55}}
  ]
}}"""

    print(f"      Calling {MODEL_NAME} via Ollama...")
    raw = call_ollama(prompt, timeout=90)
    if not raw:
        return None

    # Extract JSON from response
    try:
        # Try direct parse first
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    print(f"[WARN] LLM response was not valid JSON:\n{raw[:200]}")
    return None


# ── Step 3: Write to ChromaDB ─────────────────────────────────────────────────
def write_to_chromadb(payload: dict, chroma_path: str, news: list[dict]) -> int:
    try:
        import chromadb  # type: ignore
    except ImportError:
        print("[ERROR] chromadb not installed. Run: pip3 install chromadb")
        return 0

    written = 0
    try:
        client     = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(name="trading_signals")

        topic_signals = payload.get("topic_signals", [])
        news_sources  = list({item["source"] for item in news})
        ts_now        = datetime.now(timezone.utc).isoformat()

        for sig in topic_signals:
            topic     = str(sig.get("topic", "unknown"))
            direction = str(sig.get("direction", "neutral"))
            strength  = int(sig.get("strength", 1))
            conf      = float(sig.get("confidence", 0.5))

            doc_id   = f"topic_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            document = (
                f"[{direction.upper()}] {topic} | "
                f"strength={strength} conf={conf:.2f} | "
                f"summary={payload.get('summary', '')}"
            )
            metadata = {
                "timestamp":  ts_now,
                "topic":      topic,
                "direction":  direction,
                "strength":   strength,
                "confidence": conf,
                "status":     "PENDING",
                "source":     ",".join(news_sources[:3]),
                "model":      MODEL_NAME,
            }
            collection.add(ids=[doc_id], documents=[document], metadatas=[metadata])
            written += 1

        print(f"      Wrote {written} topic signals → {chroma_path}/trading_signals")
    except Exception as e:
        print(f"[ERROR] ChromaDB write failed: {e}")

    return written


# ── Main pipeline cycle ───────────────────────────────────────────────────────
def check_industry_health(chroma_path: str) -> None:
    """Check industry_signals collection freshness and print status."""
    try:
        import chromadb  # type: ignore
    except ImportError:
        print("[INDUSTRY_HEALTH] chromadb not installed — cannot check industry signals")
        return

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            collection = client.get_collection(name="industry_signals")
        except Exception:
            print("[INDUSTRY_HEALTH] WARN: industry_signals collection not found — "
                  "run: python3 GlobalWatch_V2.py --run-industry-runtime-once")
            return

        results = collection.get(include=["metadatas"])
        metadata_rows = results.get("metadatas", []) if isinstance(results, dict) else []
        if not metadata_rows:
            print("[INDUSTRY_HEALTH] WARN: industry_signals collection is empty — "
                  "run: python3 GlobalWatch_V2.py --run-industry-runtime-once")
            return

        now_utc = datetime.now(timezone.utc)
        most_recent_ts = None
        for meta in metadata_rows:
            if not isinstance(meta, dict):
                continue
            ts_str = meta.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if most_recent_ts is None or ts > most_recent_ts:
                    most_recent_ts = ts
            except Exception:
                pass

        if most_recent_ts is None:
            print("[INDUSTRY_HEALTH] WARN: no parseable timestamps in industry_signals")
            return

        age_hours = max(0.0, (now_utc - most_recent_ts).total_seconds() / 3600.0)
        count = len(metadata_rows)
        if age_hours > 4.0:
            print(f"[INDUSTRY_HEALTH] STALE: last signal age={age_hours:.1f}h, count={count} — "
                  f"run: python3 GlobalWatch_V2.py --run-industry-runtime-once")
        else:
            print(f"[INDUSTRY_HEALTH] OK: last signal age={age_hours:.1f}h, count={count}")
    except Exception as e:
        print(f"[INDUSTRY_HEALTH] ERROR: {e}")


def run_once(include_industry: bool = False) -> bool:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[NEWS PIPELINE] {now_str}  model={MODEL_NAME}")
    print(f"{'='*60}")

    cfg        = _load_config()
    macro_cfg  = cfg.get("macro_integration", {})
    chroma_path = str(macro_cfg.get("chroma_path", "./memory_db"))

    # 1. Fetch news
    print("[1/3] Fetching RSS news...")
    news = fetch_news()
    print(f"      Got {len(news)} articles")
    if not news:
        print("[WARN] No articles. Skipping.")
        return False

    # 2. LLM extraction
    if not bool(macro_cfg.get("enable_llm_topic_signals", True)):
        print("[2/3] LLM signals disabled in config — skipping")
        payload = None
    else:
        print("[2/3] Running LLM topic extraction...")
        payload = extract_topic_signals(news)
        if payload:
            print(f"      summary: {payload.get('summary', '')[:80]}")
            print(f"      topic_signals: {len(payload.get('topic_signals', []))}")
        else:
            print("      No payload returned (model may still be loading)")

    # 3. Write to ChromaDB
    print("[3/3] Writing to ChromaDB...")
    n_written = 0
    if payload:
        n_written = write_to_chromadb(payload, chroma_path, news)

    if n_written == 0 and payload:
        print("      0 signals written — check chromadb installation")
    elif n_written == 0:
        print("      Skipped (no payload)")

    # 4. Check industry signal health (optional)
    if include_industry:
        print("[4/4] Checking industry signal health...")
        check_industry_health(chroma_path)

    print(f"\n[OK] Done at {datetime.now().strftime('%H:%M:%S')}")
    return True


def run_loop(interval_minutes: int, include_industry: bool = False):
    if not _acquire_lock():
        sys.exit(1)
    try:
        print(f"\nGlobalWatch News Pipeline — every {interval_minutes} min")
        print(f"Model: {MODEL_NAME}  |  Config: {CONFIG_PATH}")
        print(f"Ollama: {OLLAMA_URL}")
        print(f"Lock: {_lock_path()}")
        print("Press Ctrl+C to stop\n")

        run_once(include_industry=include_industry)

        while True:
            next_run = time.time() + interval_minutes * 60
            print(f"\n[SLEEP] Next run at {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')} "
                  f"(in {interval_minutes} min)...")
            try:
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\n[STOP] Pipeline stopped.")
                break
            run_once(include_industry=include_industry)
    finally:
        _release_lock()


def main():
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description="GlobalWatch standalone news pipeline")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MINUTES,
                        help=f"Minutes between runs (default: {DEFAULT_INTERVAL_MINUTES})")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Path to paper_config.json")
    parser.add_argument("--include-industry", action="store_true",
                        help="Also check industry_signals collection health each cycle")
    args = parser.parse_args()

    CONFIG_PATH = args.config

    if args.once:
        sys.exit(0 if run_once(include_industry=args.include_industry) else 1)
    else:
        run_loop(args.interval, include_industry=args.include_industry)


if __name__ == "__main__":
    main()
