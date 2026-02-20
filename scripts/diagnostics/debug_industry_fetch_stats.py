import json
import os
import sys
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone

import GlobalWatch_V2 as gw

CFG_PATH = "paper_config.json"
OUT_PATH = "outputs/debug_industry_fetch_stats.json"


def _safe_reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _count_by_seed(items, expected_l2=None):
    counts = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        seed = str(item.get("seed_l2", "")).strip() or "UNSEEDED"
        counts[seed] += 1
    if isinstance(expected_l2, list):
        for l2 in expected_l2:
            counts.setdefault(str(l2), 0)
    return dict(counts)


def _make_dedup_key(item):
    title = str(item.get("title", "")).strip()
    url = str(item.get("url", "")).strip()
    url_key = url.lower()
    title_key = gw.normalize_title(title) if title else ""
    return url_key or f"title::{title_key}"


def _dedup_only_with_collisions(items):
    dedup = {}
    collisions = defaultdict(lambda: {"kept": None, "dropped": []})

    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not title and not url:
            continue

        key = _make_dedup_key(raw)
        seed = str(raw.get("seed_l2", "")).strip() or "UNSEEDED"
        source = str(raw.get("source", "")).strip()

        if key not in dedup:
            item = dict(raw)
            item["id"] = gw._stable_news_id(item)
            item["_dedup_key"] = key
            item["_merge_idx"] = idx
            dedup[key] = item
            collisions[key]["kept"] = {
                "seed_l2": seed,
                "source": source,
                "title": title[:200],
                "url": url,
            }
            continue

        collisions[key]["dropped"].append(
            {
                "seed_l2": seed,
                "source": source,
                "title": title[:200],
                "url": url,
            }
        )

    dedup_items = list(dedup.values())
    dedup_items.sort(
        key=lambda x: gw._parse_iso_or_none(x.get("published_at"))
        or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )

    collision_rows = []
    for key, info in collisions.items():
        dropped = info.get("dropped") or []
        if not dropped:
            continue
        collision_rows.append(
            {
                "dedup_key": key,
                "kept": info.get("kept"),
                "dropped_count": len(dropped),
                "dropped": dropped[:20],
            }
        )
    return dedup_items, collision_rows


def main():
    _safe_reconfigure_stdout()
    cfg = json.load(open(CFG_PATH, "r", encoding="utf-8"))
    sources_cfg = cfg.get("news_sources", {}) if isinstance(cfg.get("news_sources"), dict) else {}

    l2_to_tickers, _, ticker_to_tags = gw.build_industry_membership(cfg)
    l2_list = list(l2_to_tickers.keys())

    provider = gw.IndustryGoogleRSSProvider(
        sources_cfg.get("industry_rss_template", ""),
        sources_cfg.get("industry_topic_queries", {}),
        timeout_seconds=int(sources_cfg.get("timeout_seconds", 8)),
        retries=int(sources_cfg.get("retries", 1)),
    )

    # S1: raw fetch counts per L2/query (before provider limits entries[:5])
    raw_count_by_l2_query = {}
    raw_fetch_errors = {}
    for l2 in l2_list:
        queries = provider.topic_queries.get(l2, [l2])
        if not isinstance(queries, list):
            queries = [str(queries)]
        for query_text in queries[:3]:
            key = f"{l2}::{query_text}"
            url = str(provider.template).format(query=urllib.parse.quote(str(query_text)))
            try:
                entries = gw._fetch_feed_entries(
                    url,
                    timeout_seconds=provider.timeout_seconds,
                    retries=provider.retries,
                )
                raw_count_by_l2_query[key] = len(entries)
            except Exception as e:
                raw_count_by_l2_query[key] = 0
                raw_fetch_errors[key] = str(e)

    # S2: merged from provider.fetch (before dedup/limit)
    context = {"industry_topics": l2_list}
    merged_items = provider.fetch(context)
    merged_count_by_seed_l2 = _count_by_seed(merged_items, expected_l2=l2_list)

    # S3: dedup only + collisions
    dedup_items, collision_rows = _dedup_only_with_collisions(merged_items)
    dedup_count_by_seed_l2 = _count_by_seed(dedup_items, expected_l2=l2_list)

    # S4: max_total limit after dedup (legacy pre-bucket truncation behavior)
    max_total = max(1, int(sources_cfg.get("max_total", 60)))
    limited_items = dedup_items[:max_total]
    limited_count_by_seed_l2 = _count_by_seed(limited_items, expected_l2=l2_list)

    # S5: post-bucket counts using current runtime path (dedup no global truncation -> map -> bucket)
    mapped = gw.map_news_items_to_taxonomy(
        dedup_items,
        ticker_to_tags=ticker_to_tags,
        industry_taxonomy=cfg.get("industry_taxonomy", {}),
        industry_keyword_map=sources_cfg.get("industry_keyword_map", {}),
    )
    buckets = gw.bucket_news_by_l2(
        mapped,
        l2_list=l2_list,
        max_per_l2=int(sources_cfg.get("max_per_l2", 8)),
        prefer_seed_primary=bool(sources_cfg.get("prefer_seed_primary", True)),
    )
    buckets = gw._apply_post_bucket_total_cap(buckets, sources_cfg.get("post_bucket_max_total"))
    post_bucket_count_by_l2 = {
        str(l2): len(buckets.get(str(l2), []) or [])
        for l2 in l2_list
    }
    post_bucket_total = int(sum(post_bucket_count_by_l2.values()))

    # Conclusion guess for technology starvation
    tech_s1 = sum(v for k, v in raw_count_by_l2_query.items() if k.startswith("technology::"))
    tech_s3 = int(dedup_count_by_seed_l2.get("technology", 0))
    tech_s4 = int(limited_count_by_seed_l2.get("technology", 0))
    if tech_s1 <= 1:
        conclusion_guess = "technology_raw_source_too_sparse_or_query_weak"
    elif tech_s1 > 1 and tech_s3 == 0:
        conclusion_guess = "technology_starved_by_global_dedup"
    elif tech_s3 > 0 and tech_s4 == 0:
        conclusion_guess = "technology_starved_by_max_total_truncation"
    else:
        conclusion_guess = "technology_not_starved_in_fetch_dedup_limit_pipeline"

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "l2_list_order": l2_list,
        "max_total": max_total,
        "max_per_l2": int(sources_cfg.get("max_per_l2", 8)),
        "post_bucket_max_total": sources_cfg.get("post_bucket_max_total"),
        "S1_raw_count_by_l2_query": raw_count_by_l2_query,
        "S1_raw_fetch_errors": raw_fetch_errors,
        "S2_merged_count_by_seed_l2": merged_count_by_seed_l2,
        "S2_merged_total": len(merged_items),
        "S3_dedup_count_by_seed_l2": dedup_count_by_seed_l2,
        "S3_dedup_total": len(dedup_items),
        "S3_collisions": {
            "total_collision_keys": len(collision_rows),
            "rows": collision_rows[:300],
        },
        "S4_limited_count_by_seed_l2": limited_count_by_seed_l2,
        "S4_limited_total": len(limited_items),
        "S5_post_bucket_count_by_l2": post_bucket_count_by_l2,
        "S5_post_bucket_total": post_bucket_total,
        "conclusion_guess": conclusion_guess,
        "technology_summary": {
            "S1_raw_total_for_technology_queries": tech_s1,
            "S2_merged_count_seed_technology": int(merged_count_by_seed_l2.get("technology", 0)),
            "S3_dedup_count_seed_technology": tech_s3,
            "S4_limited_count_seed_technology": tech_s4,
            "S5_post_bucket_count_technology": int(post_bucket_count_by_l2.get("technology", 0)),
        },
    }

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[FETCH-STATS] wrote:", OUT_PATH)
    print("[FETCH-STATS] l2_list_order:", ", ".join(l2_list))
    print("[FETCH-STATS] merged_total:", len(merged_items), "dedup_total:", len(dedup_items), "limited_total:", len(limited_items))
    print("[FETCH-STATS] post_bucket_total:", post_bucket_total)
    print("[FETCH-STATS] technology S1/S3/S4/S5:", tech_s1, tech_s3, tech_s4, post_bucket_count_by_l2.get("technology", 0))
    print("[FETCH-STATS] conclusion_guess:", conclusion_guess)


if __name__ == "__main__":
    main()
