import json
import os
import sys
from datetime import datetime, timezone
from collections import Counter

import GlobalWatch_V2 as gw

CFG_PATH = "paper_config.json"
OUT_PATH = "outputs/debug_industry_mapping_after_fix.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cfg = json.load(open(CFG_PATH, "r", encoding="utf-8"))
l2_to_tickers, l3_to_tickers, ticker_to_tags = gw.build_industry_membership(cfg)
l2_list = list(l2_to_tickers.keys())

# Only pull industry RSS to isolate seed_l2 routing behavior.
sources_cfg = cfg.get("news_sources", {})
prov = gw.IndustryGoogleRSSProvider(
    sources_cfg.get("industry_rss_template", ""),
    sources_cfg.get("industry_topic_queries", {}),
    timeout_seconds=int(sources_cfg.get("timeout_seconds", 8)),
    retries=int(sources_cfg.get("retries", 1)),
)

context = {"industry_topics": l2_list}
items = prov.fetch(context)
items = [dict(x) for x in items if isinstance(x, dict)]
items = gw._dedup_and_limit_news(items, max_total=int(sources_cfg.get("max_total", 60)))

mapped = gw.map_news_items_to_taxonomy(
    items,
    ticker_to_tags=ticker_to_tags,
    industry_taxonomy=cfg.get("industry_taxonomy", {}),
    industry_keyword_map=sources_cfg.get("industry_keyword_map", {}),
)

seed_total = 0
seed_miss = 0
seed_miss_examples = []
for it in mapped:
    seed = str(it.get("seed_l2") or "").strip()
    if seed:
        seed_total += 1
        matched = [str(x) for x in (it.get("matched_L2") or [])]
        if seed not in matched:
            seed_miss += 1
            if len(seed_miss_examples) < 10:
                seed_miss_examples.append(
                    {
                        "seed_l2": seed,
                        "matched_L2": matched,
                        "source": it.get("source"),
                        "title": it.get("title"),
                    }
                )

seed_miss_ratio = (seed_miss / seed_total) if seed_total else 0.0
print("seed_total =", seed_total)
print("seed_miss  =", seed_miss)
print("seed_miss_ratio =", round(seed_miss_ratio, 3))

buckets = gw.bucket_news_by_l2(
    mapped,
    l2_list=l2_list,
    max_per_l2=int(sources_cfg.get("max_per_l2", 8)),
    prefer_seed_primary=True,
)

print("\n=== TECHNOLOGY BUCKET TOP ===")
for it in (buckets.get("technology") or [])[:8]:
    print(
        {
            "seed_l2": it.get("seed_l2"),
            "primary_l2": it.get("primary_l2"),
            "source": it.get("source"),
            "matched_L2": it.get("matched_L2"),
            "title": it.get("title"),
        }
    )

print("\n=== BUCKET CONTAMINATION (seed_l2 != bucket) ===")
pollution_summary = {}
pollution_samples = {}
for l2 in sorted(l2_list):
    entries = buckets.get(l2, []) or []
    total = len(entries)
    bad = 0
    samples = []
    for it in entries:
        seed = str(it.get("seed_l2") or "").strip()
        if seed and seed != l2:
            bad += 1
            if len(samples) < 3:
                samples.append(
                    {
                        "seed_l2": seed,
                        "bucket": l2,
                        "source": it.get("source"),
                        "title": it.get("title"),
                    }
                )
    ratio = (bad / total) if total else 0.0
    pollution_summary[l2] = {
        "bad": bad,
        "total": total,
        "ratio": ratio,
    }
    pollution_samples[l2] = samples
    if total:
        print(f"{l2} bad/total = {bad} / {total} ratio={ratio:.3f}")

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "config_path": CFG_PATH,
    "seed_total": seed_total,
    "seed_miss": seed_miss,
    "seed_miss_ratio": seed_miss_ratio,
    "seed_miss_examples_top10": seed_miss_examples,
    "pollution": pollution_summary,
    "pollution_samples_top3_each_bucket": pollution_samples,
    "technology_bucket_preview_top8": [
        {
            "seed_l2": it.get("seed_l2"),
            "primary_l2": it.get("primary_l2"),
            "source": it.get("source"),
            "matched_L2": it.get("matched_L2"),
            "title": it.get("title"),
        }
        for it in (buckets.get("technology") or [])[:8]
    ],
}

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"\n[DEBUG] wrote: {OUT_PATH}")
