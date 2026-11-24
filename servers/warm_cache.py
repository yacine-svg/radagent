"""Fill data/cache/pubmed.json so the demo works without internet. Run once online:
    python -m servers.warm_cache
"""
from ml.labels import CLASSES
from servers.knowledge import PUBMED_CACHE, PUBMED_QUERY, search_pubmed

if __name__ == "__main__":
    for label in CLASSES:
        refs = search_pubmed(PUBMED_QUERY.format(label=label), max_results=3)
        print(f"{label:<14} {len(refs)} reference(s)" + (f" — e.g. PMID {refs[0]['pmid']}: {refs[0]['title'][:60]}" if refs else "  ← check your connection"))
    print("cache:", PUBMED_CACHE)
