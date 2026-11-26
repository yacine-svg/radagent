"""Fill data/cache/pubmed.json so the demo works without internet. Run once online:
    python -m servers.warm_cache
"""
from ml.labels import CLASSES
from servers.knowledge import PUBMED_CACHE, pubmed_query, search_pubmed

if __name__ == "__main__":
    for label in CLASSES:
        refs = search_pubmed(pubmed_query(label), max_results=3)
        print(f"{label:<14} {len(refs)} reference(s)")
        for r in refs:
            print(f"    PMID {r['pmid']} ({r['year']}) {r['title'][:90]}")
        if not refs:
            print("    ← none: check your connection")
    print("cache:", PUBMED_CACHE)
