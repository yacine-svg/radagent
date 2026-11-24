"""knowledge-server logic (no MCP here, so it is easy to test).

Tools (contracts.md §6):
    search_pubmed(query, max_results=3)   -> list[Reference]
    icd10_lookup(condition)               -> ICD10
    guideline_snippets(condition, k=2)    -> list[GuidelineSnippet]
All functions never raise (AGENTS.md 6.1).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from contracts import label_slug

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PUBMED_CACHE = DATA / "cache" / "pubmed.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_QUERY = "{label} chest radiograph"   # one query per class (warm_cache.py + knowledge agent)
log = logging.getLogger("knowledge-server")


# ------------------------------------------------------------------ PubMed
def _http_get_json(url: str, params: dict) -> dict:
    import httpx                                         # imported here: only needed online
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params = {**params, "api_key": key}
    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _load_cache() -> dict:
    try:
        return json.loads(PUBMED_CACHE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    PUBMED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PUBMED_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def search_pubmed(query: str, max_results: int = 3, fetch=_http_get_json) -> list[dict]:
    """Top PubMed results for `query`. Served from data/cache/pubmed.json when possible,
    so the demo works offline. On any network error: cached results, else []."""
    query = " ".join(str(query).split())
    max_results = max(1, min(int(max_results), 10))
    cache = _load_cache()
    if query in cache and len(cache[query]) >= max_results:
        return cache[query][:max_results]
    try:
        ids = fetch(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": query, "retmode": "json",
                                                "retmax": max_results, "sort": "relevance"}
                    )["esearchresult"]["idlist"]
        refs = []
        if ids:
            if not os.environ.get("NCBI_API_KEY"):
                time.sleep(0.34)                          # NCBI limit: 3 requests/second without a key
            res = fetch(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})["result"]
            for pmid in ids:
                item = res.get(pmid, {})
                year = re.match(r"\d{4}", item.get("pubdate", ""))
                refs.append({"pmid": str(pmid), "title": item.get("title", "").strip() or "(untitled)",
                             "journal": item.get("source", ""), "year": year.group(0) if year else ""})
        cache[query] = refs
        _save_cache(cache)
        return refs
    except Exception as e:                                # noqa: BLE001 — never raise
        log.warning("PubMed search failed for %r: %s", query, e)
        return cache.get(query, [])[:max_results]


# ------------------------------------------------------------------ ICD-10
@lru_cache(maxsize=1)
def _icd10_table() -> dict[str, dict]:
    with open(DATA / "icd10.csv", encoding="utf-8", newline="") as f:
        return {row["condition"].lower(): row for row in csv.DictReader(f)}


def icd10_lookup(condition: str) -> dict:
    """ICD-10 code for a supported finding; code/label are null when unknown."""
    try:
        row = _icd10_table().get(str(condition).strip().lower())
    except Exception as e:                                # noqa: BLE001
        log.warning("ICD-10 table unavailable: %s", e)
        row = None
    if row:
        return {"condition": condition, "code": row["code"], "label": row["label"]}
    return {"condition": condition, "code": None, "label": None}


# ------------------------------------------------------------------ guideline notes (BM25 retrieval)
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by can for from in into is it its may more of on or than that the "
           "their this to up used when where which with without".split())


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@lru_cache(maxsize=1)
def _guideline_corpus() -> tuple[list[dict], dict[str, str]]:
    """Paragraphs of data/guidelines/*.md + the query (title + aliases) of each file."""
    paragraphs, queries = [], {}
    for path in sorted((DATA / "guidelines").glob("*.md")):
        title, aliases, source, body = "", "", "", []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            lines = block.strip().splitlines()
            if not lines:
                continue
            if lines[0].startswith("# "):
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                    elif line.startswith("aliases:"):
                        aliases = line.split(":", 1)[1]
                    elif line.startswith("source:"):
                        source = line.split(":", 1)[1].strip()
            else:
                body.append(" ".join(l.strip() for l in lines))
        queries[path.stem] = f"{title} {aliases}"
        paragraphs += [{"text": p, "source": source, "tokens": _tokens(p)} for p in body]
    return paragraphs, queries


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(docs)
    avg = sum(map(len, docs)) / max(n, 1)
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in set(query):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


def guideline_snippets(condition: str, k: int = 2) -> list[dict]:
    """Top-k guideline paragraphs for a finding, ranked with BM25 over ALL notes."""
    try:
        paragraphs, queries = _guideline_corpus()
        if not paragraphs:
            return []
        q = queries.get(label_slug(str(condition)).lower(), str(condition))
        scores = _bm25(_tokens(q), [p["tokens"] for p in paragraphs])
        ranked = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        return [{"text": paragraphs[i]["text"], "source": paragraphs[i]["source"]}
                for i in ranked[:max(1, int(k))] if scores[i] > 0]
    except Exception as e:                                # noqa: BLE001
        log.warning("guideline retrieval failed for %r: %s", condition, e)
        return []
