import arxiv
import requests
import json
import hashlib
import os
import time
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CACHE LAYER
# Avoids redundant API calls for same queries.
# Stores results in cache.json, expires after 24h.
#
# FIX: _load_cache now evicts expired entries on every
# load so the file doesn't grow unbounded forever.
# ─────────────────────────────────────────────

CACHE_FILE = "cache.json"
CACHE_TTL  = timedelta(hours=24)


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # Evict expired entries so cache.json doesn't grow forever.
    # Only rewrite the file if something was actually removed.
    now = datetime.now()
    cleaned = {
        key: val for key, val in cache.items()
        if now - datetime.fromisoformat(val["timestamp"]) < CACHE_TTL
    }
    if len(cleaned) != len(cache):
        _save_cache(cleaned)
        print(f"[Cache] Evicted {len(cache) - len(cleaned)} expired entries.")
    return cleaned


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(source: str, query: str) -> str:
    # MD5 of "source:query" — deterministic, fixed-length key.
    # Not a security function so MD5 collision risk is acceptable.
    return hashlib.md5(f"{source}:{query}".encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        cached_time = datetime.fromisoformat(entry["timestamp"])
        if datetime.now() - cached_time < CACHE_TTL:
            print(f"[Cache HIT] Returning cached result.")
            return entry["result"]
    return None


def _set_cached(key: str, result: str):
    cache = _load_cache()
    cache[key] = {
        "timestamp": datetime.now().isoformat(),
        "result": result
    }
    _save_cache(cache)


# ─────────────────────────────────────────────
# RETRY HELPER
# Exponential backoff: waits 1s, 2s, 4s between
# attempts before giving up.
# Used by Tavily and Semantic Scholar fetchers.
# ─────────────────────────────────────────────

MAX_RETRIES = 3


def _with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) up to MAX_RETRIES times.
    On failure, waits 2^attempt seconds before retrying.
    Returns the result on success, raises on final failure.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt          # 1s → 2s → 4s
            print(f"[Retry {attempt + 1}/{MAX_RETRIES}] Error: {e}. Waiting {wait}s...")
            time.sleep(wait)
    raise last_error


# ─────────────────────────────────────────────
# TOOL 1: arXiv Search
# For technical/research angle.
# Uses the arxiv library — no API key needed.
# ─────────────────────────────────────────────

def fetch_arxiv_papers(query: str, max_results: int = 4) -> str:
    """
    Searches arXiv for academic papers on a topic.
    Returns titles, links, and summaries.
    Results are cached for 24 hours.
    """
    key = _cache_key("arxiv", query)
    cached = _get_cached(key)
    if cached:
        return cached

    print(f"[arXiv] Searching: '{query}'")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for paper in client.results(search):
        results.append(
            f"Title: {paper.title}\n"
            f"Link: {paper.entry_id}\n"
            f"Summary: {paper.summary[:400]}...\n"
        )

    if not results:
        return "No arXiv papers found for this query."

    output = "\n---\n".join(results)
    _set_cached(key, output)
    return output


# ─────────────────────────────────────────────
# TOOL 2: Semantic Scholar Search
# Second academic source — free, no API key needed.
# Provides citation counts the Critic uses for scoring.
# Deduplication of arXiv overlap happens in the Critic agent.
# ─────────────────────────────────────────────

def fetch_semantic_scholar(query: str, max_results: int = 4) -> str:
    """
    Searches Semantic Scholar for papers with citation counts.
    Returns titles, citation counts, links, and abstracts.
    Results are cached for 24 hours.
    Retries up to 3 times with exponential backoff on failure.
    """
    key = _cache_key("semantic_scholar", query)
    cached = _get_cached(key)
    if cached:
        return cached

    print(f"[Semantic Scholar] Searching: '{query}'")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,citationCount,url,year"
    }

    def _fetch():
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    try:
        data = _with_retry(_fetch)
    except Exception as e:
        return f"Semantic Scholar search failed after {MAX_RETRIES} attempts: {e}"

    papers = data.get("data", [])
    if not papers:
        return "No Semantic Scholar papers found for this query."

    results = []
    for p in papers:
        results.append(
            f"Title: {p.get('title', 'N/A')}\n"
            f"Year: {p.get('year', 'N/A')} | Citations: {p.get('citationCount', 0)}\n"
            f"Link: {p.get('url', 'N/A')}\n"
            f"Abstract: {(p.get('abstract') or 'No abstract available.')[:400]}...\n"
        )

    output = "\n---\n".join(results)
    _set_cached(key, output)
    return output


# ─────────────────────────────────────────────
# TOOL 3: Tavily Web Search
# For news and market intelligence angle.
# Requires TAVILY_API_KEY from user (BYOK).
# Retries up to 3 times with exponential backoff on failure.
# ─────────────────────────────────────────────

def fetch_web_news(query: str, tavily_api_key: str, max_results: int = 5) -> str:
    """
    Searches the live web for recent news and market intelligence.
    Uses Tavily Search API (free tier: 1000 searches/month).
    Results are cached for 24 hours.
    Retries up to 3 times with exponential backoff on failure.
    """
    key = _cache_key("tavily", query)
    cached = _get_cached(key)
    if cached:
        return cached

    print(f"[Web Search] Searching news: '{query}'")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True
    }

    def _fetch():
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    try:
        data = _with_retry(_fetch)
    except Exception as e:
        return f"Web search failed after {MAX_RETRIES} attempts: {e}"

    results = []
    if data.get("answer"):
        results.append(f"Quick Answer: {data['answer']}\n")

    for r in data.get("results", []):
        results.append(
            f"Title: {r.get('title', 'N/A')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Content: {r.get('content', '')[:400]}...\n"
        )

    if not results:
        return "No web results found."

    output = "\n---\n".join(results)
    _set_cached(key, output)
    return output
