import arxiv
import requests
import json
import hashlib
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CACHE LAYER
# Avoids redundant API calls for same queries.
# Stores results in cache.json, expires after 24h.
# ─────────────────────────────────────────────

CACHE_FILE = "cache.json"

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def _cache_key(source: str, query: str) -> str:
    return hashlib.md5(f"{source}:{query}".encode()).hexdigest()

def _get_cached(key: str) -> str | None:
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        cached_time = datetime.fromisoformat(entry["timestamp"])
        if datetime.now() - cached_time < timedelta(hours=24):
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
# TOOL 1: arXiv Search
# For technical/research angle
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
# Deduplication happens in the Critic agent.
# ─────────────────────────────────────────────

def fetch_semantic_scholar(query: str, max_results: int = 4) -> str:
    """
    Searches Semantic Scholar for papers with citation counts.
    Returns titles, citation counts, links, and abstracts.
    Results are cached for 24 hours.
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

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Semantic Scholar search failed: {str(e)}"

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
# Requires TAVILY_API_KEY from user.
# ─────────────────────────────────────────────

def fetch_web_news(query: str, tavily_api_key: str, max_results: int = 5) -> str:
    """
    Searches the live web for recent news and market intelligence.
    Uses Tavily Search API (free tier: 1000 searches/month).
    Results are cached for 24 hours.
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

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Web search failed: {str(e)}"

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
