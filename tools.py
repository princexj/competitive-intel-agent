"""External research tools with structured results and a small disk cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import arxiv
import requests

from state import Source

CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "search_cache.json"
CACHE_TTL = timedelta(hours=24)
USER_AGENT = "competitive-intel-agent/2.0"


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = CACHE_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temp_file.replace(CACHE_FILE)


def _cache_key(source: str, query: str, max_results: int) -> str:
    value = f"{source}:{query.strip().lower()}:{max_results}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> list[Source] | None:
    entry = _load_cache().get(key)
    if not entry:
        return None
    try:
        cached_at = datetime.fromisoformat(entry["timestamp"])
        if datetime.now(timezone.utc) - cached_at < CACHE_TTL:
            return entry["results"]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _set_cached(key: str, results: list[Source]) -> None:
    cache = _load_cache()
    cache[key] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    _save_cache(cache)


def _clean(text: str | None, limit: int = 700) -> str:
    return " ".join((text or "").split())[:limit]


def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[Source]:
    key = _cache_key("arxiv", query, max_results)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: list[Source] = []
    for paper in arxiv.Client().results(search):
        results.append(
            {
                "title": paper.title,
                "url": paper.entry_id,
                "snippet": _clean(paper.summary),
                "source": "arXiv",
                "published": paper.published.date().isoformat(),
                "citations": None,
                "category": "research",
            }
        )
    _set_cached(key, results)
    return results


def fetch_semantic_scholar(query: str, max_results: int = 5) -> list[Source]:
    key = _cache_key("semantic_scholar", query, max_results)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,citationCount,url,year,publicationDate",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    results: list[Source] = []
    for paper in response.json().get("data", []):
        results.append(
            {
                "title": paper.get("title") or "Untitled paper",
                "url": paper.get("url") or "",
                "snippet": _clean(paper.get("abstract") or "No abstract available."),
                "source": "Semantic Scholar",
                "published": paper.get("publicationDate")
                or str(paper.get("year") or ""),
                "citations": paper.get("citationCount") or 0,
                "category": "research",
            }
        )
    _set_cached(key, results)
    return results


def fetch_web_results(
    query: str,
    tavily_api_key: str,
    *,
    category: str = "news",
    max_results: int = 6,
) -> list[Source]:
    if not tavily_api_key:
        return []

    key = _cache_key(f"tavily_{category}", query, max_results)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    payload: dict[str, Any] = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "topic": "news" if category == "news" else "general",
        "max_results": max_results,
        "include_answer": False,
    }
    if category == "community":
        payload["include_domains"] = [
            "reddit.com",
            "news.ycombinator.com",
            "dev.to",
            "github.com",
        ]

    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    results = [
        {
            "title": item.get("title") or "Untitled result",
            "url": item.get("url") or "",
            "snippet": _clean(item.get("content")),
            "source": "Tavily",
            "published": item.get("published_date") or "",
            "citations": None,
            "category": category,
        }
        for item in response.json().get("results", [])
    ]
    _set_cached(key, results)
    return results


def deduplicate_sources(sources: list[Source]) -> list[Source]:
    """Deduplicate by normalized URL, falling back to normalized title."""
    unique: dict[str, Source] = {}
    for source in sources:
        key = source["url"].rstrip("/").lower()
        if not key:
            key = " ".join(source["title"].lower().split())
        current = unique.get(key)
        if current is None or len(source["snippet"]) > len(current["snippet"]):
            unique[key] = source
    return list(unique.values())


# Backward-compatible helper for older callers.
def fetch_web_news(
    query: str, tavily_api_key: str, max_results: int = 5
) -> list[Source]:
    return fetch_web_results(
        query, tavily_api_key, category="news", max_results=max_results
    )
