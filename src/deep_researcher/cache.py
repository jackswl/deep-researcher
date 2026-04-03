"""Local disk cache for Phase 1+2 search results.

Cache is keyed by a hash of the normalized query and stores enriched papers
so that Phase 3 (LLM synthesis) can be retried without re-fetching.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

from deep_researcher.models import Paper

logger = logging.getLogger("deep_researcher")

_CACHE_DIR = os.path.expanduser("~/.deep-researcher/cache")
_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


def _cache_path(query: str) -> str:
    return os.path.join(_CACHE_DIR, f"{_cache_key(query)}.json")


def load(query: str) -> dict[str, Paper] | None:
    """Return cached enriched papers for *query*, or None if missing/expired."""
    path = _cache_path(query)
    if not os.path.isfile(path):
        return None

    age = time.time() - os.path.getmtime(path)
    if age > _TTL_SECONDS:
        logger.debug("Cache expired for query (%.0f days old)", age / 86400)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        papers: dict[str, Paper] = {}
        for item in data:
            p = Paper(
                title=item.get("title", ""),
                authors=item.get("authors", []),
                year=item.get("year"),
                abstract=item.get("abstract"),
                doi=item.get("doi"),
                url=item.get("url"),
                source=item.get("source", ""),
                citation_count=item.get("citation_count"),
                journal=item.get("journal"),
                volume=item.get("volume"),
                pages=item.get("pages"),
                publisher=item.get("publisher"),
                arxiv_id=item.get("arxiv_id"),
                pmid=item.get("pmid"),
                open_access_url=item.get("open_access_url"),
                keywords=item.get("keywords", []),
            )
            papers[p.unique_key] = p
        logger.debug("Cache hit: loaded %d papers for query", len(papers))
        return papers
    except Exception as e:
        logger.warning("Cache read failed, ignoring: %s", e)
        return None


def save(query: str, papers: dict[str, Paper]) -> None:
    """Persist enriched papers to cache."""
    path = _cache_path(query)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        papers_list = [p.to_dict() for p in papers.values() if p.title]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(papers_list, f, indent=2, ensure_ascii=False)
        logger.debug("Cache saved: %d papers for query", len(papers_list))
    except Exception as e:
        logger.warning("Cache write failed, ignoring: %s", e)


def invalidate(query: str) -> bool:
    """Delete the cache entry for *query*. Returns True if it existed."""
    path = _cache_path(query)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False
