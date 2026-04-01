from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

CORE_BASE = "https://api.core.ac.uk/v3"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class CoreSearchTool(Tool):
    name = "search_core"
    category = "open_access"
    quality_tier = 3  # Open access aggregator — mixed quality
    description = (
        "Search CORE for open access academic papers. Covers 300M+ open access articles "
        "and metadata from repositories worldwide. Good for finding free full-text versions "
        "of papers. Requires a free CORE API key (set CORE_API_KEY env var)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 20).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        if not self._api_key:
            return ToolResult(
                text="CORE search is not available (no API key configured). Set CORE_API_KEY environment variable with a free key from https://core.ac.uk/api-keys/register."
            )

        max_results = min(max_results, 20)
        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    f"{CORE_BASE}/search/works",
                    params={"q": query, "limit": max_results},
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching CORE: {e}")

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ToolResult(text="No papers found on CORE for this query.")

        papers = self._filter_by_year([_parse_core_work(w) for w in results])
        if not papers:
            return ToolResult(text="No papers found on CORE for this query (after year filter).")
        lines = [f"Found {len(papers)} open access papers on CORE:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_core_work(data: dict) -> Paper:
    title = data.get("title", "")

    authors = []
    for a in data.get("authors", []):
        name = a.get("name", "")
        if not name:
            # Fallback: some CORE records use first_name / last_name
            first = a.get("first_name", "")
            last = a.get("last_name", "")
            name = f"{first} {last}".strip()
        if name:
            authors.append(name)

    year = data.get("yearPublished")
    abstract = clean_abstract(data.get("abstract"))
    doi = data.get("doi")

    download_url = data.get("downloadUrl")

    source_urls = data.get("sourceFulltextUrls")
    if isinstance(source_urls, list) and source_urls:
        fallback_url = source_urls[0]
    else:
        fallback_url = None
    url = download_url or fallback_url

    journals = data.get("journals") or []
    journal = journals[0].get("title") if journals else None

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        source="core",
        journal=journal,
        open_access_url=download_url,
    )
