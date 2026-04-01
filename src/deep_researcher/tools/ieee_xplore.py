from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

IEEE_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class IEEEXploreSearchTool(Tool):
    name = "search_ieee"
    category = "publisher"
    quality_tier = 1  # Peer-reviewed IEEE/IET publications
    description = (
        "Search IEEE Xplore for engineering and computer science papers. Covers 6M+ "
        "articles from IEEE journals, conferences, and standards, plus IET publications. "
        "Returns abstracts, citation counts, and metadata. Strong for electrical engineering, "
        "computer science, and related fields. "
        "Requires a free API key from developer.ieee.org (set IEEE_API_KEY env var)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Searches across metadata, abstracts, and index terms.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 25).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        if not self._api_key:
            return ToolResult(
                text="IEEE Xplore search is not available (no API key configured). "
                "Get a free key from https://developer.ieee.org/ and set IEEE_API_KEY."
            )

        max_results = min(max_results, 25)

        ieee_params: dict = {
            "apikey": self._api_key,
            "querytext": query,
            "max_records": max_results,
            "sort_field": "publication_year",
            "sort_order": "desc",
        }
        # IEEE Xplore supports year range natively
        if self._start_year is not None:
            ieee_params["start_year"] = self._start_year
        if self._end_year is not None:
            ieee_params["end_year"] = self._end_year

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    IEEE_BASE,
                    params=ieee_params,
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            if resp.status_code == 403:
                return ToolResult(text="IEEE Xplore API key is invalid or rate-limited. Check your IEEE_API_KEY.")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching IEEE Xplore: {e}")

        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return ToolResult(text="No papers found on IEEE Xplore for this query.")

        papers = self._filter_by_year([p for a in articles if (p := _parse_ieee_article(a)) is not None])
        if not papers:
            return ToolResult(text="No papers found on IEEE Xplore for this query.")

        lines = [f"Found {len(papers)} papers on IEEE Xplore:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_ieee_article(data: dict) -> Paper | None:
    title = data.get("title", "")
    if not title:
        return None

    # Authors
    authors = []
    author_data = data.get("authors", {})
    author_list = author_data.get("authors", []) if isinstance(author_data, dict) else []
    for a in author_list:
        name = a.get("full_name", "")
        if name:
            authors.append(name)

    year = None
    pub_year = data.get("publication_year")
    if pub_year:
        try:
            year = int(pub_year)
        except (ValueError, TypeError):
            pass

    abstract = clean_abstract(data.get("abstract"))
    doi = data.get("doi")

    cited_by = data.get("citing_paper_count")
    citation_count = None
    if cited_by is not None:
        try:
            citation_count = int(cited_by)
        except (ValueError, TypeError):
            pass

    journal = data.get("publication_title")
    volume = data.get("volume")

    start_page = data.get("start_page", "")
    end_page = data.get("end_page", "")
    pages = f"{start_page}-{end_page}" if start_page and end_page else start_page or None

    publisher = data.get("publisher")

    url = data.get("html_url") or data.get("pdf_url")
    if not url and doi:
        url = f"https://doi.org/{doi}"

    # IEEE open access
    open_access_url = None
    if data.get("access_type") == "OPEN_ACCESS" and url:
        open_access_url = url

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        source="ieee",
        citation_count=citation_count,
        journal=journal,
        volume=volume,
        pages=pages,
        publisher=publisher,
        open_access_url=open_access_url,
    )
