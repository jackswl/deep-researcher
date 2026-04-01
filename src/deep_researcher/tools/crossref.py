from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

CROSSREF_BASE = "https://api.crossref.org"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class CrossrefSearchTool(Tool):
    name = "search_crossref"
    category = "publisher"
    quality_tier = 2  # Broad publisher metadata, includes all DOI-registered works
    description = (
        "Search CrossRef for academic papers by DOI metadata. Covers 150M+ records "
        "from most major publishers (Elsevier, Springer, Wiley, IEEE, etc.). "
        "Best for finding papers from traditional publishers and getting accurate metadata."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for finding papers."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 20).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, email: str = "") -> None:
        self._email = email

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 20)
        headers = {}
        if self._email:
            headers["User-Agent"] = f"DeepResearcher/0.2 (mailto:{self._email})"

        params: dict = {"query": query, "rows": max_results, "sort": "relevance"}
        # CrossRef supports date filtering via filter param
        filters = []
        if self._start_year is not None:
            filters.append(f"from-pub-date:{self._start_year}")
        if self._end_year is not None:
            filters.append(f"until-pub-date:{self._end_year}")
        if filters:
            params["filter"] = ",".join(filters)

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    f"{CROSSREF_BASE}/works",
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching CrossRef: {e}")

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return ToolResult(text="No papers found on CrossRef for this query.")

        papers = []
        for item in items:
            titles = item.get("title") or []
            if titles and titles[0]:
                papers.append(_parse_crossref_item(item))

        papers = self._filter_by_year(papers)
        if not papers:
            return ToolResult(text="No papers found on CrossRef for this query.")

        lines = [f"Found {len(papers)} papers on CrossRef:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_crossref_item(data: dict) -> Paper:
    title_list = data.get("title") or []
    title = title_list[0] if title_list else ""

    authors = []
    for a in data.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    year = None
    pub_info = data.get("published-print") or data.get("published-online") or {}
    date_parts = pub_info.get("date-parts", [[]]) if isinstance(pub_info, dict) else [[]]
    if date_parts and date_parts[0]:
        try:
            year = int(date_parts[0][0])
        except (ValueError, TypeError, IndexError):
            pass

    doi = data.get("DOI")
    abstract = clean_abstract(data.get("abstract"))

    cited_by = data.get("is-referenced-by-count")

    container = data.get("container-title", [])
    journal = container[0] if container else None

    publisher = data.get("publisher")

    url = data.get("URL")

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        source="crossref",
        citation_count=cited_by,
        journal=journal,
        publisher=publisher,
    )
