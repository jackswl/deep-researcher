from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

OPENALEX_BASE = "https://api.openalex.org"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class OpenAlexSearchTool(Tool):
    name = "search_openalex"
    category = "open_access"
    quality_tier = 2  # Broad open metadata, includes all publication types
    description = (
        "Search OpenAlex for academic papers. Covers 250M+ works across all fields. "
        "Fully open dataset with excellent metadata coverage. Good for broad searches "
        "and finding works that may not appear in other databases."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for finding papers."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 25).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, email: str = "") -> None:
        self._email = email

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 25)
        params: dict = {"search": query, "per_page": max_results}
        if self._email:
            params["mailto"] = self._email
        # OpenAlex supports year range via filter param
        year_filters = []
        if self._start_year is not None:
            year_filters.append(f"publication_year:>{self._start_year - 1}")
        if self._end_year is not None:
            year_filters.append(f"publication_year:<{self._end_year + 1}")
        if year_filters:
            params["filter"] = ",".join(year_filters)

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(f"{OPENALEX_BASE}/works", params=params, timeout=30)
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching OpenAlex: {e}")

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ToolResult(text="No papers found on OpenAlex for this query.")

        papers = self._filter_by_year([_parse_openalex_work(w) for w in results])
        if not papers:
            return ToolResult(text="No papers found on OpenAlex for this query (after year filter).")
        lines = [f"Found {len(papers)} papers on OpenAlex:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    try:
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)
    except (TypeError, ValueError, AttributeError):
        return None


def _parse_openalex_work(data: dict) -> Paper:
    title = data.get("title") or ""
    authorships = data.get("authorships", [])
    authors = []
    for a in authorships:
        author = a.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)

    year = data.get("publication_year")
    doi_url = data.get("doi") or ""
    doi = None
    if doi_url:
        doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")

    raw_abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))
    abstract = clean_abstract(raw_abstract)

    cited_by = data.get("cited_by_count")

    source = data.get("primary_location", {}) or {}
    source_info = source.get("source", {}) or {}
    journal = source_info.get("display_name")

    oa = data.get("open_access", {}) or {}
    oa_url = oa.get("oa_url")

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=doi_url or data.get("id"),
        source="openalex",
        citation_count=cited_by,
        journal=journal,
        open_access_url=oa_url if oa_url else None,
    )
