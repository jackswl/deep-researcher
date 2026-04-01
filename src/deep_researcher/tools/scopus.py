from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

SCOPUS_BASE = "https://api.elsevier.com/content/search/scopus"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class ScopusSearchTool(Tool):
    name = "search_scopus"
    category = "publisher"
    quality_tier = 1  # Peer-reviewed, curated by Elsevier
    description = (
        "Search Scopus (Elsevier) for academic papers. Covers 90M+ records from most "
        "major publishers including Elsevier, Springer, Wiley, IEEE, ASCE, and ACM. "
        "Returns abstracts, citation counts, and metadata even for paywalled papers. "
        "Excellent for engineering, medicine, and sciences. "
        "Requires a free API key from dev.elsevier.com (set SCOPUS_API_KEY env var)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Supports Scopus syntax: TITLE-ABS-KEY(term), AU-ID(), ISSN(), etc. Plain text also works.",
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
                text="Scopus search is not available (no API key configured). "
                "Get a free key from https://dev.elsevier.com/ and set SCOPUS_API_KEY."
            )

        max_results = min(max_results, 25)

        # Wrap plain text queries in TITLE-ABS-KEY() for better results
        if not any(kw in query.upper() for kw in ("TITLE-ABS-KEY", "TITLE(", "ABS(", "KEY(", "AU-ID", "ISSN(")):
            query = f"TITLE-ABS-KEY({query})"
        # Scopus supports year filtering via PUBYEAR in query
        if self._start_year is not None:
            query += f" AND PUBYEAR > {self._start_year - 1}"
        if self._end_year is not None:
            query += f" AND PUBYEAR < {self._end_year + 1}"

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    SCOPUS_BASE,
                    params={
                        "query": query,
                        "count": max_results,
                        "sort": "relevancy",
                        "field": "dc:title,dc:creator,author,prism:coverDate,"
                                 "dc:description,prism:doi,citedby-count,"
                                 "prism:publicationName,prism:volume,prism:pageRange,"
                                 "prism:aggregationType,subtypeDescription,"
                                 "openaccessFlag,link",
                    },
                    headers={
                        "X-ELS-APIKey": self._api_key,
                        "Accept": "application/json",
                    },
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            if resp.status_code == 401:
                return ToolResult(text="Scopus API key is invalid. Check your SCOPUS_API_KEY.")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching Scopus: {e}")

        data = resp.json()
        results = data.get("search-results", {}).get("entry", [])

        # Scopus returns an error entry when no results found
        if not results or (len(results) == 1 and results[0].get("error")):
            return ToolResult(text="No papers found on Scopus for this query.")

        papers = self._filter_by_year([p for r in results if (p := _parse_scopus_entry(r)) is not None])
        if not papers:
            return ToolResult(text="No papers found on Scopus for this query.")

        lines = [f"Found {len(papers)} papers on Scopus:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_scopus_entry(data: dict) -> Paper | None:
    title = data.get("dc:title", "")
    if not title or data.get("error"):
        return None

    # Try structured author list first, fall back to dc:creator (first author only)
    authors = []
    author_list = data.get("author", [])
    if isinstance(author_list, list):
        for a in author_list:
            if isinstance(a, dict):
                name = a.get("authname") or a.get("given-name", "") + " " + a.get("surname", "")
                name = name.strip()
                if name:
                    authors.append(name)
    if not authors:
        creator = data.get("dc:creator")
        if creator:
            authors.append(creator)

    # Year from cover date (format: YYYY-MM-DD)
    year = None
    cover_date = data.get("prism:coverDate", "")
    if cover_date and len(cover_date) >= 4:
        try:
            year = int(cover_date[:4])
        except ValueError:
            pass

    abstract = clean_abstract(data.get("dc:description"))
    doi = data.get("prism:doi")
    cited_by = data.get("citedby-count")
    citation_count = int(cited_by) if cited_by else None

    journal = data.get("prism:publicationName")
    volume = data.get("prism:volume")
    pages = data.get("prism:pageRange")

    # Build URL from Scopus link
    url = None
    links = data.get("link", [])
    for link in links:
        if isinstance(link, dict) and link.get("@ref") == "scopus":
            url = link.get("@href")
            break
    if not url and doi:
        url = f"https://doi.org/{doi}"

    # Open access detection
    oa_flag = data.get("openaccessFlag")
    open_access_url = url if oa_flag == "true" else None

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        source="scopus",
        citation_count=citation_count,
        journal=journal,
        volume=volume,
        pages=pages,
        open_access_url=open_access_url,
    )
