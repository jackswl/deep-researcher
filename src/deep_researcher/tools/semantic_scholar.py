from __future__ import annotations

import time

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,authors,year,abstract,doi,url,citationCount,journal,externalIds"

_RETRIABLE_STATUSES = {429, 500, 502, 503}
_NON_RETRIABLE_STATUSES = {400, 404}


class SemanticScholarSearchTool(Tool):
    name = "search_semantic_scholar"
    category = "index"
    quality_tier = 1  # Curated academic index with citation data
    description = (
        "Search Semantic Scholar for academic papers. Covers 200M+ papers across "
        "all academic fields. Returns citation counts and has excellent coverage of "
        "computer science, biomedical, and engineering literature."
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

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 20)
        params: dict = {"query": query, "limit": max_results, "fields": S2_FIELDS}
        # Semantic Scholar supports year range filter
        if self._start_year is not None or self._end_year is not None:
            yr_start = self._start_year if self._start_year is not None else 1900
            yr_end = self._end_year if self._end_year is not None else 2100
            params["year"] = f"{yr_start}-{yr_end}"
        try:
            resp = _request_with_retry(
                f"{S2_BASE}/paper/search",
                params=params,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching Semantic Scholar: {e}")

        data = resp.json()
        papers_data = data.get("data", [])
        if not papers_data:
            return ToolResult(text="No papers found on Semantic Scholar for this query.")

        papers = self._filter_by_year([_parse_s2_paper(p) for p in papers_data])
        if not papers:
            return ToolResult(text="No papers found on Semantic Scholar for this query (after year filter).")
        lines = [f"Found {len(papers)} papers on Semantic Scholar:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


class GetCitationsTool(Tool):
    name = "get_citations"
    category = "citation"
    description = (
        "Get papers that cite a given paper, or papers that a given paper references. "
        "Use this to follow citation chains — find foundational work (references) "
        "or recent developments (citations). Requires a Semantic Scholar paper ID or DOI."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Semantic Scholar paper ID, DOI (prefixed with 'DOI:'), or arXiv ID (prefixed with 'ARXIV:').",
            },
            "direction": {
                "type": "string",
                "enum": ["citations", "references"],
                "description": "'citations' for papers that cite this one, 'references' for papers this one cites.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 20).",
            },
        },
        "required": ["paper_id", "direction"],
    }

    def execute(self, paper_id: str, direction: str = "citations", max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 20)
        try:
            fields_param = f"citingPaper.{S2_FIELDS}" if direction == "citations" else f"citedPaper.{S2_FIELDS}"
            resp = _request_with_retry(
                f"{S2_BASE}/paper/{paper_id}/{direction}",
                params={"fields": fields_param, "limit": max_results},
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error fetching {direction}: {e}")

        data = resp.json().get("data", [])
        if not data:
            return ToolResult(text=f"No {direction} found for this paper.")

        paper_key = "citingPaper" if direction == "citations" else "citedPaper"
        papers = [_parse_s2_paper(item[paper_key]) for item in data if item.get(paper_key)]
        lines = [f"Found {len(papers)} {direction}:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _request_with_retry(url: str, params: dict, max_retries: int = 3) -> httpx.Response:
    resp = None
    for attempt in range(max_retries):
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code in _NON_RETRIABLE_STATUSES:
            return resp
        if resp.status_code in _RETRIABLE_STATUSES:
            time.sleep(2 ** (attempt + 1))
            continue
        return resp
    return resp  # type: ignore[return-value]


def _parse_s2_paper(data: dict) -> Paper:
    authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")]
    journal_info = data.get("journal")
    journal = journal_info.get("name") if isinstance(journal_info, dict) else None
    external_ids = data.get("externalIds") or {}
    return Paper(
        title=data.get("title", ""),
        authors=authors,
        year=data.get("year"),
        abstract=clean_abstract(data.get("abstract")),
        doi=external_ids.get("DOI") or data.get("doi"),
        url=data.get("url"),
        source="semantic_scholar",
        citation_count=data.get("citationCount"),
        journal=journal,
        arxiv_id=external_ids.get("ArXiv"),
        pmid=external_ids.get("PubMed"),
    )
