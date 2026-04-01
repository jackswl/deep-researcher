from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

ARXIV_NS = "{http://www.w3.org/2005/Atom}"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class ArxivSearchTool(Tool):
    name = "search_arxiv"
    category = "preprint"
    quality_tier = 3  # Preprints — no peer review
    description = (
        "Search arXiv for preprints and papers. Covers physics, mathematics, "
        "computer science, quantitative biology, statistics, electrical engineering, "
        "systems science, and economics. Great for recent research and preprints."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Use arXiv search syntax: 'all:' for all fields, 'ti:' for title, 'au:' for author, 'abs:' for abstract.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 10, max 30).",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 30)
        if not any(prefix in query for prefix in ("all:", "ti:", "au:", "abs:", "cat:")):
            query = f"all:{query}"

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    "https://export.arxiv.org/api/query",
                    params={"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance"},
                    timeout=30,
                    follow_redirects=True,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching arXiv: {e}")

        papers = self._filter_by_year(_parse_arxiv_response(resp.text))
        if not papers:
            return ToolResult(text="No papers found on arXiv for this query.")

        lines = [f"Found {len(papers)} papers on arXiv:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_arxiv_response(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ARXIV_NS}entry"):
        title_el = entry.find(f"{ARXIV_NS}title")
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        if not title:
            continue

        summary_el = entry.find(f"{ARXIV_NS}summary")
        raw_abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else None
        abstract = clean_abstract(raw_abstract)

        authors = []
        for author_el in entry.findall(f"{ARXIV_NS}author"):
            name_el = author_el.find(f"{ARXIV_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        published_el = entry.find(f"{ARXIV_NS}published")
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except (ValueError, IndexError):
                pass

        arxiv_id = None
        url = None
        for link_el in entry.findall(f"{ARXIV_NS}link"):
            href = link_el.get("href", "")
            if link_el.get("title") == "pdf":
                url = href
            elif "/abs/" in href:
                url = url or href
                arxiv_id = href.split("/abs/")[-1]

        doi = None
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                doi=doi,
                url=url,
                source="arxiv",
                arxiv_id=arxiv_id,
            )
        )
    return papers
