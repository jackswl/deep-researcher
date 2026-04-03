"""Google Scholar search tool.

Wraps the scholarly library as a proper Tool with validation,
structured output, and error handling (Principle 1: tools as unit of action).
"""
from __future__ import annotations

import logging
import threading

from scholarly import scholarly

from deep_researcher.constants import SCHOLAR_MAX_RESULTS
from deep_researcher.models import Paper, ToolResult
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")


class ScholarSearchTool(Tool):
    name = "scholar_search"
    description = "Search Google Scholar for academic papers by query"
    is_read_only = True
    is_concurrency_safe = False  # Web scraping, single request stream
    category = "index"
    quality_tier = 2
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Research query"},
            "max_results": {"type": "integer", "description": "Maximum papers to return"},
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str = "",
        max_results: int = SCHOLAR_MAX_RESULTS,
        cancel: threading.Event | None = None,
    ) -> ToolResult:
        max_results = min(max_results, SCHOLAR_MAX_RESULTS)
        seen_titles: set[str] = set()
        papers: list[Paper] = []

        try:
            for result in scholarly.search_pubs(query):
                if len(papers) >= max_results:
                    break
                if cancel and cancel.is_set():
                    break

                title = result.get("bib", {}).get("title", "")
                if not title or title.lower().strip() in seen_titles:
                    continue
                seen_titles.add(title.lower().strip())

                bib = result.get("bib", {})
                authors = bib.get("author", [])
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(" and ")]

                year_str = bib.get("pub_year", "")
                year = int(year_str) if year_str and year_str.isdigit() else None

                papers.append(Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=bib.get("abstract", ""),
                    journal=bib.get("venue", ""),
                    citation_count=result.get("num_citations", 0) or None,
                    url=result.get("pub_url", ""),
                    source="google_scholar",
                ))
        except Exception as e:
            logger.warning("Google Scholar search failed: %s", e)

        return ToolResult(
            text=f"Found {len(papers)} papers from Google Scholar",
            papers=papers,
        )
