"""Exa AI-powered search tool.

Wraps the Exa Search API via the exa-py SDK. Exa is a neural/keyword web
search engine with a dedicated `research paper` category, making it useful
for discovering academic papers hosted on publisher sites, institutional
repositories, and preprint servers that may not appear in traditional
scholarly indexes.

Requires a free API key from https://dashboard.exa.ai/api-keys
(set EXA_API_KEY env var).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")

# Tracking header — attributes API usage to this integration.
_INTEGRATION_HEADER = "deep-researcher"

# Exa supports these search types. `auto` lets Exa pick neural vs keyword.
_SEARCH_TYPES = {"auto", "neural", "fast", "keyword"}

# Academic-friendly categories that Exa exposes.
_CATEGORIES = {"research paper", "pdf", "news", "company", "personal site"}


@dataclass
class ExaResult:
    """Typed view of a single Exa search result.

    Exa returns a pydantic-like result object per hit; we normalize it into
    this dataclass so parsing and snippet fallback logic stay explicit.
    """
    title: str
    url: str
    author: str | None = None
    published_date: str | None = None
    text: str | None = None
    highlights: list[str] | None = None
    summary: str | None = None

    @classmethod
    def from_sdk(cls, obj: object) -> ExaResult:
        return cls(
            title=_safe_attr(obj, "title") or "",
            url=_safe_attr(obj, "url") or "",
            author=_safe_attr(obj, "author"),
            published_date=_safe_attr(obj, "published_date"),
            text=_safe_attr(obj, "text"),
            highlights=_safe_attr(obj, "highlights"),
            summary=_safe_attr(obj, "summary"),
        )

    def best_abstract(self) -> str | None:
        """Cascade through available content fields for a usable abstract.

        Exa may return any combination of text/highlights/summary depending
        on the request, so build the snippet in priority order rather than
        assuming one field will be present.
        """
        if self.summary:
            return clean_abstract(self.summary)
        if self.highlights:
            joined = " ... ".join(h for h in self.highlights if h)
            if joined:
                return clean_abstract(joined)
        if self.text:
            return clean_abstract(self.text)
        return None


class ExaSearchTool(Tool):
    name = "search_exa"
    category = "index"
    quality_tier = 2  # Web search — broad coverage, not peer-reviewed by itself
    description = (
        "Search the web with Exa's AI-powered search engine. Uses semantic "
        "(neural) search to surface research papers, preprints, and technical "
        "content that traditional scholarly indexes may miss — including "
        "publisher sites, institutional repositories, blog posts, and PDFs. "
        "Defaults to the `research paper` category for academic queries. "
        "Requires a free API key from https://dashboard.exa.ai/api-keys "
        "(set EXA_API_KEY env var)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language research query. Exa's neural search "
                               "understands descriptive queries better than keyword lists.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 25).",
            },
            "search_type": {
                "type": "string",
                "enum": ["auto", "neural", "fast", "keyword"],
                "description": "Search algorithm. 'auto' (default) lets Exa pick; 'neural' "
                               "for semantic queries; 'keyword' for exact-term lookups; "
                               "'fast' for lower-latency lookups.",
            },
            "category": {
                "type": "string",
                "enum": ["research paper", "pdf", "news", "company", "personal site"],
                "description": "Restrict results to a category. Defaults to 'research paper' "
                               "for this academic tool. Pass an empty string to disable.",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only return results from these domains (e.g. ['arxiv.org']).",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude results from these domains.",
            },
            "include_text": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only include pages containing these strings (max 1 string, "
                               "max 5 words per Exa API).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    def execute(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "auto",
        category: str = "research paper",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_text: list[str] | None = None,
    ) -> ToolResult:
        if not self._api_key:
            return ToolResult(
                text="Exa search is not available (no API key configured). "
                "Get a free key from https://dashboard.exa.ai/api-keys and set EXA_API_KEY."
            )

        max_results = min(max_results, 25)
        if search_type not in _SEARCH_TYPES:
            search_type = "auto"

        # Build request kwargs — only include optional params when set so we
        # don't override Exa's server-side defaults with None.
        kwargs: dict = {
            "type": search_type,
            "num_results": max_results,
            # The `contents` fields are not mutually exclusive; request
            # highlights + text so downstream fallback logic has options.
            "text": {"max_characters": 1500},
            "highlights": {"num_sentences": 3, "highlights_per_url": 2},
        }
        if category and category in _CATEGORIES:
            kwargs["category"] = category
        if include_domains:
            kwargs["include_domains"] = list(include_domains)
        if exclude_domains:
            kwargs["exclude_domains"] = list(exclude_domains)
        if include_text:
            kwargs["include_text"] = list(include_text)
        # Exa supports date filters via ISO-8601 datetimes. Convert year range
        # bounds from the Tool base class into published-date filters.
        if self._start_year is not None:
            kwargs["start_published_date"] = f"{self._start_year}-01-01T00:00:00.000Z"
        if self._end_year is not None:
            kwargs["end_published_date"] = f"{self._end_year}-12-31T23:59:59.999Z"

        try:
            from exa_py import Exa

            client = Exa(api_key=self._api_key)
            # Tracking header — attributes API usage to this integration.
            client.headers["x-exa-integration"] = _INTEGRATION_HEADER
            response = client.search_and_contents(query, **kwargs)
        except ImportError:
            return ToolResult(
                text="Exa search requires the 'exa-py' package. Install with: pip install exa-py"
            )
        except Exception as e:
            logger.debug("Exa search failed: %s", e, exc_info=True)
            return ToolResult(text=f"Error searching Exa: {e}")

        raw_results = getattr(response, "results", []) or []
        if not raw_results:
            return ToolResult(text="No results found on Exa for this query.")

        papers = self._filter_by_year(
            [_result_to_paper(ExaResult.from_sdk(r)) for r in raw_results]
        )
        if not papers:
            return ToolResult(
                text="No results found on Exa for this query (after year filter)."
            )

        lines = [f"Found {len(papers)} results on Exa:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _safe_attr(obj: object, name: str) -> object:
    """Read an attribute off an Exa SDK result object, tolerating missing fields.

    Works for both attribute-style SDK objects and dict-style mocks so tests
    can pass fixtures without constructing full pydantic models.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_doi(url: str | None, text: str | None) -> str | None:
    """Best-effort DOI extraction from URL or content body."""
    if not url and not text:
        return None
    # doi.org URLs embed the DOI directly.
    if url and "doi.org/" in url:
        doi = url.split("doi.org/", 1)[1].split("?")[0].rstrip("/")
        if doi:
            return doi
    # Match a DOI anywhere in available text.
    candidates = " ".join(x for x in (url, text) if x)
    m = re.search(r"\b10\.\d{4,9}/[^\s\"<>)]+", candidates)
    if m:
        return m.group(0).rstrip(".,);")
    return None


def _extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?/]+?)(?:v\d+)?(?:\.pdf)?(?:$|[?#])", url)
    if m:
        return m.group(1)
    return None


def _parse_authors(author: str | None) -> list[str]:
    """Exa returns author as a single string; split common delimiters."""
    if not author:
        return []
    # Try common separators: comma, semicolon, " and ", " & ".
    parts = re.split(r",| and | & |;", author)
    return [p.strip() for p in parts if p.strip()]


def _parse_year(published_date: str | None) -> int | None:
    if not published_date or len(published_date) < 4:
        return None
    try:
        return int(published_date[:4])
    except ValueError:
        return None


def _result_to_paper(r: ExaResult) -> Paper:
    doi = _extract_doi(r.url, r.text)
    arxiv_id = _extract_arxiv_id(r.url)
    return Paper(
        title=r.title,
        authors=_parse_authors(r.author),
        year=_parse_year(r.published_date),
        abstract=r.best_abstract(),
        doi=doi,
        url=r.url,
        source="exa",
        arxiv_id=arxiv_id,
    )
