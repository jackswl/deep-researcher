from __future__ import annotations

import hashlib
import html
import re
import copy
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str = ""
    citation_count: int | None = None
    journal: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    open_access_url: str | None = None
    keywords: list[str] = field(default_factory=list)
    volume: str | None = None
    pages: str | None = None
    publisher: str | None = None

    @property
    def unique_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower().strip()}"
        if self.pmid:
            return f"pmid:{self.pmid.strip()}"
        normalized = re.sub(r"\s+", " ", self.title.lower().strip())
        return f"title:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"

    def merge(self, other: Paper) -> None:
        """Merge metadata from another Paper representing the same work."""
        if other.abstract and len(other.abstract) > len(self.abstract or ""):
            self.abstract = other.abstract
        if not self.doi and other.doi:
            self.doi = other.doi
        if not self.url and other.url:
            self.url = other.url
        if other.citation_count is not None:
            if self.citation_count is None or other.citation_count > self.citation_count:
                self.citation_count = other.citation_count
        if not self.journal and other.journal:
            self.journal = other.journal
        if not self.arxiv_id and other.arxiv_id:
            self.arxiv_id = other.arxiv_id
        if not self.pmid and other.pmid:
            self.pmid = other.pmid
        if not self.open_access_url and other.open_access_url:
            self.open_access_url = other.open_access_url
        if not self.year and other.year:
            self.year = other.year
        if not self.authors and other.authors:
            self.authors = other.authors
        if other.source and other.source not in self.source:
            self.source = f"{self.source},{other.source}"

    def to_summary(self) -> str:
        parts = [f"**{self.title}**"]
        if self.authors:
            author_str = self.authors[0]
            if len(self.authors) > 1:
                author_str += " et al."
            parts.append(f"Authors: {author_str}")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.journal:
            parts.append(f"Journal: {self.journal}")
        if self.citation_count is not None:
            parts.append(f"Citations: {self.citation_count}")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        if self.abstract:
            abstract = self.abstract
            if len(abstract) > 300:
                cut = abstract[:320].rfind(". ")
                abstract = abstract[: cut + 1] if cut > 200 else abstract[:300] + "..."
            parts.append(f"Abstract: {abstract}")
        if self.open_access_url:
            parts.append(f"Open Access: {self.open_access_url}")
        return "\n".join(parts)

    def to_bibtex(self, key_suffix: str = "") -> str:
        if self.doi:
            key = re.sub(r"[^a-zA-Z0-9]", "_", self.doi)
        else:
            author_part = "unknown"
            if self.authors:
                parts = self.authors[0].split()
                author_part = parts[-1].lower() if parts else "unknown"
            year_part = str(self.year) if self.year else "nd"
            title_part = "untitled"
            if self.title:
                words = self.title.split()
                if words:
                    cleaned = re.sub(r"[^a-z]", "", words[0].lower())
                    title_part = cleaned or "untitled"
            key = f"{author_part}{year_part}{title_part}{key_suffix}"

        entry_type = "article" if self.journal else "misc"
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{_bib_escape(self.title)}}},")
        if self.authors:
            lines.append(f"  author = {{{' and '.join(_bib_escape(a) for a in self.authors)}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if self.journal:
            lines.append(f"  journal = {{{_bib_escape(self.journal)}}},")
        if self.volume:
            lines.append(f"  volume = {{{self.volume}}},")
        if self.pages:
            lines.append(f"  pages = {{{self.pages}}},")
        if self.publisher:
            lines.append(f"  publisher = {{{_bib_escape(self.publisher)}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "url": self.url,
            "source": self.source,
            "citation_count": self.citation_count,
            "journal": self.journal,
            "volume": self.volume,
            "pages": self.pages,
            "publisher": self.publisher,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "open_access_url": self.open_access_url,
            "keywords": self.keywords,
        }


@dataclass
class ToolResult:
    """Structured result from a tool execution."""
    text: str
    papers: list[Paper] = field(default_factory=list)
    data: Any = None


@dataclass
class PipelineState:
    """State that flows through the research pipeline.

    Each phase receives the current state and returns a new state via evolve().
    Container fields (dicts, lists) are shallow-copied so adding/removing entries
    in one state doesn't affect another. Paper objects within the dict are shared
    references — callers must not mutate individual Papers after evolve().
    Pipeline tools (enrichment, search) return fresh Paper objects, so this is
    safe in practice.
    """
    query: str
    papers: dict[str, Paper] = field(default_factory=dict)
    categories: dict[str, list[int]] | None = None
    synthesis_papers: list[Paper] = field(default_factory=list)
    category_sections: list[tuple[str, str]] = field(default_factory=list)
    cross_section: str = ""
    report: str = ""

    def evolve(self, **kwargs: Any) -> PipelineState:
        """Return a new PipelineState with specified fields replaced.

        Non-overridden mutable fields are shallow-copied to prevent
        accidental cross-state mutation.
        """
        # Deep-copy mutable fields that aren't being replaced
        defaults: dict[str, Any] = {}
        if "papers" not in kwargs:
            defaults["papers"] = copy.copy(self.papers)
        if "synthesis_papers" not in kwargs:
            defaults["synthesis_papers"] = list(self.synthesis_papers)
        if "category_sections" not in kwargs:
            defaults["category_sections"] = list(self.category_sections)
        if "categories" not in kwargs and self.categories is not None:
            defaults["categories"] = {k: list(v) for k, v in self.categories.items()}
        return replace(self, **defaults, **kwargs)


def clean_abstract(text: str | None) -> str | None:
    if not text:
        return None
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _bib_escape(s: str) -> str:
    return s.replace("{", r"\{").replace("}", r"\}")
