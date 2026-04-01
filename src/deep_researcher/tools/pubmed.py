from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import httpx

from deep_researcher.models import Paper, ToolResult, clean_abstract
from deep_researcher.tools.base import Tool

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_RETRIABLE_STATUSES = {429, 500, 502, 503}


class PubMedSearchTool(Tool):
    name = "search_pubmed"
    category = "index"
    quality_tier = 1  # Curated biomedical index (NLM)
    description = (
        "Search PubMed for biomedical and life sciences literature. Covers 36M+ "
        "citations including biomedicine, health, genomics, and related fields. "
        "Use this for medical, biological, pharmaceutical, and health-related research."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for PubMed."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 20).",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> ToolResult:
        max_results = min(max_results, 20)

        search_params: dict = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        # PubMed supports date range via mindate/maxdate
        if self._start_year is not None or self._end_year is not None:
            search_params["datetype"] = "pdat"
            if self._start_year is not None:
                search_params["mindate"] = str(self._start_year)
            if self._end_year is not None:
                search_params["maxdate"] = str(self._end_year)

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    f"{EUTILS_BASE}/esearch.fcgi",
                    params=search_params,
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error searching PubMed: {e}")

        search_data = resp.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return ToolResult(text="No papers found on PubMed for this query.")

        try:
            resp = None
            for attempt in range(3):
                resp = httpx.get(
                    f"{EUTILS_BASE}/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
                    timeout=30,
                )
                if resp.status_code in _RETRIABLE_STATUSES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(text=f"Error fetching PubMed details: {e}")

        papers = self._filter_by_year(_parse_pubmed_xml(resp.text))
        if not papers:
            return ToolResult(text="No papers found on PubMed for this query.")

        lines = [f"Found {len(papers)} papers on PubMed:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.to_summary()}\n")
        return ToolResult(text="\n".join(lines), papers=papers)


def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find(".//MedlineCitation")
        if medline is None:
            continue

        article_el = medline.find(".//Article")
        if article_el is None:
            continue

        title_el = article_el.find(".//ArticleTitle")
        title = _get_text(title_el)
        if not title:
            continue

        # Collect ALL AbstractText elements (structured abstracts have multiple)
        abstract_parts = []
        for abs_el in article_el.findall(".//Abstract/AbstractText"):
            label = abs_el.get("Label")
            text = _get_text(abs_el)
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        raw_abstract = " ".join(abstract_parts) if abstract_parts else None
        abstract = clean_abstract(raw_abstract)

        authors = []
        for author_el in article_el.findall(".//AuthorList/Author"):
            last = _get_text(author_el.find("LastName"))
            first = _get_text(author_el.find("ForeName"))
            if last:
                name = f"{first} {last}".strip() if first else last
                authors.append(name)

        year = None
        year_el = article_el.find(".//Journal/JournalIssue/PubDate/Year")
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text)
            except ValueError:
                pass

        # MedlineDate fallback (e.g. "2023 Jan-Feb" or "2023 Spring")
        if year is None:
            medline_date_el = article_el.find(".//Journal/JournalIssue/PubDate/MedlineDate")
            if medline_date_el is not None and medline_date_el.text:
                match = re.match(r"(\d{4})", medline_date_el.text)
                if match:
                    year = int(match.group(1))

        journal_el = article_el.find(".//Journal/Title")
        journal = _get_text(journal_el)

        pmid_el = medline.find(".//PMID")
        pmid = _get_text(pmid_el)

        doi = None
        for id_el in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                doi=doi,
                url=url,
                source="pubmed",
                journal=journal,
                pmid=pmid,
            )
        )
    return papers


def _get_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()
