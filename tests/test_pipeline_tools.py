"""Tests for pipeline tools (search, enrichment, categorize, synthesize, cross-analysis)."""
from __future__ import annotations

import threading
from unittest.mock import patch

from deep_researcher.models import Paper, ToolResult


class TestScholarSearchTool:
    def _make_tool(self):
        from deep_researcher.tools.scholar_search import ScholarSearchTool
        return ScholarSearchTool()

    def test_returns_tool_result_with_papers(self):
        tool = self._make_tool()
        mock_results = [
            {"bib": {"title": "Paper A", "author": ["Alice"], "pub_year": "2023", "abstract": "Abstract A", "venue": "Nature"}, "num_citations": 50, "pub_url": "http://a.com"},
            {"bib": {"title": "Paper B", "author": "Bob and Carol", "pub_year": "2024", "abstract": "Abstract B", "venue": "Science"}, "num_citations": 30, "pub_url": "http://b.com"},
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test query", max_results=10)
        assert isinstance(result, ToolResult)
        assert len(result.papers) == 2
        assert result.papers[0].title == "Paper A"
        assert result.papers[1].authors == ["Bob", "Carol"]

    def test_deduplicates_by_title(self):
        tool = self._make_tool()
        mock_results = [
            {"bib": {"title": "Same Paper", "author": ["A"], "pub_year": "2023"}, "num_citations": 10},
            {"bib": {"title": "Same Paper", "author": ["B"], "pub_year": "2023"}, "num_citations": 5},
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test", max_results=10)
        assert len(result.papers) == 1

    def test_handles_search_failure(self):
        tool = self._make_tool()
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.side_effect = Exception("Network error")
            result = tool.execute(query="test", max_results=10)
        assert len(result.papers) == 0
        assert "0 papers" in result.text

    def test_respects_max_results(self):
        tool = self._make_tool()
        mock_results = [
            {"bib": {"title": f"Paper {i}", "author": [f"Author {i}"], "pub_year": "2023"}, "num_citations": i}
            for i in range(20)
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test", max_results=5)
        assert len(result.papers) == 5

    def test_respects_cancel_event(self):
        tool = self._make_tool()
        cancel = threading.Event()
        cancel.set()  # pre-cancelled
        mock_results = [
            {"bib": {"title": f"Paper {i}", "author": [f"A{i}"], "pub_year": "2023"}, "num_citations": i}
            for i in range(10)
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test", max_results=10, cancel=cancel)
        assert len(result.papers) == 0

    def test_is_read_only(self):
        tool = self._make_tool()
        assert tool.is_read_only is True


from unittest.mock import MagicMock, patch  # noqa: F811 (re-import for clarity)


class TestEnrichmentTool:
    def _make_tool(self):
        from deep_researcher.tools.enrichment import EnrichmentTool
        return EnrichmentTool()

    def _mock_openalex_response(self, title="Paper A", doi="10.1234/test", cited_by=50):
        inv_idx = {"This": [0], "is": [1], "a": [2], "test": [3], "abstract": [4]}
        return {
            "results": [{
                "title": title,
                "doi": f"https://doi.org/{doi}",
                "abstract_inverted_index": inv_idx,
                "primary_location": {"source": {"display_name": "Nature"}},
                "open_access": {"oa_url": "http://oa.example.com"},
                "cited_by_count": cited_by,
            }]
        }

    def test_enriches_papers_from_openalex(self):
        tool = self._make_tool()
        papers = [Paper(title="Paper A", authors=["Alice"], year=2023)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._mock_openalex_response()
        with patch("deep_researcher.tools.enrichment.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = tool.execute(papers=papers, email="test@example.com")
        assert len(result.papers) == 1
        assert result.papers[0].doi == "10.1234/test"
        assert result.papers[0].journal == "Nature"

    def test_returns_original_on_failure(self):
        tool = self._make_tool()
        papers = [Paper(title="Paper A", authors=["Alice"], year=2023)]
        with patch("deep_researcher.tools.enrichment.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("Network error")
            result = tool.execute(papers=papers, email="test@example.com")
        assert len(result.papers) == 1
        assert result.papers[0].title == "Paper A"
        assert result.papers[0].doi is None

    def test_does_not_mutate_input_papers(self):
        tool = self._make_tool()
        original = Paper(title="Paper A", authors=["Alice"], year=2023)
        papers = [original]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._mock_openalex_response()
        with patch("deep_researcher.tools.enrichment.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = tool.execute(papers=papers, email="test@example.com")
        assert original.doi is None  # original not mutated
        assert result.papers[0].doi == "10.1234/test"

    def test_is_read_only(self):
        tool = self._make_tool()
        assert tool.is_read_only is True


class TestCategorizeTool:
    def _make_tool(self, llm_response="CATEGORY: Group A\nPAPERS: 1, 2\n\nCATEGORY: Group B\nPAPERS: 3"):
        from deep_researcher.tools.categorize import CategorizeTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.return_value = llm_response
        return CategorizeTool(llm=mock_llm)

    def test_returns_categories_in_data(self):
        tool = self._make_tool()
        papers = [Paper(title=f"Paper {i}", authors=[f"A{i}"], year=2023) for i in range(3)]
        result = tool.execute(papers=papers, query="test query")
        assert result.data is not None
        assert "Group A" in result.data
        assert "Group B" in result.data

    def test_handles_llm_failure(self):
        from deep_researcher.tools.categorize import CategorizeTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.side_effect = Exception("LLM error")
        tool = CategorizeTool(llm=mock_llm)
        papers = [Paper(title=f"Paper {i}") for i in range(3)]
        result = tool.execute(papers=papers, query="test query")
        assert result.data is None or result.data == {}

    def test_no_llm_returns_none(self):
        from deep_researcher.tools.categorize import CategorizeTool
        tool = CategorizeTool(llm=None)
        papers = [Paper(title="P")]
        result = tool.execute(papers=papers, query="test")
        assert result.data is None


class TestSynthesisTool:
    def _make_tool(self, llm_response="## Section\nSynthesis content here"):
        from deep_researcher.tools.synthesize import SynthesisTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.return_value = llm_response
        return SynthesisTool(llm=mock_llm)

    def test_returns_section_text(self):
        tool = self._make_tool()
        indexed = [(0, Paper(title="Paper A", abstract="Test abstract", citation_count=10))]
        result = tool.execute(indexed_papers=indexed, query="test query", category_name="Group A")
        assert "Synthesis content here" in result.text

    def test_handles_llm_failure(self):
        from deep_researcher.tools.synthesize import SynthesisTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.side_effect = Exception("LLM error")
        tool = SynthesisTool(llm=mock_llm)
        indexed = [(0, Paper(title="Paper A"))]
        result = tool.execute(indexed_papers=indexed, query="test", category_name="A")
        assert "failed" in result.text.lower()

    def test_is_read_only(self):
        tool = self._make_tool()
        assert tool.is_read_only is True


class TestCrossAnalysisTool:
    def _make_tool(self, llm_response="#### Cross-Category Patterns\nPatterns here"):
        from deep_researcher.tools.cross_analysis import CrossAnalysisTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.return_value = llm_response
        return CrossAnalysisTool(llm=mock_llm)

    def test_returns_analysis_text(self):
        tool = self._make_tool()
        sections = [("Group A", "Content A"), ("Group B", "Content B")]
        result = tool.execute(sections=sections, query="test query")
        assert "Patterns here" in result.text

    def test_handles_llm_failure(self):
        from deep_researcher.tools.cross_analysis import CrossAnalysisTool
        mock_llm = MagicMock()
        mock_llm.chat_no_think.side_effect = Exception("LLM error")
        tool = CrossAnalysisTool(llm=mock_llm)
        sections = [("A", "Content")]
        result = tool.execute(sections=sections, query="test")
        assert "unavailable" in result.text.lower()

    def test_no_sections_returns_unavailable(self):
        tool = self._make_tool()
        result = tool.execute(sections=[], query="test")
        assert "unavailable" in result.text.lower()
