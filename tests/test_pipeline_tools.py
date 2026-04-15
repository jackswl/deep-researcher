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

    def test_filters_papers_outside_year_range(self):
        tool = self._make_tool()
        tool.set_year_range(2023, 2025)
        mock_results = [
            {"bib": {"title": "Old Paper", "author": ["A"], "pub_year": "2018"}, "num_citations": 10},
            {"bib": {"title": "Recent Paper", "author": ["B"], "pub_year": "2024"}, "num_citations": 5},
            {"bib": {"title": "Future Paper", "author": ["C"], "pub_year": "2027"}, "num_citations": 1},
            {"bib": {"title": "No Year Paper", "author": ["D"]}, "num_citations": 3},
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test", max_results=10)
        titles = [p.title for p in result.papers]
        assert "Old Paper" not in titles
        assert "Future Paper" not in titles
        assert "Recent Paper" in titles
        assert "No Year Paper" in titles  # unknown year kept

    def test_no_year_range_returns_all(self):
        tool = self._make_tool()
        # no set_year_range call
        mock_results = [
            {"bib": {"title": "Paper 2010", "author": ["A"], "pub_year": "2010"}, "num_citations": 1},
            {"bib": {"title": "Paper 2024", "author": ["B"], "pub_year": "2024"}, "num_citations": 1},
        ]
        with patch("deep_researcher.tools.scholar_search.scholarly") as mock_scholarly:
            mock_scholarly.search_pubs.return_value = iter(mock_results)
            result = tool.execute(query="test", max_results=10)
        assert len(result.papers) == 2

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


class TestExaSearchTool:
    def _make_tool(self, api_key: str = "test-key"):
        from deep_researcher.tools.exa_search import ExaSearchTool
        return ExaSearchTool(api_key=api_key)

    def _make_response(self, results: list[dict]):
        """Build a fake Exa SDK response. Our _safe_attr helper accepts dicts,
        so we don't need to construct real pydantic result objects."""
        return MagicMock(results=results)

    def test_disabled_without_api_key(self):
        tool = self._make_tool(api_key="")
        result = tool.execute(query="test query")
        assert "not available" in result.text.lower()
        assert "exa_api_key" in result.text.lower()

    def test_parses_response_into_papers(self):
        tool = self._make_tool()
        results = [
            {
                "title": "Paper A",
                "url": "https://doi.org/10.1234/paper-a",
                "author": "Alice Smith, Bob Jones",
                "published_date": "2023-05-01T00:00:00.000Z",
                "text": "Full text of paper A.",
                "highlights": ["Key finding one", "Key finding two"],
                "summary": "A concise summary of paper A.",
            },
            {
                "title": "Paper B",
                "url": "https://arxiv.org/abs/2401.12345v2",
                "author": "Carol Lee",
                "published_date": "2024-01-15T00:00:00.000Z",
                "text": None,
                "highlights": ["Highlight only"],
                "summary": None,
            },
        ]
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response(results)
        with patch("exa_py.Exa", return_value=mock_exa_instance) as mock_exa_cls:
            result = tool.execute(query="test query", max_results=10)

        # Verify the tracking header was set (integration attribution)
        assert mock_exa_instance.headers.get("x-exa-integration") == "deep-researcher"
        mock_exa_cls.assert_called_once_with(api_key="test-key")

        assert isinstance(result, ToolResult)
        assert len(result.papers) == 2
        assert result.papers[0].title == "Paper A"
        assert result.papers[0].authors == ["Alice Smith", "Bob Jones"]
        assert result.papers[0].year == 2023
        assert result.papers[0].doi == "10.1234/paper-a"
        assert result.papers[0].source == "exa"
        # Summary should win over highlights and text (priority order)
        assert result.papers[0].abstract == "A concise summary of paper A."

        # Paper B: no summary, should fall back to highlights, and extract arxiv_id
        assert result.papers[1].abstract == "Highlight only"
        assert result.papers[1].arxiv_id == "2401.12345"

    def test_content_fallback_to_text_when_summary_and_highlights_missing(self):
        tool = self._make_tool()
        results = [{
            "title": "Text Only Paper",
            "url": "https://example.com/paper",
            "author": None,
            "published_date": None,
            "text": "Only the text field is populated.",
            "highlights": None,
            "summary": None,
        }]
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response(results)
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            result = tool.execute(query="q")
        assert result.papers[0].abstract == "Only the text field is populated."

    def test_handles_no_content_fields(self):
        tool = self._make_tool()
        results = [{
            "title": "Bare Paper",
            "url": "https://example.com/bare",
            "author": None,
            "published_date": None,
            "text": None,
            "highlights": None,
            "summary": None,
        }]
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response(results)
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            result = tool.execute(query="q")
        assert result.papers[0].abstract is None
        assert result.papers[0].title == "Bare Paper"

    def test_empty_results(self):
        tool = self._make_tool()
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response([])
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            result = tool.execute(query="q")
        assert len(result.papers) == 0
        assert "no results" in result.text.lower()

    def test_handles_api_exception(self):
        tool = self._make_tool()
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.side_effect = Exception("Network error")
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            result = tool.execute(query="q")
        assert "Error searching Exa" in result.text
        assert len(result.papers) == 0

    def test_respects_year_range_filter_in_request(self):
        tool = self._make_tool()
        tool.set_year_range(2020, 2024)
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response([])
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            tool.execute(query="q")
        _, kwargs = mock_exa_instance.search_and_contents.call_args
        assert kwargs["start_published_date"].startswith("2020-01-01")
        assert kwargs["end_published_date"].startswith("2024-12-31")

    def test_requests_both_text_and_highlights(self):
        """Exa's contents field is not mutually exclusive — we want both."""
        tool = self._make_tool()
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response([])
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            tool.execute(query="q")
        _, kwargs = mock_exa_instance.search_and_contents.call_args
        assert "text" in kwargs
        assert "highlights" in kwargs
        assert kwargs["category"] == "research paper"  # sensible academic default

    def test_invalid_search_type_falls_back_to_auto(self):
        tool = self._make_tool()
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = self._make_response([])
        with patch("exa_py.Exa", return_value=mock_exa_instance):
            tool.execute(query="q", search_type="invalid")
        _, kwargs = mock_exa_instance.search_and_contents.call_args
        assert kwargs["type"] == "auto"

    def test_is_read_only(self):
        tool = self._make_tool()
        assert tool.is_read_only is True
