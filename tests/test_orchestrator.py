"""Tests for the research pipeline orchestrator."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

from deep_researcher.config import Config
from deep_researcher.models import Paper, PipelineState, ToolResult


class TestOrchestrator:
    def _make_orchestrator(self):
        from deep_researcher.orchestrator import Orchestrator
        config = Config(
            model="test-model",
            base_url="http://localhost:11434/v1",
            api_key="test",
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = config
        orch.console = MagicMock()
        orch._cancel = threading.Event()
        orch._output_folder = ""

        # Mock tools
        orch._search_tool = MagicMock()
        orch._enrichment_tool = MagicMock()
        orch._categorize_tool = MagicMock()
        orch._synthesize_tool = MagicMock()
        orch._cross_analysis_tool = MagicMock()
        orch.llm = MagicMock()
        return orch

    def test_search_phase_returns_papers(self):
        orch = self._make_orchestrator()
        papers = [Paper(title="Paper A"), Paper(title="Paper B")]
        orch._search_tool.execute.return_value = ToolResult(text="Found 2", papers=papers)
        state = PipelineState(query="test")
        new_state = orch._run_search(state)
        assert len(new_state.papers) == 2

    def test_enrich_phase_returns_enriched_papers(self):
        orch = self._make_orchestrator()
        papers = {"k1": Paper(title="Paper A"), "k2": Paper(title="Paper B")}
        enriched = [Paper(title="Paper A", doi="10.1/a"), Paper(title="Paper B", doi="10.1/b")]
        orch._enrichment_tool.execute.return_value = ToolResult(text="Enriched", papers=enriched)
        state = PipelineState(query="test", papers=papers)
        new_state = orch._run_enrichment(state)
        assert all(p.doi for p in new_state.papers.values())

    def test_search_failure_returns_empty_state(self):
        orch = self._make_orchestrator()
        orch._search_tool.execute.return_value = ToolResult(text="Found 0", papers=[])
        state = PipelineState(query="test")
        new_state = orch._run_search(state)
        assert len(new_state.papers) == 0

    def test_synthesis_fallback_on_categorization_failure(self):
        orch = self._make_orchestrator()
        orch._categorize_tool.execute.return_value = ToolResult(text="Failed", data=None)
        orch.llm.chat_no_think.return_value = "Fallback synthesis content"
        papers = [Paper(title=f"P{i}", citation_count=10 - i) for i in range(5)]
        state = PipelineState(
            query="test",
            papers={p.unique_key: p for p in papers},
            synthesis_papers=papers,
        )
        report = orch._run_synthesis(state)
        assert report.report  # should have fallback content

    def test_state_immutability(self):
        orch = self._make_orchestrator()
        papers = [Paper(title="Paper A")]
        orch._search_tool.execute.return_value = ToolResult(text="Found 1", papers=papers)
        state = PipelineState(query="test")
        original_papers = state.papers
        new_state = orch._run_search(state)
        assert state.papers is original_papers  # original unchanged
        assert len(new_state.papers) == 1

    def test_assemble_report_format(self):
        orch = self._make_orchestrator()
        papers = [
            Paper(title="Paper A", authors=["Alice"], year=2023, doi="10.1/a"),
            Paper(title="Paper B", authors=["Bob", "Carol"], year=2024),
        ]
        state = PipelineState(
            query="test query",
            papers={p.unique_key: p for p in papers},
            synthesis_papers=papers,
            categories={"Group A": [0, 1]},
            category_sections=[("Group A", "Section content here")],
            cross_section="Cross patterns here",
        )
        report = orch._assemble_report(state)
        assert "### test query" in report
        assert "#### Coverage" in report
        assert "##### Group A" in report
        assert "Section content here" in report
        assert "Cross patterns here" in report
        assert "#### References" in report
        assert "[1] Alice (2023)" in report
        assert "[2] Bob et al. (2024)" in report
