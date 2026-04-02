import json

from deep_researcher.agent import (
    _build_tiered_corpus,
    _compact_messages,
    _expand_for_matching,
    _is_relevant,
    _parse_categories,
    _parse_merged_categories,
)
from deep_researcher.models import Paper
from deep_researcher.tools.base import Tool, ToolRegistry, ToolResult


class TestIsRelevant:
    def test_matches_method_and_domain(self):
        p = Paper(title="Transformer for structural health monitoring", abstract="attention mechanism for SHM")
        assert _is_relevant(p, "query", ["transformer", "attention"], ["structural health monitoring"])

    def test_rejects_method_only(self):
        p = Paper(title="Transformer architecture improvements", abstract="self-attention mechanism")
        assert not _is_relevant(p, "query", ["transformer"], ["structural health monitoring"])

    def test_rejects_domain_only(self):
        p = Paper(title="Structural health monitoring with sensors", abstract="SHM damage detection")
        assert not _is_relevant(p, "query", ["transformer"], ["structural health monitoring"])

    def test_empty_text_passes(self):
        p = Paper(title="", abstract=None)
        assert _is_relevant(p, "query", ["transformer"], ["shm"])

    def test_no_terms_uses_fallback(self):
        p = Paper(title="Deep learning for crack detection", abstract="")
        assert _is_relevant(p, "deep learning crack detection", None, None)


class TestExpandForMatching:
    def test_single_word_kept(self):
        result = _expand_for_matching(["transformer"])
        assert "transformer" in result

    def test_long_phrase_generates_trigrams(self):
        result = _expand_for_matching(["structural health monitoring damage detection"])
        assert "structural health monitoring" in result
        assert "health monitoring damage" in result
        assert "monitoring damage detection" in result

    def test_no_bigrams_generated(self):
        """Bigrams are too generic — only trigrams from 4+ word phrases."""
        result = _expand_for_matching(["structural health monitoring damage detection"])
        assert "health monitoring" not in result
        assert "damage detection" not in result

    def test_trigram_kept_as_is(self):
        result = _expand_for_matching(["code compliance checking"])
        assert "code compliance checking" in result

    def test_two_word_term_no_split(self):
        result = _expand_for_matching(["bridge condition"])
        assert result == ["bridge condition"]


class TestParseCategories:
    def test_basic_parsing(self):
        text = "CATEGORY: Vision Methods\nPAPERS: 1, 2, 3\n\nCATEGORY: NLP Methods\nPAPERS: 4, 5"
        result = _parse_categories(text, 5)
        assert "Vision Methods" in result
        assert "NLP Methods" in result
        assert result["Vision Methods"] == [0, 1, 2]

    def test_strips_markdown_bold(self):
        text = "**CATEGORY:** Vision Methods\n**PAPERS:** 1, 2, 3"
        result = _parse_categories(text, 3)
        assert "Vision Methods" in result

    def test_strips_list_markers(self):
        text = "- CATEGORY: Vision Methods\n- PAPERS: 1, 2"
        result = _parse_categories(text, 2)
        assert "Vision Methods" in result

    def test_case_insensitive(self):
        text = "category: Test\npapers: 1, 2"
        result = _parse_categories(text, 2)
        assert "Test" in result

    def test_ignores_out_of_range(self):
        text = "CATEGORY: Test\nPAPERS: 1, 2, 99"
        result = _parse_categories(text, 5)
        assert 98 not in result["Test"]

    def test_empty_input(self):
        assert _parse_categories("", 5) == {}


class TestParseMergedCategories:
    def test_basic_merge(self):
        original = {
            "Crack Detection A": [0, 1],
            "Crack Detection B": [2, 3],
            "SHM Sensors": [4, 5],
        }
        text = (
            "FINAL: Crack Detection\n"
            "MERGE: Crack Detection A, Crack Detection B\n\n"
            "FINAL: Sensor-Based SHM\n"
            "MERGE: SHM Sensors\n"
        )
        result = _parse_merged_categories(text, original)
        assert result is not None
        assert "Crack Detection" in result
        assert sorted(result["Crack Detection"]) == [0, 1, 2, 3]
        assert result["Sensor-Based SHM"] == [4, 5]

    def test_returns_none_on_empty(self):
        assert _parse_merged_categories("", {"A": [0]}) is None

    def test_returns_none_if_too_many_papers_lost(self):
        original = {"A": [0], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        # Only merges A (1 paper), losing 9 of 10 (>50% lost)
        text = "FINAL: Group\nMERGE: A\n"
        result = _parse_merged_categories(text, original)
        assert result is None

    def test_fuzzy_match(self):
        original = {"Vision-Based Crack Detection": [0, 1]}
        text = "FINAL: Vision\nMERGE: Vision-Based Crack Detection\n"
        result = _parse_merged_categories(text, original)
        assert result is not None
        assert result["Vision"] == [0, 1]


class TestBuildTieredCorpus:
    def test_empty_papers(self):
        result = _build_tiered_corpus([], token_budget=15000)
        assert result == ""

    def test_includes_abstracts_for_top_papers(self):
        papers = [Paper(title=f"Paper {i}", abstract="Abstract text here", citation_count=100-i) for i in range(5)]
        result = _build_tiered_corpus(papers, token_budget=15000)
        assert "Abstract" in result

    def test_respects_budget(self):
        papers = [Paper(title=f"Paper {i}", abstract="x" * 500, citation_count=100-i) for i in range(100)]
        result = _build_tiered_corpus(papers, token_budget=500)
        assert "additional papers" in result


class TestCompactMessages:
    def test_no_compaction_under_budget(self):
        messages = [{"role": "user", "content": "short"}]
        result = _compact_messages(messages, lambda m: 100)
        assert result == messages

    def test_compacts_old_tool_results(self):
        messages = [
            {"role": "user", "content": "query"},
            {"role": "tool", "content": "Found 5 papers on Scopus:\n" + "x" * 500},
            {"role": "tool", "content": "Found 3 papers on IEEE:\n" + "y" * 500},
        ]
        result = _compact_messages(messages, lambda m: 100_000)
        assert "[Results compressed" in result[1]["content"]


# --- ToolRegistry input validation tests ---


class _DummyTool(Tool):
    """Minimal tool for testing ToolRegistry validation."""
    name = "test_tool"
    description = "A test tool"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }

    def execute(self, query: str = "", max_results: int = 10) -> ToolResult:
        return ToolResult(text=f"OK: query={query}, max_results={max_results}")


class TestToolRegistryValidation:
    def _make_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(_DummyTool())
        return reg

    def test_unknown_tool(self):
        reg = self._make_registry()
        result = reg.execute("nonexistent", "{}")
        assert "Unknown tool" in result.text

    def test_invalid_json(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", "not json")
        assert "Invalid JSON" in result.text

    def test_missing_required_param(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", json.dumps({"max_results": 5}))
        assert "Missing required" in result.text
        assert "query" in result.text

    def test_valid_call(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", json.dumps({"query": "test"}))
        assert "OK" in result.text

    def test_max_results_clamped_high(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", json.dumps({"query": "test", "max_results": 999}))
        assert "max_results=100" in result.text

    def test_max_results_clamped_low(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", json.dumps({"query": "test", "max_results": 0}))
        assert "max_results=10" in result.text

    def test_empty_arguments(self):
        reg = self._make_registry()
        result = reg.execute("test_tool", "")
        assert "Missing required" in result.text
