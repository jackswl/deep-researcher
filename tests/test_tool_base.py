import json

from deep_researcher.tools.base import Tool, ToolRegistry, ToolResult


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


class _FailingTool(Tool):
    name = "failing_tool"
    description = "Always fails"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Tool execution failed")


class TestToolValidateInput:
    def test_passes_with_required_params(self):
        tool = _DummyTool()
        result = tool.validate_input(query="test")
        assert result["query"] == "test"

    def test_raises_on_missing_required(self):
        tool = _DummyTool()
        try:
            tool.validate_input(max_results=5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "query" in str(e)

    def test_clamps_max_results(self):
        tool = _DummyTool()
        result = tool.validate_input(query="test", max_results=999)
        assert result["max_results"] == 100


class TestToolSafeExecute:
    def test_returns_result_on_success(self):
        tool = _DummyTool()
        result = tool.safe_execute(query="test")
        assert "OK" in result.text

    def test_wraps_validation_error(self):
        tool = _DummyTool()
        result = tool.safe_execute(max_results=5)  # missing required 'query'
        assert result.text.startswith("Error:")

    def test_wraps_execution_error(self):
        tool = _FailingTool()
        result = tool.safe_execute()
        assert "Tool execution failed" in result.text

    def test_safe_execute_never_raises(self):
        tool = _FailingTool()
        result = tool.safe_execute()
        assert isinstance(result, ToolResult)
