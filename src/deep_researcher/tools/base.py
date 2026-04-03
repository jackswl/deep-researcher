from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

from deep_researcher.models import ToolResult

logger = logging.getLogger("deep_researcher")

# Progress callback: (message, current, total)
# Mirrors Claude Code's onProgress pattern in Tool.call()
ProgressCallback = Callable[[str, int, int], None]


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    is_read_only: bool = True
    # Whether this tool can safely run concurrently with other tools
    # (Claude Code Tool.isConcurrencySafe pattern)
    is_concurrency_safe: bool = False
    # Taxonomy: helps the agent reason about which tools to use when
    # Values: "preprint", "index", "open_access", "publisher", "citation", "utility"
    category: str = "index"
    # Quality tier: 1=peer-reviewed/curated, 2=broad coverage, 3=preprints/open
    quality_tier: int = 2
    # Year range filter (set at construction, applied automatically)
    _start_year: int | None = None
    _end_year: int | None = None

    def set_year_range(self, start_year: int | None, end_year: int | None) -> None:
        self._start_year = start_year
        self._end_year = end_year

    def _filter_by_year(self, papers: list) -> list:
        """Post-filter papers by year range (fallback when API doesn't support date filters)."""
        if self._start_year is None and self._end_year is None:
            return papers
        filtered = []
        for p in papers:
            if p.year is None:
                filtered.append(p)  # Keep papers with unknown year
                continue
            if self._start_year is not None and p.year < self._start_year:
                continue
            if self._end_year is not None and p.year > self._end_year:
                continue
            filtered.append(p)
        return filtered

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def validate_input(self, **kwargs: Any) -> dict[str, Any]:
        """Validate input at tool boundary (claude-code Principle 5).
        Checks required parameters and clamps max_results.
        Returns validated kwargs dict.
        """
        required = self.parameters.get("required", [])
        missing = [r for r in required if r not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        if "max_results" in kwargs:
            val = kwargs["max_results"]
            if not isinstance(val, int) or val < 1:
                kwargs["max_results"] = 10
            elif val > 100:
                kwargs["max_results"] = 100
        return kwargs

    def safe_execute(
        self,
        on_progress: ProgressCallback | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute with validation and error wrapping (claude-code Principle 4).
        Never raises -- wraps all errors into ToolResult.

        on_progress: optional callback for real-time progress reporting
        (Claude Code onProgress pattern).
        """
        try:
            validated = self.validate_input(**kwargs)
            # Only inject on_progress when the tool's execute() can accept it
            # (has an explicit on_progress param or **kwargs). Prevents TypeError
            # on tools with fixed-parameter signatures.
            if on_progress is not None:
                sig = inspect.signature(self.execute)
                params = sig.parameters
                accepts_progress = (
                    "on_progress" in params
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                )
                if accepts_progress:
                    validated["on_progress"] = on_progress
            return self.execute(**validated)
        except Exception as e:
            logger.debug("Tool %s failed: %s", self.name, e, exc_info=True)
            return ToolResult(text=f"Error: {e}")

    def to_openai_schema(self) -> dict[str, Any]:
        desc = self.description
        if self.category:
            desc = f"[{self.category}] {desc}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: str) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(text=f"Error: Unknown tool '{name}'")
        try:
            kwargs = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return ToolResult(text=f"Error: Invalid JSON arguments for tool '{name}'")
        return tool.safe_execute(**kwargs)

