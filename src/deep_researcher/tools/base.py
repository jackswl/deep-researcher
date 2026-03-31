from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from deep_researcher.models import ToolResult

logger = logging.getLogger("deep_researcher")


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    is_read_only: bool = True

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
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
            return tool.execute(**kwargs)
        except json.JSONDecodeError:
            return ToolResult(text=f"Error: Invalid JSON arguments for tool '{name}'")
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return ToolResult(text=f"Error executing {name}: {e}")

    def execute_concurrent(self, tool_calls: list[dict]) -> list[tuple[str, ToolResult]]:
        """Execute multiple read-only tool calls concurrently (Claude Code pattern)."""
        results: list[tuple[str, ToolResult]] = [("", ToolResult(text="")) for _ in tool_calls]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for i, tc in enumerate(tool_calls):
                future = executor.submit(self.execute, tc["name"], tc["arguments"])
                futures[future] = (i, tc["id"])

            for future in as_completed(futures):
                idx, call_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ToolResult(text=f"Error: {e}")
                results[idx] = (call_id, result)

        return results
