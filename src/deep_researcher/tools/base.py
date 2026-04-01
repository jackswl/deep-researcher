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
            return tool.execute(**kwargs)
        except json.JSONDecodeError:
            return ToolResult(text=f"Error: Invalid JSON arguments for tool '{name}'")
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return ToolResult(text=f"Error executing {name}: {e}")

    def execute_partitioned(self, tool_calls: list[dict]) -> list[tuple[str, ToolResult]]:
        """Execute tool calls with Claude Code's partitioning pattern.

        Consecutive read-only tools run concurrently in batches.
        Non-read-only tools run serially, blocking the queue.
        This prevents concurrent writes while maximizing read throughput.
        """
        # Partition into batches (Claude Code partitionToolCalls pattern)
        batches: list[tuple[bool, list[dict]]] = []
        for tc in tool_calls:
            tool = self._tools.get(tc["name"])
            is_safe = tool.is_read_only if tool else False

            if batches and is_safe and batches[-1][0]:
                # Extend current concurrent batch
                batches[-1][1].append(tc)
            else:
                batches.append((is_safe, [tc]))

        # Execute batches
        results: list[tuple[str, ToolResult]] = []
        for is_concurrent, batch in batches:
            if is_concurrent and len(batch) > 1:
                # Run read-only tools concurrently
                batch_results = self._run_concurrent(batch)
                results.extend(batch_results)
            else:
                # Run serially
                for tc in batch:
                    result = self.execute(tc["name"], tc["arguments"])
                    results.append((tc["id"], result))

        return results

    def _run_concurrent(self, tool_calls: list[dict]) -> list[tuple[str, ToolResult]]:
        """Run multiple tool calls concurrently via ThreadPoolExecutor."""
        results: list[tuple[str, ToolResult]] = [("", ToolResult(text="")) for _ in tool_calls]

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
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
