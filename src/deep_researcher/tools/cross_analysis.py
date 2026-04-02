# src/deep_researcher/tools/cross_analysis.py
"""Cross-category analysis tool.

Analyzes patterns across all category sections.
"""
from __future__ import annotations

import logging

from deep_researcher.llm import LLMClient
from deep_researcher.models import ToolResult
from deep_researcher.prompts import CROSS_CATEGORY_PROMPT
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")


class CrossAnalysisTool(Tool):
    name = "cross_category_analysis"
    description = "Analyze patterns across paper categories"
    is_read_only = True
    category = "utility"
    quality_tier = 1
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def execute(
        self,
        sections: list[tuple[str, str]] | None = None,
        query: str = "",
        **kwargs,
    ) -> ToolResult:
        if not sections or not self._llm:
            return ToolResult(text="Cross-category analysis unavailable: no sections")

        summaries = []
        for name, content in sections:
            summary = content[:500]
            if len(content) > 500:
                cut = summary.rfind(". ")
                summary = summary[:cut + 1] if cut > 300 else summary + "..."
            summaries.append(f"**{name}:**\n{summary}")

        prompt = CROSS_CATEGORY_PROMPT.format(
            query=query,
            category_summaries="\n\n".join(summaries),
        )

        try:
            content = self._llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the cross-category analysis."},
            ])
            return ToolResult(text=content)
        except Exception as e:
            return ToolResult(text=f"Cross-category analysis unavailable: {e}")
