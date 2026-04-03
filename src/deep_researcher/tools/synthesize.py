# src/deep_researcher/tools/synthesize.py
"""Per-category synthesis tool.

Takes papers in one category and produces an LLM-written literature review section.
"""
from __future__ import annotations

import logging

from deep_researcher.constants import CATEGORY_TOKEN_BUDGET
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper, ToolResult
from deep_researcher.parsing import build_tiered_corpus
from deep_researcher.prompts import CATEGORY_SYNTHESIS_PROMPT
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")


class SynthesisTool(Tool):
    name = "synthesize_category"
    description = "Write a literature review section for one category of papers"
    is_read_only = True
    is_concurrency_safe = True
    category = "utility"
    quality_tier = 1
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def execute(
        self,
        indexed_papers: list[tuple[int, Paper]] | None = None,
        query: str = "",
        category_name: str = "",
        token_budget: int = CATEGORY_TOKEN_BUDGET,
        **kwargs,
    ) -> ToolResult:
        if not indexed_papers or not self._llm:
            return ToolResult(text="No papers to synthesize")

        corpus = build_tiered_corpus(indexed_papers, token_budget=token_budget)
        prompt = CATEGORY_SYNTHESIS_PROMPT.format(
            query=query,
            category=category_name,
            count=len(indexed_papers),
            corpus=corpus,
        )

        try:
            content = self._llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Write the synthesis for: {category_name}"},
            ])
            return ToolResult(text=content)
        except Exception as e:
            logger.warning("Synthesis for '%s' failed: %s", category_name, e)
            return ToolResult(text=f"Synthesis failed for {category_name}: {e}")
