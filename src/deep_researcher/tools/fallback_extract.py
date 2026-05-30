"""Fallback extraction tool.

Single-pass extraction table when the theme-by-theme path fails.
This is a recovery tool (layered error recovery).
"""
from __future__ import annotations

import logging

from deep_researcher.constants import FALLBACK_MAX_PAPERS, FALLBACK_TOKEN_BUDGET
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper, ToolResult
from deep_researcher.parsing import build_tiered_corpus
from deep_researcher.prompts import CATEGORY_EXTRACTION_PROMPT
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")


class FallbackExtractionTool(Tool):
    name = "fallback_extraction"
    description = "Single-pass extraction table fallback when theme-by-theme extraction fails"
    is_read_only = True
    category = "utility"
    quality_tier = 1
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def execute(
        self,
        papers: list[Paper] | None = None,
        query: str = "",
        **kwargs,
    ) -> ToolResult:
        if not papers or not self._llm:
            return ToolResult(text="Extraction failed: no papers or LLM")

        top_papers = papers[:FALLBACK_MAX_PAPERS]
        corpus = build_tiered_corpus(
            list(enumerate(top_papers)),
            token_budget=FALLBACK_TOKEN_BUDGET,
        )
        prompt = CATEGORY_EXTRACTION_PROMPT.format(
            query=query,
            category="All papers",
            count=len(top_papers),
            corpus=corpus,
        )
        try:
            content = self._llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract the table."},
            ])
            return ToolResult(text=content)
        except Exception as e:
            return ToolResult(text=f"Extraction failed: {e}")
