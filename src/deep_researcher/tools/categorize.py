# src/deep_researcher/tools/categorize.py
"""Paper categorization tool.

Uses LLM to assign papers to thematic categories.
Handles batching and category merging.
"""
from __future__ import annotations

import logging

from deep_researcher.constants import CATEGORIZE_BATCH_SIZE, MAX_FINAL_CATEGORIES
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper, ToolResult
from deep_researcher.parsing import parse_categories, parse_merged_categories
from deep_researcher.prompts import CATEGORIZE_PROMPT, MERGE_CATEGORIES_PROMPT
from deep_researcher.tools.base import Tool

logger = logging.getLogger("deep_researcher")


class CategorizeTool(Tool):
    name = "categorize_papers"
    description = "Categorize papers by theme using LLM"
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
            return ToolResult(text="No papers to categorize", data=None)

        all_categories: dict[str, list[int]] = {}

        for batch_start in range(0, len(papers), CATEGORIZE_BATCH_SIZE):
            batch_end = min(batch_start + CATEGORIZE_BATCH_SIZE, len(papers))
            batch = papers[batch_start:batch_end]

            lines = []
            for i, p in enumerate(batch):
                global_idx = batch_start + i
                author = p.authors[0] if p.authors else "Unknown"
                if len(p.authors) > 1:
                    author += " et al."
                year = p.year or "n.d."
                cites = f", {p.citation_count} cites" if p.citation_count else ""
                abstract = f"\n   Abstract: {p.abstract}" if p.abstract else ""
                lines.append(f"{global_idx + 1}. {p.title} ({author}, {year}{cites}){abstract}")

            prompt = CATEGORIZE_PROMPT.format(
                count=len(batch),
                query=query,
                paper_list="\n".join(lines),
            )

            try:
                content = self._llm.chat_no_think([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Categorize these papers now."},
                ])
                batch_cats = parse_categories(content, len(papers))
                for cat_name, indices in batch_cats.items():
                    if cat_name in all_categories:
                        all_categories[cat_name].extend(indices)
                    else:
                        all_categories[cat_name] = indices
            except Exception as e:
                logger.warning("Categorization batch %d-%d failed: %s", batch_start, batch_end, e)
                continue

        if len(all_categories) > MAX_FINAL_CATEGORIES:
            all_categories = self._merge(query, all_categories)

        return ToolResult(
            text=f"Categorized into {len(all_categories)} categories",
            data=all_categories if all_categories else None,
        )

    def _merge(self, query: str, categories: dict[str, list[int]]) -> dict[str, list[int]]:
        """Merge semantically similar categories into MAX_FINAL_CATEGORIES groups."""
        cat_list = "\n".join(
            f"- {name} ({len(indices)} papers)" for name, indices in categories.items()
        )
        prompt = MERGE_CATEGORIES_PROMPT.format(
            query=query,
            count=len(categories),
            target=MAX_FINAL_CATEGORIES,
            category_list=cat_list,
        )

        try:
            content = self._llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Merge the categories now."},
            ])
            merged = parse_merged_categories(content, categories)
            if merged:
                return merged
        except Exception as e:
            logger.warning("Category merge failed: %s", e)

        sorted_cats = sorted(categories.items(), key=lambda x: -len(x[1]))
        return dict(sorted_cats[:MAX_FINAL_CATEGORIES])
