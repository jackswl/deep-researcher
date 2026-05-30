"""Prompt templates for the ResearchAgent LLM calls."""

CATEGORIZE_PROMPT = """\
You are a research librarian. Below are {count} papers on: "{query}"

Assign each paper to exactly one category (3-6 categories). \
Categorize by approach/theme, NOT by database or year.

## Papers
{paper_list}

## Output Format
Return ONLY a list in this exact format (one line per category, paper numbers comma-separated):

CATEGORY: Category Name
PAPERS: 1, 5, 12, 23

CATEGORY: Another Category
PAPERS: 2, 7, 8, 19

Rules:
- Every paper number must appear in exactly one category
- 3-6 categories total
- Category names should be specific (e.g., "Vision-Based Damage Detection", not "Methods")
- No explanation needed — just the categories and paper numbers
"""

MERGE_CATEGORIES_PROMPT = """\
/no_think
You are a research librarian. Papers on "{query}" were categorized in batches, \
producing {count} overlapping categories. Merge them into {target} final categories \
by grouping semantically similar ones together.

## Current categories (name -> paper count)
{category_list}

## Output Format
Return ONLY a mapping in this exact format (one line per final category):

FINAL: Final Category Name
MERGE: Old Category A, Old Category B, Old Category C

FINAL: Another Final Category
MERGE: Old Category D, Old Category E

Rules:
- Exactly {target} final categories
- Every old category must appear in exactly one MERGE line
- Use the old category names exactly as listed above
- Final category names should be descriptive (not generic like "Other")
"""

CATEGORY_EXTRACTION_PROMPT = """\
You are a research librarian compiling a literature screening matrix on: "{query}"

This theme covers: **{category}** ({count} papers)

## Papers in this theme
{corpus}

## Task
Produce ONE markdown table that extracts the key details of each paper. \
Use exactly these columns and header:

| Ref | Paper | Year | Method | Key finding (as stated) | Cites |
|-----|-------|------|--------|-------------------------|-------|

Use the [number] shown beside each paper above as its Ref. Put one row per paper.

## Rules
- Extract ONLY what each abstract explicitly states. Do not infer, generalize, or invent.
- Method: the approach or technique the abstract describes. If the abstract does not \
state it, write "not stated".
- Key finding (as stated): the specific result the abstract reports. If none is stated, \
write "not stated".
- Year and Cites come from the metadata shown for each paper. Leave blank only if absent.
- Keep every cell short. Shorten long titles. Do not use line breaks inside a cell.
- Include EVERY paper listed above, in the order given, one row each.
- Output ONLY the table. No prose, no headings, no commentary before or after it.
"""

CLARIFY_PROMPT = """\
You are a research assistant helping to refine a research question before searching academic databases.

Given the user's research topic, generate exactly 3 short, focused clarifying questions that would \
help narrow the search and produce better results. Focus on:
- Specific subfield or application domain
- Time period or recency preferences
- Methodological focus (theoretical, empirical, computational, etc.)

Format: Return ONLY the 3 questions, one per line, numbered 1-3. No preamble.
"""
