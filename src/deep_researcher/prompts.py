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

CATEGORY_SYNTHESIS_PROMPT = """\
You are a research analyst writing one section of a detailed literature review on: "{query}"

This section covers the category: **{category}** ({count} papers)

## Papers in this category
{corpus}

## Write this section. Reference papers by [number].

**What this group does:**
Write a paragraph (4-6 sentences) explaining the shared approach/theme.
Reference individual papers: e.g., "Smith et al. [1] introduced X. Jones et al. [2] extended this by Y."

**Key methods:**
Write a paragraph describing the specific methods and techniques.
For each method, cite which paper(s) used it.

**Main findings:**
Write a paragraph on collective findings. Include specific results ONLY if the \
abstract explicitly states them (e.g., accuracy percentages, performance metrics). \
Do NOT infer, generalize, or fabricate results that are not in the abstracts.

**Limitations & gaps (your analysis):**
Write YOUR OWN analysis of common weaknesses and gaps across this group. \
This is your synthesis — do NOT attribute these observations to specific papers \
with [number] citations. Instead write: "A common limitation across these studies is..." \
or "This group does not address..."

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
(Include EVERY paper listed above in the table)

## CRITICAL RULES
- ONLY state what the abstracts explicitly say. If a metric is not in the abstract, do NOT invent it.
- When citing [N], the claim MUST come from that paper's abstract above. Verify before writing.
- The Limitations section is YOUR analysis — do NOT fake-attribute observations to papers.
- Include ALL papers from this category in the table.
- Be direct. No filler. No "In recent years..."
- Do NOT write references or cross-category analysis — just this one section.
"""

CROSS_CATEGORY_PROMPT = """\
You are a research analyst. You've categorized papers on "{query}" into these groups:

{category_summaries}

Now write ONLY these sections:

#### Cross-Category Patterns
What patterns emerge across categories? Which are converging? \
What contradictions exist? Which papers bridge multiple categories?

#### Gaps & Opportunities
Be specific. Name concrete research questions nobody has addressed. \
Point to specific method/domain combinations that haven't been tried.

#### Open Access Papers
List any papers with free full-text URLs mentioned above.

Rules:
- Be direct and specific — no vague generalities
- Reference specific paper numbers when possible
- Do NOT repeat the per-category analysis
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
