from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config import Config
from deep_researcher.constants import (
    ABSTRACT_MAX_CHARS,
    ABSTRACT_MIN_CUT,
    CATEGORIZE_BATCH_SIZE,
    CATEGORY_SYNTHESIS_TIMEOUT,
    CATEGORY_TOKEN_BUDGET,
    CHARS_PER_TOKEN,
    FALLBACK_MAX_PAPERS,
    FALLBACK_TOKEN_BUDGET,
    MAX_FINAL_CATEGORIES,
    MAX_SYNTHESIS_PAPERS,
    MIN_CATEGORIZATION_COVERAGE,
    NUM_QUERY_VARIATIONS,
    SCHOLAR_RESULTS_PER_QUERY,
)
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper
from deep_researcher.report import get_output_folder, save_checkpoint, save_report

import logging
import threading

logger = logging.getLogger("deep_researcher")


# --- Synthesis prompts ---

_CATEGORIZE_PROMPT = """\
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

_MERGE_CATEGORIES_PROMPT = """\
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

_CATEGORY_SYNTHESIS_PROMPT = """\
You are a research analyst writing one section of a detailed literature review on: "{query}"

This section covers the category: **{category}** ({count} papers)

## Papers in this category
{corpus}

## Write this section with DETAILED analysis. Reference papers by [number] throughout.

**What this group does:**
Write a detailed paragraph (4-6 sentences) explaining the shared approach/theme.
Reference individual papers: e.g., "Smith et al. [1] introduced X. Jones et al. [2] extended this by Y.
Lee et al. [3] proposed a different approach using Z." Include specific contributions from each paper.

**Key methods:**
Write a detailed paragraph describing the specific methods and techniques.
For each method, cite which paper(s) used it: e.g., "Neural semantic parsing [1], domain-specific
fine-tuning on regulatory corpora [3], hybrid NLP-BIM integration [4]. Smith et al. [1] used
first-order logic transformation, while Jones et al. [2] employed prompt engineering with chain-of-thought."

**Main findings:**
Write a detailed paragraph on collective findings. Include specific results where available:
e.g., "Smith et al. [1] reported 92% accuracy on structured clauses but only 67% on conditional ones.
Jones et al. [2] found that fine-tuning outperformed zero-shot by 23%." Note agreements and disagreements.

**Limitations:**
Write a paragraph on common weaknesses, citing specific papers where relevant.

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
(Include EVERY paper listed above in the table)

Rules:
- Be DETAILED — this is a literature review, not an abstract
- Reference papers by [number] throughout ALL sections, not just the table
- Include specific metrics, results, and comparisons where the abstracts mention them
- Be direct. No filler. No "In recent years..."
- Include ALL papers from this category in the table
- Do NOT invent papers or results — only use what's in the abstracts above
- Do NOT write references or cross-category analysis — just this one section
"""

_CROSS_CATEGORY_PROMPT = """\
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

_CLARIFY_PROMPT = """\
You are a research assistant helping to refine a research question before searching academic databases.

Given the user's research topic, generate exactly 3 short, focused clarifying questions that would \
help narrow the search and produce better results. Focus on:
- Specific subfield or application domain
- Time period or recency preferences
- Methodological focus (theoretical, empirical, computational, etc.)

Format: Return ONLY the 3 questions, one per line, numbered 1-3. No preamble.
"""


class ResearchAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.papers: dict[str, Paper] = {}
        self.console = Console()
        self._output_folder: str = ""
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Signal the agent to stop gracefully."""
        self._cancel.set()

    def clarify(self, query: str) -> str:
        """Ask clarifying questions and return an enhanced query."""
        self.console.print("\n[bold]Generating clarifying questions...[/bold]\n")
        try:
            response = self.llm.chat([
                {"role": "system", "content": _CLARIFY_PROMPT},
                {"role": "user", "content": query},
            ])
            questions = (response.content or "").strip()
        except Exception as e:
            self.console.print(f"[yellow]Could not generate questions: {e}. Proceeding with original query.[/yellow]")
            return query

        if not questions:
            return query

        self.console.print(questions)
        self.console.print("\n[dim]Answer each question (press Enter to skip):[/dim]\n")

        answers = []
        for line in questions.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                answer = input(f"  {line}\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if answer:
                answers.append(answer)

        if not answers:
            return query

        enhanced = f"{query}\n\nAdditional context from the researcher:\n"
        enhanced += "\n".join(f"- {a}" for a in answers)
        self.console.print(f"\n[green]Enhanced query ready.[/green]")
        return enhanced

    def research(self, query: str) -> str:
        self.console.print(Panel(
            f"[bold]{query}[/bold]",
            title="Deep Researcher",
            border_style="blue",
        ))

        self._output_folder = get_output_folder(query, self.config.output_dir)

        # === Phase 1: Search Google Scholar ===
        self.console.print("\n[bold blue]Phase 1: Searching Google Scholar...[/bold blue]")
        self._search_scholar(query)

        # === Phase 2: Enrich via OpenAlex + CrossRef ===
        self.console.print(f"\n[bold blue]Phase 2: Enriching {len(self.papers)} papers...[/bold blue]")
        self._enrich_papers()

        # Checkpoint
        if self.papers and self._output_folder:
            try:
                save_checkpoint(self.papers, self._output_folder)
            except Exception:
                logger.debug("Checkpoint save failed", exc_info=True)

        # === Phase 3: Synthesize ===
        self.console.print(f"\n[bold blue]Phase 3: Synthesizing {len(self.papers)} papers...[/bold blue]")
        report = self._synthesis_phase(query)

        self._print_summary()
        self._save(query, report)
        return report

    # ------------------------------------------------------------------
    # Phase 1: Google Scholar search
    # ------------------------------------------------------------------

    def _search_scholar(self, query: str) -> None:
        """Search Google Scholar with the original query + LLM-generated variations."""
        from scholarly import scholarly

        # Generate query variations for broader coverage
        self.console.print("  [dim]Generating query variations...[/dim]")
        try:
            variations_text = self.llm.chat_no_think([
                {"role": "system", "content": (
                    f'/no_think\nGenerate {NUM_QUERY_VARIATIONS} Google Scholar search queries for: "{query}"\n'
                    f'Rephrase the same topic using synonyms and alternative academic phrasing.\n'
                    f'Stay close to the original topic — do NOT expand to tangentially related fields.\n'
                    f'One per line, no numbering, no explanation.'
                )},
                {"role": "user", "content": "Generate the queries now."},
            ])
            variations = [
                l.strip().strip("\"'") for l in variations_text.strip().split("\n")
                if l.strip() and len(l.strip()) > 10
            ][:NUM_QUERY_VARIATIONS]
        except Exception as e:
            logger.warning("Query variation generation failed: %s", e)
            variations = []

        all_queries = [query] + variations
        for q in all_queries:
            self.console.print(f"    [dim]{q}[/dim]")

        # Search each query
        seen_titles: set[str] = set()
        for q in all_queries:
            if self._cancel.is_set():
                self.console.print("  [yellow]Cancelled.[/yellow]")
                return
            count = 0
            try:
                for result in scholarly.search_pubs(q):
                    if count >= SCHOLAR_RESULTS_PER_QUERY:
                        break
                    title = result.get("bib", {}).get("title", "")
                    if not title or title.lower().strip() in seen_titles:
                        continue
                    seen_titles.add(title.lower().strip())

                    bib = result.get("bib", {})
                    authors = bib.get("author", [])
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(" and ")]

                    year_str = bib.get("pub_year", "")
                    year = int(year_str) if year_str and year_str.isdigit() else None

                    paper = Paper(
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=bib.get("abstract", ""),
                        journal=bib.get("venue", ""),
                        citation_count=result.get("num_citations", 0) or None,
                        url=result.get("pub_url", ""),
                        source="google_scholar",
                    )
                    key = paper.unique_key
                    if key not in self.papers:
                        self.papers[key] = paper
                    count += 1
            except Exception as e:
                logger.debug("Scholar search failed for '%s': %s", q[:50], e)

            self.console.print(
                f"  [cyan]{q[:60]}[/cyan] -> +{count} new (total: {len(self.papers)})"
            )

    # ------------------------------------------------------------------
    # Phase 2: OpenAlex + CrossRef enrichment
    # ------------------------------------------------------------------

    def _enrich_papers(self) -> None:
        """Enrich papers with full metadata from OpenAlex (+ CrossRef DOI fallback)."""
        import httpx

        enriched = 0
        total = len(self.papers)
        email = self.config.email or "deep-researcher@example.com"
        ua_openalex = {"User-Agent": f"mailto:{email}"}
        ua_crossref = {"User-Agent": f"deep-researcher (mailto:{email})"}

        for i, paper in enumerate(self.papers.values()):
            if self._cancel.is_set():
                break
            try:
                # Attempt 1: OpenAlex title search
                resp = httpx.get(
                    "https://api.openalex.org/works",
                    params={"filter": f"title.search:{paper.title[:100]}", "per_page": 1},
                    headers=ua_openalex, timeout=10,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        self._apply_openalex(paper, results[0])
                        enriched += 1
                        continue

                # Attempt 2: CrossRef title -> DOI -> OpenAlex
                resp = httpx.get(
                    "https://api.crossref.org/works",
                    params={"query.title": paper.title, "rows": 1, "select": "DOI,title"},
                    headers=ua_crossref, timeout=10,
                )
                if resp.status_code == 200:
                    items = resp.json().get("message", {}).get("items", [])
                    if items and items[0].get("DOI"):
                        doi = items[0]["DOI"]
                        resp2 = httpx.get(
                            f"https://api.openalex.org/works/doi:{doi}",
                            headers=ua_openalex, timeout=10,
                        )
                        if resp2.status_code == 200:
                            self._apply_openalex(paper, resp2.json())
                            enriched += 1
                            continue
            except Exception:
                logger.debug("Enrichment failed for '%s'", paper.title[:50], exc_info=True)

            if (i + 1) % 20 == 0:
                self.console.print(f"    [dim]{i + 1}/{total}...[/dim]")

        has_abstract = sum(1 for p in self.papers.values() if p.abstract and len(p.abstract) > 200)
        has_doi = sum(1 for p in self.papers.values() if p.doi)
        self.console.print(
            f"  [green]Enriched {enriched}/{total} | Full abstracts: {has_abstract} | DOIs: {has_doi}[/green]"
        )

    @staticmethod
    def _apply_openalex(paper: Paper, work: dict) -> None:
        """Apply OpenAlex metadata to a Paper object."""
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        if doi:
            paper.doi = doi

        # Reconstruct abstract from inverted index
        inv_idx = work.get("abstract_inverted_index")
        if inv_idx:
            words = [""] * (max(max(pos) for pos in inv_idx.values()) + 1)
            for word, positions in inv_idx.items():
                for pos in positions:
                    words[pos] = word
            full_abstract = " ".join(w for w in words if w)
            if full_abstract and len(full_abstract) > len(paper.abstract or ""):
                paper.abstract = full_abstract

        source = (work.get("primary_location") or {}).get("source") or {}
        if source.get("display_name"):
            paper.journal = source["display_name"]

        oa = work.get("open_access", {})
        if oa.get("oa_url"):
            paper.open_access_url = oa["oa_url"]

        oa_cites = work.get("cited_by_count", 0)
        if oa_cites and (paper.citation_count is None or oa_cites > paper.citation_count):
            paper.citation_count = oa_cites

    # ------------------------------------------------------------------
    # Phase 3: Multi-step synthesis
    # ------------------------------------------------------------------

    def _synthesis_phase(self, query: str) -> str:
        """Multi-step synthesis: categorize -> per-category -> cross-category -> assemble."""
        if not self.papers:
            return "No papers were found for this query."

        # Sort by citations
        all_papers = self.papers

        def _paper_sort_key(p: Paper) -> tuple:
            return (-(p.citation_count or 0), -(p.year or 0))

        if len(all_papers) > MAX_SYNTHESIS_PAPERS:
            sorted_all = sorted(all_papers.values(), key=_paper_sort_key)
            synthesis_papers = sorted_all[:MAX_SYNTHESIS_PAPERS]
            self.console.print(
                f"  [yellow]Corpus capped: synthesizing top {MAX_SYNTHESIS_PAPERS} of "
                f"{len(all_papers)} papers (all saved to papers.json)[/yellow]"
            )
        else:
            synthesis_papers = sorted(all_papers.values(), key=_paper_sort_key)

        # === Step 1: Categorize ===
        self.console.print("  [cyan]Step 1/3: Categorizing papers...[/cyan]")
        categories = self._categorize_papers(query, synthesis_papers)
        if not categories:
            self.console.print("  [yellow]Categorization failed — falling back to single-pass synthesis[/yellow]")
            return self._fallback_synthesis(query, synthesis_papers)

        self.console.print(f"  [green]Found {len(categories)} categories[/green]")
        for name, indices in categories.items():
            self.console.print(f"    {name}: {len(indices)} papers")

        # === Step 2: Per-category synthesis ===
        self.console.print("  [cyan]Step 2/3: Synthesizing per category...[/cyan]")
        category_sections: list[tuple[str, str]] = []
        for cat_name, paper_indices in categories.items():
            if self._cancel.is_set():
                self.console.print("  [yellow]Synthesis cancelled.[/yellow]")
                break
            # Build (global_index, paper) pairs so numbering matches the reference list
            cat_indexed = [(i, synthesis_papers[i]) for i in paper_indices if i < len(synthesis_papers)]
            if not cat_indexed:
                continue
            self.console.print(f"    [cyan]{cat_name}[/cyan] ({len(cat_indexed)} papers)...")
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._synthesize_category, query, cat_name, cat_indexed)
                    section = future.result(timeout=CATEGORY_SYNTHESIS_TIMEOUT)
                category_sections.append((cat_name, section))
            except concurrent.futures.TimeoutError:
                self.console.print(f"    [red]Timed out — skipping[/red]")
                continue
            except Exception as e:
                self.console.print(f"    [red]Failed: {e} — skipping[/red]")
                continue

        if not category_sections:
            self.console.print("  [yellow]All categories failed — falling back[/yellow]")
            return self._fallback_synthesis(query, synthesis_papers)

        # === Step 3: Cross-category analysis ===
        self.console.print("  [cyan]Step 3/3: Cross-category analysis...[/cyan]")
        cross_section = self._cross_category_analysis(query, category_sections)

        # === Step 4: Assemble report ===
        return self._assemble_report(query, synthesis_papers, categories, category_sections, cross_section)

    def _categorize_papers(self, query: str, papers: list[Paper]) -> dict[str, list[int]]:
        """Assign papers to categories in batches, then merge if too many."""
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

            prompt = _CATEGORIZE_PROMPT.format(
                count=len(batch),
                query=query,
                paper_list="\n".join(lines),
            )

            try:
                self.console.print(f"    [dim]Batch {batch_start + 1}-{batch_end} of {len(papers)}...[/dim]")
                content = self.llm.chat_no_think([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Categorize these papers now."},
                ])
                batch_cats = _parse_categories(content, len(papers))
                for cat_name, indices in batch_cats.items():
                    if cat_name in all_categories:
                        all_categories[cat_name].extend(indices)
                    else:
                        all_categories[cat_name] = indices
            except Exception as e:
                logger.warning("Categorization batch %d-%d failed: %s", batch_start, batch_end, e)
                continue

        if len(all_categories) > MAX_FINAL_CATEGORIES:
            all_categories = self._merge_categories(query, all_categories)

        return all_categories

    def _merge_categories(self, query: str, categories: dict[str, list[int]]) -> dict[str, list[int]]:
        """Merge semantically similar categories into MAX_FINAL_CATEGORIES groups."""
        cat_list = "\n".join(
            f"- {name} ({len(indices)} papers)" for name, indices in categories.items()
        )
        prompt = _MERGE_CATEGORIES_PROMPT.format(
            query=query,
            count=len(categories),
            target=MAX_FINAL_CATEGORIES,
            category_list=cat_list,
        )

        try:
            self.console.print(f"    [dim]Merging {len(categories)} batch categories into {MAX_FINAL_CATEGORIES}...[/dim]")
            content = self.llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Merge the categories now."},
            ])
            merged = _parse_merged_categories(content, categories)
            if merged:
                return merged
        except Exception as e:
            logger.warning("Category merge failed: %s", e)

        self.console.print(f"    [yellow]Merge failed — keeping {MAX_FINAL_CATEGORIES} largest categories[/yellow]")
        sorted_cats = sorted(categories.items(), key=lambda x: -len(x[1]))
        return dict(sorted_cats[:MAX_FINAL_CATEGORIES])

    def _synthesize_category(self, query: str, cat_name: str, indexed_papers: list[tuple[int, Paper]]) -> str:
        """Synthesize one category. indexed_papers are (global_index, Paper) pairs
        so [N] references match the final reference list."""
        corpus = _build_tiered_corpus(indexed_papers, token_budget=CATEGORY_TOKEN_BUDGET)

        prompt = _CATEGORY_SYNTHESIS_PROMPT.format(
            query=query,
            category=cat_name,
            count=len(indexed_papers),
            corpus=corpus,
        )

        return self.llm.chat_no_think([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Write the synthesis for: {cat_name}"},
        ])

    def _cross_category_analysis(self, query: str, sections: list[tuple[str, str]]) -> str:
        """Analyze patterns across categories."""
        summaries = []
        for name, content in sections:
            summary = content[:500]
            if len(content) > 500:
                cut = summary.rfind(". ")
                summary = summary[:cut + 1] if cut > 300 else summary + "..."
            summaries.append(f"**{name}:**\n{summary}")

        prompt = _CROSS_CATEGORY_PROMPT.format(
            query=query,
            category_summaries="\n\n".join(summaries),
        )

        try:
            return self.llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the cross-category analysis."},
            ])
        except Exception as e:
            return f"Cross-category analysis unavailable: {e}"

    def _assemble_report(
        self, query: str, papers: list[Paper],
        categories: dict[str, list[int]],
        sections: list[tuple[str, str]], cross_section: str,
    ) -> str:
        """Assemble the final report programmatically."""
        years = [p.year for p in papers if p.year]
        yr_range = f"{min(years)}-{max(years)}" if years else "unknown"
        total = len(self.papers)

        has_doi = sum(1 for p in papers if p.doi)
        parts = [
            f"### {query}\n",
            f"#### Coverage",
            f"{total} papers found via Google Scholar, enriched via OpenAlex. "
            f"Years {yr_range}. {has_doi} with DOIs.\n",
            "#### Categories\n",
        ]

        for cat_name, content in sections:
            cat_indices = categories.get(cat_name, [])
            parts.append(f"##### {cat_name} ({len(cat_indices)} papers)\n")
            parts.append(content)
            parts.append("")

        parts.append(cross_section)
        parts.append("")

        # References (generated programmatically — never by LLM)
        parts.append("#### References\n")
        for i, p in enumerate(papers, 1):
            author = p.authors[0] if p.authors else "Unknown"
            if len(p.authors) > 1:
                author += " et al."
            year = p.year or "n.d."
            journal = f" *{p.journal}*." if p.journal else ""
            doi = f" DOI: {p.doi}" if p.doi else ""
            oa = f" [Open Access]({p.open_access_url})" if p.open_access_url else ""
            parts.append(f"[{i}] {author} ({year}). {p.title}.{journal}{doi}{oa}")

        return "\n".join(parts)

    def _fallback_synthesis(self, query: str, papers: list[Paper]) -> str:
        """Single-pass fallback if multi-step synthesis fails."""
        top_papers = papers[:FALLBACK_MAX_PAPERS]
        corpus = _build_tiered_corpus(
            list(enumerate(top_papers)),
            token_budget=FALLBACK_TOKEN_BUDGET,
        )
        prompt = (
            f"Write a brief literature review on \"{query}\" based on these {len(top_papers)} papers. "
            f"Categorize by theme, include a table per category.\n\n{corpus}"
        )
        try:
            return self.llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the review."},
            ])
        except Exception as e:
            return f"Synthesis failed: {e}"

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        self.console.print(f"\n[green]Research complete.[/green]")
        table = Table(title="Research Summary", show_header=False, border_style="green")
        table.add_row("Papers found", str(len(self.papers)))

        has_doi = sum(1 for p in self.papers.values() if p.doi)
        table.add_row("With DOIs", f"{has_doi}/{len(self.papers)}")

        has_abstract = sum(1 for p in self.papers.values() if p.abstract and len(p.abstract) > 200)
        table.add_row("Full abstracts", f"{has_abstract}/{len(self.papers)}")

        years = [p.year for p in self.papers.values() if p.year]
        if years:
            table.add_row("Year range", f"{min(years)}-{max(years)}")

        oa_count = sum(1 for p in self.papers.values() if p.open_access_url)
        if oa_count:
            table.add_row("Open access", f"{oa_count}/{len(self.papers)}")

        self.console.print(table)

    def _save(self, query: str, report: str) -> None:
        if not report.strip():
            return
        try:
            paths = save_report(query, report, self.papers, self.config.output_dir, folder=self._output_folder or None)
            self.console.print(f"\n[green bold]Files saved:[/green bold]")
            for label, path in paths.items():
                self.console.print(f"  {label}: [blue]{path}[/blue]")
        except Exception as e:
            self.console.print(f"[red]Error saving report: {e}[/red]")


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _parse_categories(text: str, paper_count: int) -> dict[str, list[int]]:
    """Parse LLM category output into {name: [0-based indices]}."""
    import re as _re
    categories: dict[str, list[int]] = {}
    current_cat = None

    for line in text.split("\n"):
        cleaned = _re.sub(r"[*_`#>]", "", line).strip()
        cleaned = _re.sub(r"^[-+]\s*", "", cleaned)
        if not cleaned:
            continue

        cat_match = _re.match(r"(?:CATEGORY|Category)\s*:\s*(.+)", cleaned, _re.IGNORECASE)
        if cat_match:
            current_cat = cat_match.group(1).strip()
            continue

        papers_match = _re.match(r"(?:PAPERS|Papers)\s*:\s*(.+)", cleaned, _re.IGNORECASE)
        if papers_match and current_cat:
            nums = _re.findall(r"\d+", papers_match.group(1))
            indices = [int(n) - 1 for n in nums if 0 < int(n) <= paper_count]
            if indices:
                categories[current_cat] = indices
            current_cat = None

    assigned = set()
    for indices in categories.values():
        assigned.update(indices)
    if len(assigned) < paper_count * MIN_CATEGORIZATION_COVERAGE:
        logger.warning("Categorization covered only %d/%d papers", len(assigned), paper_count)

    return categories


def _parse_merged_categories(
    text: str, original: dict[str, list[int]]
) -> dict[str, list[int]] | None:
    """Parse LLM merge output into consolidated categories."""
    import re as _re
    merged: dict[str, list[int]] = {}
    current_final = None

    for line in text.split("\n"):
        cleaned = _re.sub(r"[*_`#>]", "", line).strip()
        cleaned = _re.sub(r"^[-+]\s*", "", cleaned)
        if not cleaned:
            continue

        final_match = _re.match(r"(?:FINAL)\s*:\s*(.+)", cleaned, _re.IGNORECASE)
        if final_match:
            current_final = final_match.group(1).strip()
            continue

        merge_match = _re.match(r"(?:MERGE)\s*:\s*(.+)", cleaned, _re.IGNORECASE)
        if merge_match and current_final:
            old_names = [n.strip() for n in merge_match.group(1).split(",")]
            indices: list[int] = []
            for old_name in old_names:
                if old_name in original:
                    indices.extend(original[old_name])
                else:
                    for orig_name in original:
                        if old_name.lower() in orig_name.lower() or orig_name.lower() in old_name.lower():
                            indices.extend(original[orig_name])
                            break
            if indices:
                merged[current_final] = indices
            current_final = None

    if not merged:
        return None
    total_papers = sum(len(v) for v in merged.values())
    orig_total = sum(len(v) for v in original.values())
    if total_papers < orig_total * 0.5:
        logger.warning("Category merge lost too many papers (%d/%d)", total_papers, orig_total)
        return None
    return merged


def _build_tiered_corpus(indexed_papers: list, token_budget: int = 15000) -> str:
    """Build a token-budgeted corpus with progressive compression.

    indexed_papers: list of (global_index, Paper) tuples.
    Uses global indices for [N] references so they match the final reference list.
    """
    if not indexed_papers:
        return ""

    # Sort by citations within the category
    sorted_pairs = sorted(indexed_papers, key=lambda x: (-(x[1].citation_count or 0), -(x[1].year or 0)))

    lines = []
    tokens_used = 0
    level1_budget = int(token_budget * 0.6)
    level2_budget = int(token_budget * 0.9)

    for processed, (idx, p) in enumerate(sorted_pairs):
        ref_num = idx + 1  # 0-based index -> 1-based reference number

        full_entry = _paper_full_entry(ref_num, p)
        full_tokens = len(full_entry) // CHARS_PER_TOKEN
        if tokens_used + full_tokens < level1_budget:
            lines.append(full_entry)
            tokens_used += full_tokens
            continue

        short_entry = _paper_short_entry(ref_num, p)
        short_tokens = len(short_entry) // CHARS_PER_TOKEN
        if tokens_used + short_tokens < level2_budget:
            lines.append(short_entry)
            tokens_used += short_tokens
            continue

        remaining = len(sorted_pairs) - processed
        if remaining > 0:
            lines.append(f"\n(+ {remaining} additional papers in this category, sorted by citation count)")
        break

    return "\n".join(lines)


def _paper_full_entry(idx: int, p) -> str:
    """Full paper entry with abstract for tier-1 papers."""
    parts = []
    author = p.authors[0] if p.authors else "Unknown"
    if len(p.authors) > 1:
        author += " et al."
    parts.append(f"[{idx}] {p.title}")
    meta = [author]
    if p.year:
        meta.append(str(p.year))
    if p.journal:
        meta.append(p.journal)
    if p.citation_count is not None:
        meta.append(f"{p.citation_count} citations")
    if p.doi:
        meta.append(f"DOI: {p.doi}")
    parts.append(f"   {' | '.join(meta)}")
    if p.abstract:
        abstract = p.abstract[:ABSTRACT_MAX_CHARS]
        if len(p.abstract) > ABSTRACT_MAX_CHARS:
            cut = abstract.rfind(". ")
            abstract = abstract[:cut + 1] if cut > ABSTRACT_MIN_CUT else abstract + "..."
        parts.append(f"   Abstract: {abstract}")
    return "\n".join(parts)


def _paper_short_entry(idx: int, p) -> str:
    """One-line compressed entry for tier-2 papers."""
    author = p.authors[0].split()[-1] if p.authors else "Unknown"
    year = p.year or "n.d."
    cites = f", {p.citation_count} cites" if p.citation_count else ""
    return f"[{idx}] {p.title} ({author}, {year}{cites})"


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s
