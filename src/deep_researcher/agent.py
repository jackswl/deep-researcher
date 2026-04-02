from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config import Config
from deep_researcher.constants import (
    CATEGORIZE_BATCH_SIZE,
    CATEGORY_SYNTHESIS_TIMEOUT,
    CATEGORY_TOKEN_BUDGET,
    FALLBACK_MAX_PAPERS,
    FALLBACK_TOKEN_BUDGET,
    MAX_FINAL_CATEGORIES,
    MAX_SYNTHESIS_PAPERS,
    SCHOLAR_MAX_RESULTS,
)
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper
from deep_researcher.parsing import (
    build_tiered_corpus,
    parse_categories,
    parse_merged_categories,
    titles_match,
)
from deep_researcher.prompts import (
    CATEGORIZE_PROMPT,
    CATEGORY_SYNTHESIS_PROMPT,
    CLARIFY_PROMPT,
    CROSS_CATEGORY_PROMPT,
    MERGE_CATEGORIES_PROMPT,
)
from deep_researcher.report import get_output_folder, save_checkpoint, save_report

import logging
import threading

logger = logging.getLogger("deep_researcher")


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
                {"role": "system", "content": CLARIFY_PROMPT},
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
        """Search Google Scholar for up to SCHOLAR_MAX_RESULTS papers."""
        from scholarly import scholarly

        seen_titles: set[str] = set()
        count = 0
        try:
            for result in scholarly.search_pubs(query):
                if count >= SCHOLAR_MAX_RESULTS or self._cancel.is_set():
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

                if count % 20 == 0:
                    self.console.print(f"    [dim]{count} papers...[/dim]")
        except Exception as e:
            logger.warning("Google Scholar search failed: %s", e)

        self.console.print(f"  [green]Found {count} papers[/green]")

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
                    if results and titles_match(paper.title, results[0].get("title", "")):
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
                        # Verify CrossRef result is actually the same paper
                        cr_title = (items[0].get("title") or [""])[0]
                        if titles_match(paper.title, cr_title):
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

    def _apply_openalex(self, paper: Paper, work: dict) -> None:
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

            prompt = CATEGORIZE_PROMPT.format(
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
            all_categories = self._merge_categories(query, all_categories)

        return all_categories

    def _merge_categories(self, query: str, categories: dict[str, list[int]]) -> dict[str, list[int]]:
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
            self.console.print(f"    [dim]Merging {len(categories)} batch categories into {MAX_FINAL_CATEGORIES}...[/dim]")
            content = self.llm.chat_no_think([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Merge the categories now."},
            ])
            merged = parse_merged_categories(content, categories)
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
        corpus = build_tiered_corpus(indexed_papers, token_budget=CATEGORY_TOKEN_BUDGET)

        prompt = CATEGORY_SYNTHESIS_PROMPT.format(
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

        prompt = CROSS_CATEGORY_PROMPT.format(
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
        corpus = build_tiered_corpus(
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

