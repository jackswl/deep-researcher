# src/deep_researcher/orchestrator.py
"""Research pipeline orchestrator.

Calls tools through a uniform interface, manages PipelineState flow,
and provides layered error recovery (claude-code Principles 1-4).

The orchestrator ONLY calls tools — it never makes raw API/library calls.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config import Config
from deep_researcher.constants import (
    CATEGORY_SYNTHESIS_TIMEOUT,
    FALLBACK_MAX_PAPERS,
    FALLBACK_TOKEN_BUDGET,
    MAX_SYNTHESIS_PAPERS,
)
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper, PipelineState
from deep_researcher.parsing import build_tiered_corpus
from deep_researcher.prompts import CLARIFY_PROMPT
from deep_researcher.report import get_output_folder, save_checkpoint, save_report
from deep_researcher.tools.categorize import CategorizeTool
from deep_researcher.tools.cross_analysis import CrossAnalysisTool
from deep_researcher.tools.enrichment import EnrichmentTool
from deep_researcher.tools.scholar_search import ScholarSearchTool
from deep_researcher.tools.synthesize import SynthesisTool

logger = logging.getLogger("deep_researcher")


class Orchestrator:
    """Pipeline orchestrator — the ONLY place that calls tools.

    Each phase:
    1. Calls a tool with validated input
    2. Handles errors with recovery (retry -> fallback -> degrade)
    3. Returns new PipelineState (never mutates input)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.console = Console()
        self._cancel = threading.Event()
        self._output_folder: str = ""

        # Pipeline tools (Principle 1: tools as unit of action)
        self._search_tool = ScholarSearchTool()
        self._enrichment_tool = EnrichmentTool()
        self._categorize_tool = CategorizeTool(llm=self.llm)
        self._synthesize_tool = SynthesisTool(llm=self.llm)
        self._cross_analysis_tool = CrossAnalysisTool(llm=self.llm)

    def cancel(self) -> None:
        """Signal the orchestrator to stop gracefully."""
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

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def research(self, query: str) -> str:
        """Run the full research pipeline.

        State flows immutably through phases (Principle 3):
        search -> enrich -> synthesize -> report
        """
        self.console.print(Panel(
            f"[bold]{query}[/bold]",
            title="Deep Researcher",
            border_style="blue",
        ))

        state = PipelineState(query=query)
        self._output_folder = get_output_folder(query, self.config.output_dir)

        # Phase 1: Search
        self.console.print("\n[bold blue]Phase 1: Searching Google Scholar...[/bold blue]")
        state = self._run_search(state)

        if not state.papers:
            self.console.print("[yellow]No papers found.[/yellow]")
            return "No papers were found for this query."

        # Phase 2: Enrich
        self.console.print(f"\n[bold blue]Phase 2: Enriching {len(state.papers)} papers...[/bold blue]")
        state = self._run_enrichment(state)

        # Checkpoint
        if state.papers and self._output_folder:
            try:
                save_checkpoint(state.papers, self._output_folder)
            except Exception:
                logger.debug("Checkpoint save failed", exc_info=True)

        # Phase 3: Synthesize
        self.console.print(f"\n[bold blue]Phase 3: Synthesizing {len(state.papers)} papers...[/bold blue]")
        state = self._run_synthesis(state)

        self._print_summary(state)
        self._save(state)
        return state.report

    # ------------------------------------------------------------------
    # Phase implementations (each calls tools, returns new state)
    # ------------------------------------------------------------------

    def _run_search(self, state: PipelineState) -> PipelineState:
        """Phase 1: Search Google Scholar via tool."""
        result = self._search_tool.safe_execute(
            query=state.query,
            cancel=self._cancel,
        )
        papers: dict[str, Paper] = {}
        for paper in result.papers:
            key = paper.unique_key
            if key not in papers:
                papers[key] = paper

        self.console.print(f"  [green]Found {len(papers)} papers[/green]")
        return state.evolve(papers=papers)

    def _run_enrichment(self, state: PipelineState) -> PipelineState:
        """Phase 2: Enrich papers via tool (concurrent HTTP)."""
        result = self._enrichment_tool.safe_execute(
            papers=list(state.papers.values()),
            email=self.config.email,
            cancel=self._cancel,
        )

        # Rebuild papers dict preserving original keys (match old behavior:
        # enrichment adds metadata but doesn't change which papers exist)
        original_keys = list(state.papers.keys())
        enriched: dict[str, Paper] = {}
        for i, paper in enumerate(result.papers):
            if i < len(original_keys):
                enriched[original_keys[i]] = paper
            else:
                enriched[paper.unique_key] = paper

        self.console.print(f"  [green]{result.text}[/green]")
        return state.evolve(papers=enriched)

    def _run_synthesis(self, state: PipelineState) -> PipelineState:
        """Phase 3: Multi-step synthesis with layered error recovery.

        Recovery layers (Principle 4):
        1. Per-category retry -> skip on failure
        2. Categorization failure -> fallback synthesis
        3. All categories fail -> fallback synthesis
        """
        def _sort_key(p: Paper) -> tuple:
            return (-(p.citation_count or 0), -(p.year or 0))

        all_papers = state.papers
        if len(all_papers) > MAX_SYNTHESIS_PAPERS:
            sorted_all = sorted(all_papers.values(), key=_sort_key)
            synthesis_papers = sorted_all[:MAX_SYNTHESIS_PAPERS]
            self.console.print(
                f"  [yellow]Corpus capped: synthesizing top {MAX_SYNTHESIS_PAPERS} of "
                f"{len(all_papers)} papers (all saved to papers.json)[/yellow]"
            )
        else:
            synthesis_papers = sorted(all_papers.values(), key=_sort_key)

        state = state.evolve(synthesis_papers=synthesis_papers)

        # Step 1: Categorize
        self.console.print("  [cyan]Step 1/3: Categorizing papers...[/cyan]")
        cat_result = self._categorize_tool.safe_execute(
            papers=synthesis_papers,
            query=state.query,
        )
        categories = cat_result.data

        if not categories:
            self.console.print("  [yellow]Categorization failed — falling back to single-pass synthesis[/yellow]")
            report = self._fallback_synthesis(state.query, synthesis_papers)
            return state.evolve(report=report)

        state = state.evolve(categories=categories)
        self.console.print(f"  [green]Found {len(categories)} categories[/green]")
        for name, indices in categories.items():
            self.console.print(f"    {name}: {len(indices)} papers")

        # Step 2: Per-category synthesis
        self.console.print("  [cyan]Step 2/3: Synthesizing per category...[/cyan]")
        category_sections: list[tuple[str, str]] = []
        for cat_name, paper_indices in categories.items():
            if self._cancel.is_set():
                self.console.print("  [yellow]Synthesis cancelled.[/yellow]")
                break
            cat_indexed = [(i, synthesis_papers[i]) for i in paper_indices if i < len(synthesis_papers)]
            if not cat_indexed:
                continue
            self.console.print(f"    [cyan]{cat_name}[/cyan] ({len(cat_indexed)} papers)...")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        self._synthesize_tool.safe_execute,
                        indexed_papers=cat_indexed,
                        query=state.query,
                        category_name=cat_name,
                    )
                    result = future.result(timeout=CATEGORY_SYNTHESIS_TIMEOUT)
                if not result.text.startswith("Synthesis failed"):
                    category_sections.append((cat_name, result.text))
                else:
                    self.console.print(f"    [red]{result.text} — skipping[/red]")
            except concurrent.futures.TimeoutError:
                self.console.print(f"    [red]Timed out — skipping[/red]")
            except Exception as e:
                self.console.print(f"    [red]Failed: {e} — skipping[/red]")

        if not category_sections:
            self.console.print("  [yellow]All categories failed — falling back[/yellow]")
            report = self._fallback_synthesis(state.query, synthesis_papers)
            return state.evolve(report=report)

        state = state.evolve(category_sections=category_sections)

        # Step 3: Cross-category analysis
        self.console.print("  [cyan]Step 3/3: Cross-category analysis...[/cyan]")
        cross_result = self._cross_analysis_tool.safe_execute(
            sections=category_sections,
            query=state.query,
        )
        state = state.evolve(cross_section=cross_result.text)

        # Step 4: Assemble report (programmatic — not LLM)
        report = self._assemble_report(state)
        return state.evolve(report=report)

    # ------------------------------------------------------------------
    # Report assembly (programmatic — never LLM-generated)
    # ------------------------------------------------------------------

    def _assemble_report(self, state: PipelineState) -> str:
        """Assemble the final report programmatically."""
        papers = state.synthesis_papers
        categories = state.categories or {}
        sections = state.category_sections
        cross_section = state.cross_section

        years = [p.year for p in papers if p.year]
        yr_range = f"{min(years)}-{max(years)}" if years else "unknown"
        total = len(state.papers)

        has_doi = sum(1 for p in papers if p.doi)
        parts = [
            f"### {state.query}\n",
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
        """Single-pass fallback if multi-step synthesis fails (recovery layer 2)."""
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
    # Display & save helpers
    # ------------------------------------------------------------------

    def _print_summary(self, state: PipelineState) -> None:
        self.console.print(f"\n[green]Research complete.[/green]")
        table = Table(title="Research Summary", show_header=False, border_style="green")
        table.add_row("Papers found", str(len(state.papers)))

        has_doi = sum(1 for p in state.papers.values() if p.doi)
        table.add_row("With DOIs", f"{has_doi}/{len(state.papers)}")

        has_abstract = sum(1 for p in state.papers.values() if p.abstract and len(p.abstract) > 200)
        table.add_row("Full abstracts", f"{has_abstract}/{len(state.papers)}")

        years = [p.year for p in state.papers.values() if p.year]
        if years:
            table.add_row("Year range", f"{min(years)}-{max(years)}")

        oa_count = sum(1 for p in state.papers.values() if p.open_access_url)
        if oa_count:
            table.add_row("Open access", f"{oa_count}/{len(state.papers)}")

        self.console.print(table)

    def _save(self, state: PipelineState) -> None:
        if not state.report.strip():
            return
        try:
            paths = save_report(
                state.query, state.report, state.papers,
                self.config.output_dir, folder=self._output_folder or None,
            )
            self.console.print(f"\n[green bold]Files saved:[/green bold]")
            for label, path in paths.items():
                self.console.print(f"  {label}: [blue]{path}[/blue]")
        except Exception as e:
            self.console.print(f"[red]Error saving report: {e}[/red]")
