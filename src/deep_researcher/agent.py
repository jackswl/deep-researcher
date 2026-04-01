from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config import Config
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper
from deep_researcher.report import get_output_folder, save_checkpoint, save_report
from deep_researcher.tools import build_tool_registry

import logging

logger = logging.getLogger("deep_researcher")


# --- Search phase prompt: focus on GATHERING papers, not writing ---

def _build_search_prompt(config: Config) -> str:
    year_note = ""
    if config.start_year is not None or config.end_year is not None:
        yr_start = config.start_year if config.start_year is not None else "any"
        yr_end = config.end_year if config.end_year is not None else "present"
        year_note = f"\n**Year filter active: {yr_start} to {yr_end}.** Results are automatically filtered, but you should still prioritize queries relevant to this period.\n"

    return f"""\
You are a research paper collector. Your ONLY job right now is to find as many relevant \
papers as possible on the given topic. Do NOT write a report. Just search.
{year_note}
## Strategy — Quality First, Precise Queries
1. **Start with Scopus and IEEE Xplore** — these are the primary sources for peer-reviewed \
engineering and CS literature. Search these FIRST with your best queries.
2. **Then PubMed** if the topic has biomedical/health relevance.
3. **Then CrossRef and OpenAlex** for additional publisher coverage.
4. **Use Semantic Scholar primarily for citation chains** — after finding good papers in \
Scopus/IEEE, use get_citations to follow their references and citing papers.
5. **arXiv and CORE last** — ONLY for specific, targeted queries (e.g., a known preprint). \
Do NOT use broad queries on arXiv — it returns noise.
6. Break the topic into {config.breadth} different search angles using varied terminology.
7. Use get_paper_details on papers that appear in multiple databases.
8. Look for survey/review papers — they reference dozens of other relevant papers.

## CRITICAL: Query Formulation
- Use **specific multi-word phrases**, not single keywords
- GOOD: "transformer neural network structural health monitoring"
- GOOD: "attention mechanism damage detection bridge"
- GOOD: "vision transformer crack segmentation infrastructure"
- BAD: "monitoring" (too vague — matches weather monitoring, health monitoring, etc.)
- BAD: "transformer" alone (matches electrical transformers, not neural networks)
- BAD: "structural health" (too broad — matches unrelated fields)
- Each query should combine the METHOD + DOMAIN to get precise results

## When to Stop
Stop searching (respond WITHOUT any tool calls) when:
- You have found 20+ relevant papers from peer-reviewed sources
- New searches mostly return papers you already found
- You have searched 3+ databases
- You have followed citation chains for the top-cited papers

## Rules
- **Quality over quantity** — 30 peer-reviewed papers beat 100 unvetted ones
- **Precision over recall** — irrelevant papers waste synthesis capacity
- Use different terminology across databases (different fields use different terms)
- Prioritize recent work AND foundational papers
- Do NOT write any analysis or report yet — just search
- When you are done searching, simply respond without calling any tools

## Available Databases (ordered by priority)
**Primary — Peer-reviewed (search these first):**
- **Scopus** [publisher]: 90M+ from Elsevier, Springer, Wiley, IEEE, ASCE, ACM — best for engineering
- **IEEE Xplore** [publisher]: 6M+ IEEE/IET peer-reviewed journals and conferences
- **PubMed** [index]: 36M+ biomedical and life sciences (use when topic is biomedical)

**Secondary — Broad coverage + preprints (search next):**
- **CrossRef** [publisher]: 150M+ DOI records from all major publishers
- **OpenAlex** [open_access]: 250M+ works, open metadata
- **arXiv** [preprint]: Preprints — may have cutting-edge work not yet published. Use specific queries.

**Citation chains (use after finding good papers):**
- **Semantic Scholar** [index]: 200M+ papers — use get_citations to follow references and citing papers

**Supplementary:**
- **CORE** [open_access]: 300M+ open access — only if other sources miss something
"""


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

_CATEGORY_SYNTHESIS_PROMPT = """\
You are a research analyst writing one section of a literature review on: "{query}"

This section covers the category: **{category}** ({count} papers)

## Papers in this category
{corpus}

## Write this section with:

**What this group does:** 1-2 sentences on the shared approach/theme.
**Key methods:** Specific methods/techniques across papers.
**Main findings:** What do papers collectively show? Agreements? Disagreements?
**Limitations:** Common weaknesses.

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
(Include EVERY paper listed above in the table)

Rules:
- Be direct. No filler. No "In recent years..."
- Include ALL papers from this category in the table
- Do NOT invent papers — only use what's listed above
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

_SYNTHESIS_PROMPT = """\
You are a research analyst. Below is a corpus of {count} papers found across {db_count} \
academic databases on the topic: "{query}"

Your job: **categorize these papers and synthesize findings across categories.** \
Not a story. Not a history lesson. A structured analysis.

## The Paper Corpus

{corpus}

## Output Format

### {query}

#### Coverage
One line: how many papers, which databases, what year range.

#### Categories

For each category you identify (typically 3-6 categories):

##### Category Name (N papers)
**What this group does:** 1-2 sentences describing the shared approach/theme.
**Key methods:** List the specific methods/techniques used across papers in this group.
**Main findings:** What do papers in this group collectively show? Where do they agree? Disagree?
**Limitations:** What are the common weaknesses across this group?

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
| [N] | Author et al. | Year | Approach | Result | Count |

(List ALL papers in this category in the table, not just the top ones)

#### Cross-Category Patterns
What patterns emerge across categories? Which categories are converging? \
What contradictions exist between groups? Which papers bridge multiple categories?

#### Gaps & Opportunities
Be specific. Name concrete research questions that nobody has addressed. \
Point to specific combinations of methods/domains that haven't been tried.

#### Open Access Papers
List papers with free full-text versions available (if any were found).

## Rules
- EVERY paper in the corpus must appear in at least one category table
- Categorize by approach/theme, NOT by database source
- Synthesize ACROSS papers — don't summarize each paper individually
- Be direct. No filler. No "In recent years..." No hedging.
- If papers contradict each other, say so explicitly
- Do NOT invent papers that aren't in the corpus above
- Do NOT write a References section — it will be appended automatically
"""


# Token budget for search phase context (leave room for tool schemas + response)
_MAX_SEARCH_TOKENS = 80_000


def _compact_messages(messages: list[dict], token_estimate_fn) -> list[dict]:
    """Token-aware context compression (Claude Code autoCompact pattern).

    Instead of counting messages, estimates token usage and compresses
    old tool results when approaching the context limit. Preserves
    paper identifiers so the model knows what it already found.
    """
    estimated = token_estimate_fn(messages)
    if estimated < _MAX_SEARCH_TOKENS:
        return messages

    # Find tool messages, compress oldest ones
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return messages

    # Compress from oldest until we're under budget
    compacted = list(messages)
    for idx in tool_indices:
        if token_estimate_fn(compacted) < _MAX_SEARCH_TOKENS:
            break
        content = compacted[idx]["content"]
        if len(content) > 300:
            # Keep the summary line + paper count, drop details
            lines = content.split("\n")
            summary = lines[0] if lines else "Search results"
            compacted[idx] = {**compacted[idx], "content": f"{summary}\n[Results compressed — papers tracked separately]"}

    return compacted


def _build_paper_corpus(papers: dict[str, Paper]) -> str:
    """Build a structured paper corpus for the synthesis prompt."""
    if not papers:
        return "(No papers found)"

    # Sort by citation count (highest first), then by year (newest first)
    sorted_papers = sorted(
        papers.values(),
        key=lambda p: (-(p.citation_count or 0), -(p.year or 0)),
    )

    lines = []
    for i, p in enumerate(sorted_papers, 1):
        entry = f"[{i}] {p.title}"
        parts = []
        if p.authors:
            author_str = p.authors[0]
            if len(p.authors) > 1:
                author_str += " et al."
            parts.append(author_str)
        if p.year:
            parts.append(str(p.year))
        if p.journal:
            parts.append(p.journal)
        if p.citation_count is not None:
            parts.append(f"{p.citation_count} citations")
        if p.doi:
            parts.append(f"DOI: {p.doi}")
        if p.open_access_url:
            parts.append(f"OA: {p.open_access_url}")

        entry += f"\n   {' | '.join(parts)}"

        if p.abstract:
            abstract = p.abstract[:250]
            if len(p.abstract) > 250:
                cut = abstract.rfind(". ")
                abstract = abstract[:cut + 1] if cut > 150 else abstract + "..."
            entry += f"\n   Abstract: {abstract}"

        # Track which databases found this paper
        if p.source:
            entry += f"\n   Found in: {p.source}"

        lines.append(entry)

    return "\n\n".join(lines)


_EXTRACT_TERMS_PROMPT = """\
/no_think
Given this research query, extract the key search term groups.

Query: "{query}"

Return exactly two groups:
METHOD: comma-separated list of method/technique terms and their synonyms
DOMAIN: comma-separated list of domain/application terms and their synonyms

Example for "transformer models in structural health monitoring":
METHOD: transformer, attention mechanism, vision transformer, self-attention, ViT, encoder-decoder
DOMAIN: structural health monitoring, SHM, damage detection, crack detection, structural integrity, structural assessment

Example for "CRISPR gene editing in cancer therapy":
METHOD: CRISPR, gene editing, Cas9, genome editing, CRISPR-Cas
DOMAIN: cancer therapy, oncology, tumor treatment, cancer treatment, anti-cancer

Rules:
- Include common synonyms and abbreviations for each group
- METHOD = the technique/approach being studied
- DOMAIN = the field/application area
- Return ONLY the two lines, nothing else
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
        self.registry = build_tool_registry(config)
        self.papers: dict[str, Paper] = {}
        self.console = Console()
        self._databases_used: set[str] = set()
        self._tool_call_count = 0
        self._output_folder: str = ""
        self._rejected_count = 0
        self._method_terms: list[str] = []
        self._domain_terms: list[str] = []

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

        # Combine original query + answers into enhanced query
        enhanced = f"{query}\n\nAdditional context from the researcher:\n"
        enhanced += "\n".join(f"- {a}" for a in answers)
        self.console.print(f"\n[green]Enhanced query ready.[/green]")
        return enhanced

    def _extract_search_terms(self, query: str) -> None:
        """Extract METHOD and DOMAIN term groups from the query via LLM."""
        self.console.print("[dim]Extracting search terms...[/dim]")
        try:
            response = self.llm.chat([
                {"role": "system", "content": _EXTRACT_TERMS_PROMPT.format(query=query)},
                {"role": "user", "content": "Extract the terms now."},
            ])
            text = response.content or ""
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("METHOD"):
                    raw = line.split(":", 1)[1] if ":" in line else ""
                    self._method_terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
                elif line.upper().startswith("DOMAIN"):
                    raw = line.split(":", 1)[1] if ":" in line else ""
                    self._domain_terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        except Exception as e:
            logger.warning("Term extraction failed: %s", e)

        if self._method_terms and self._domain_terms:
            self.console.print(f"  [green]Method terms:[/green] {', '.join(self._method_terms)}")
            self.console.print(f"  [green]Domain terms:[/green] {', '.join(self._domain_terms)}")
        else:
            self.console.print("  [yellow]Could not extract terms — relevance filter disabled[/yellow]")

    def research(self, query: str) -> str:
        self.console.print(Panel(
            f"[bold]{query}[/bold]\n[dim]breadth={self.config.breadth}  depth={self.config.depth}  max_iter={self.config.max_iterations}[/dim]",
            title="Deep Researcher",
            border_style="blue",
        ))

        # Create output folder early for checkpoints
        self._output_folder = get_output_folder(query, self.config.output_dir)

        # Extract search terms for relevance filtering
        self._extract_search_terms(query)

        # === PHASE 1 & 2: Search (gather papers) ===
        self.console.print("\n[bold blue]Phase 1-2: Searching databases...[/bold blue]")
        self._search_phase(query)

        # === PHASE 3: Synthesize (categorize + analyze) ===
        self.console.print(f"\n[bold blue]Phase 3: Synthesizing {len(self.papers)} papers...[/bold blue]")
        report = self._synthesis_phase(query)

        self._print_summary()
        self._save(query, report)
        return report

    def _search_phase(self, query: str) -> None:
        """Run the agentic search loop to collect papers."""
        search_prompt = _build_search_prompt(self.config)
        messages: list[dict] = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": f"Find all relevant papers on:\n\n{query}"},
        ]
        tool_schemas = self.registry.schemas()
        compact_failures = 0  # Circuit breaker (Claude Code pattern)
        _MAX_COMPACT_FAILURES = 3

        for iteration in range(1, self.config.max_iterations + 1):
            self.console.print(f"\n[dim]--- Search {iteration}/{self.config.max_iterations} | {len(self.papers)} papers | {len(self._databases_used)} databases ---[/dim]")

            messages = _compact_messages(messages, LLMClient.estimate_tokens)

            try:
                response = self.llm.chat(messages, tools=tool_schemas)
                compact_failures = 0  # Reset on success
            except Exception as e:
                self.console.print(f"[red]LLM error: {e}[/red]")
                # If context is likely too long, try compacting and retrying once
                if "too long" in str(e).lower() or "context" in str(e).lower():
                    compact_failures += 1
                    if compact_failures >= _MAX_COMPACT_FAILURES:
                        self.console.print("[red]Context compression failed repeatedly. Proceeding to synthesis.[/red]")
                        break
                    self.console.print("[yellow]Attempting context compression recovery...[/yellow]")
                    messages = _compact_messages(messages, lambda m: _MAX_SEARCH_TOKENS + 1)  # Force compress
                    try:
                        response = self.llm.chat(messages, tools=tool_schemas)
                    except Exception:
                        self.console.print("[red]Recovery failed. Proceeding to synthesis with papers found so far.[/red]")
                        break
                else:
                    break

            # No tool calls = LLM says it's done searching
            if not response.tool_calls:
                content = (response.content or "").strip()
                if content:
                    self.console.print(f"  [dim]{_truncate(content, 100)}[/dim]")
                break

            messages.append(_message_to_dict(response))

            # Execute tool calls concurrently
            tc_list = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.tool_calls
            ]

            if len(tc_list) > 1:
                self.console.print(f"  [cyan]Executing {len(tc_list)} tools (partitioned)...[/cyan]")
                results = self.registry.execute_partitioned(tc_list)
            else:
                results = []
                for tc in tc_list:
                    result = self.registry.execute(tc["name"], tc["arguments"])
                    results.append((tc["id"], result))

            papers_before = len(self.papers)
            for call_id, result in results:
                tc_info = next(tc for tc in tc_list if tc["id"] == call_id)
                self._tool_call_count += 1
                self.console.print(f"  [cyan]{tc_info['name']}[/cyan] -> {_truncate(result.text, 100)}")

                for paper in result.papers:
                    self._track_paper(paper, query)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.text,
                })
            new_papers = len(self.papers) - papers_before

            new_label = f" (+{new_papers} new)" if new_papers else " (no new)"
            rejected_label = f", {self._rejected_count} irrelevant filtered" if self._rejected_count else ""
            self.console.print(f"  [yellow]Total: {len(self.papers)} unique papers{new_label} from {len(self._databases_used)} databases{rejected_label}[/yellow]")

            # Inject reflection prompt (adapts strategy: broad → focused → gap-filling)
            progress = iteration / self.config.max_iterations
            if progress < 0.35:
                phase_hint = (
                    "You're in the BROAD EXPLORATION phase. Cast a wide net — use different databases, "
                    "vary your terminology, and try different search angles."
                )
            elif progress < 0.7:
                phase_hint = (
                    "You're in the FOCUSED DRILLING phase. Follow citation chains for your most-cited papers. "
                    "Narrow your queries to specific methods, sub-topics, or time periods you haven't covered."
                )
            else:
                phase_hint = (
                    "You're in the GAP-FILLING phase. Look at what's missing — specific methods, recent work, "
                    "foundational papers, or sub-topics with few results. Fill those specific gaps."
                )

            rejected_note = ""
            if self._rejected_count:
                rejected_note = (
                    f" ({self._rejected_count} irrelevant papers were automatically filtered out — "
                    f"use more specific queries to avoid this.)"
                )
            reflection = (
                f"[SEARCH STATUS for \"{query}\"] "
                f"{len(self.papers)} relevant papers from {len(self._databases_used)} databases "
                f"({', '.join(sorted(self._databases_used))}).{rejected_note} "
                f"{phase_hint} "
                f"IMPORTANT: Use specific multi-word queries combining method + domain. "
                f"Continue searching for papers on \"{query}\" — call search tools to fill gaps, "
                f"or stop (no tool calls) if you have good coverage."
            )
            messages.append({"role": "user", "content": reflection})

            # Checkpoint: save papers collected so far
            if self.papers and self._output_folder:
                try:
                    save_checkpoint(self.papers, self._output_folder)
                except Exception:
                    pass  # Non-critical — don't break the search loop
        else:
            # Loop completed without LLM stopping (hit max iterations)
            self.console.print(f"\n[yellow]Reached iteration limit ({self.config.max_iterations}). Proceeding to synthesis with {len(self.papers)} papers.[/yellow]")

    def _synthesis_phase(self, query: str) -> str:
        """Multi-step synthesis (STORM-inspired + Claude Code token budgeting).

        Step 1: Categorize papers (lightweight — one-line per paper)
        Step 2: Synthesize per category (token-budgeted paper injection)
        Step 3: Cross-category analysis (works on summaries, not raw papers)
        Step 4: Assemble report programmatically
        """
        if not self.papers:
            return "No papers were found for this query."

        # Cap corpus — sort by quality tier then citations
        _MAX_SYNTHESIS_PAPERS = 200
        _TIER1_SOURCES = {"scopus", "ieee", "pubmed"}
        all_papers = self.papers

        def _paper_sort_key(p: Paper) -> tuple:
            # Tier 1 sources first, then by citations, then by year
            sources = {s.strip() for s in p.source.split(",")}
            has_tier1 = 0 if sources & _TIER1_SOURCES else 1
            return (has_tier1, -(p.citation_count or 0), -(p.year or 0))

        if len(all_papers) > _MAX_SYNTHESIS_PAPERS:
            sorted_all = sorted(all_papers.values(), key=_paper_sort_key)
            synthesis_papers = sorted_all[:_MAX_SYNTHESIS_PAPERS]
            self.console.print(
                f"  [yellow]Corpus capped: synthesizing top {_MAX_SYNTHESIS_PAPERS} of "
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
            cat_papers = [synthesis_papers[i] for i in paper_indices if i < len(synthesis_papers)]
            if not cat_papers:
                continue
            self.console.print(f"    [cyan]{cat_name}[/cyan] ({len(cat_papers)} papers)...")
            # Circuit breaker: skip category on failure, with hard 5 min timeout
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._synthesize_category, query, cat_name, cat_papers)
                    section = future.result(timeout=300)  # 5 min — thinking mode needs time
                category_sections.append((cat_name, section))
            except concurrent.futures.TimeoutError:
                self.console.print(f"    [red]Timed out after 5 min — skipping[/red]")
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
        """Step 1: Assign papers to categories in batches.

        Processes papers in groups of 50 to stay within local model limits,
        then merges category assignments across batches.
        """
        _BATCH_SIZE = 20
        all_categories: dict[str, list[int]] = {}

        for batch_start in range(0, len(papers), _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, len(papers))
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
                # Use no-think mode for categorization (mechanical task, no reasoning needed)
                content = self.llm.chat_no_think([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Categorize these papers now."},
                ])
                batch_cats = _parse_categories(content, len(papers))
                # Merge into overall categories
                for cat_name, indices in batch_cats.items():
                    if cat_name in all_categories:
                        all_categories[cat_name].extend(indices)
                    else:
                        all_categories[cat_name] = indices
            except Exception as e:
                logger.warning("Categorization batch %d-%d failed: %s", batch_start, batch_end, e)
                continue  # Skip this batch, continue with others

        return all_categories

    def _synthesize_category(self, query: str, cat_name: str, papers: list[Paper]) -> str:
        """Step 2: Synthesize one category with token-budgeted paper injection.

        Uses Claude Code's progressive compression:
        - Level 1 (top papers): Full entry with abstract
        - Level 2 (middle): One-line entry
        - Level 3 (tail): Counted
        """
        corpus = _build_tiered_corpus(papers, token_budget=15000)

        prompt = _CATEGORY_SYNTHESIS_PROMPT.format(
            query=query,
            category=cat_name,
            count=len(papers),
            corpus=corpus,
        )

        response = self.llm.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Write the synthesis for: {cat_name}"},
        ])
        return response.content or ""

    def _cross_category_analysis(self, query: str, sections: list[tuple[str, str]]) -> str:
        """Step 3: Analyze patterns across categories."""
        # Build category summaries (first 500 chars of each section)
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
            response = self.llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the cross-category analysis."},
            ])
            return response.content or ""
        except Exception as e:
            return f"Cross-category analysis unavailable: {e}"

    def _assemble_report(
        self, query: str, papers: list[Paper],
        categories: dict[str, list[int]],
        sections: list[tuple[str, str]], cross_section: str,
    ) -> str:
        """Step 4: Assemble the final report programmatically."""
        years = [p.year for p in papers if p.year]
        yr_range = f"{min(years)}-{max(years)}" if years else "unknown"
        total = len(self.papers)  # All papers, not just synthesis subset
        db_str = ", ".join(sorted(self._databases_used))

        parts = [
            f"### {query}\n",
            f"#### Coverage",
            f"{total} papers found across {len(self._databases_used)} databases ({db_str}), "
            f"years {yr_range}. Top {len(papers)} by citation count synthesized below.\n",
            "#### Categories\n",
        ]

        # Add each category section
        for cat_name, content in sections:
            cat_indices = categories.get(cat_name, [])
            parts.append(f"##### {cat_name} ({len(cat_indices)} papers)\n")
            parts.append(content)
            parts.append("")

        # Cross-category analysis
        parts.append(cross_section)
        parts.append("")

        # References (generated programmatically — never relies on LLM for this)
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
        # Use top 20 papers with tiered corpus to keep within local model limits
        top_papers = papers[:20]
        corpus = _build_tiered_corpus(top_papers, token_budget=8000)
        prompt = (
            f"Write a brief literature review on \"{query}\" based on these {len(top_papers)} papers. "
            f"Categorize by theme, include a table per category.\n\n{corpus}"
        )
        try:
            response = self.llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the review."},
            ])
            return response.content or ""
        except Exception as e:
            return f"Synthesis failed: {e}"

    def _track_paper(self, paper: Paper, query: str = "") -> None:
        # Relevance filter: reject papers that don't match both METHOD and DOMAIN terms
        if query and not _is_relevant(paper, query, self._method_terms, self._domain_terms):
            self._rejected_count += 1
            return

        key = paper.unique_key
        if key in self.papers:
            self.papers[key].merge(paper)
        else:
            self.papers[key] = paper
        for src in paper.source.split(","):
            src = src.strip()
            if src:
                self._databases_used.add(src)

    def _print_summary(self) -> None:
        self.console.print(f"\n[green]Research complete.[/green]")
        table = Table(title="Research Summary", show_header=False, border_style="green")
        table.add_row("Papers found", str(len(self.papers)))
        table.add_row("Databases searched", ", ".join(sorted(self._databases_used)))
        table.add_row("Tool calls made", str(self._tool_call_count))

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


def _message_to_dict(msg) -> dict:
    d: dict = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _parse_categories(text: str, paper_count: int) -> dict[str, list[int]]:
    """Parse LLM category output into {name: [0-based indices]}."""
    import re as _re
    categories: dict[str, list[int]] = {}
    current_cat = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match "CATEGORY: Name" or "Category: Name"
        cat_match = _re.match(r"(?:CATEGORY|Category)\s*:\s*(.+)", line, _re.IGNORECASE)
        if cat_match:
            current_cat = cat_match.group(1).strip()
            continue
        # Match "PAPERS: 1, 5, 12" or "Papers: 1, 5, 12"
        papers_match = _re.match(r"(?:PAPERS|Papers)\s*:\s*(.+)", line, _re.IGNORECASE)
        if papers_match and current_cat:
            nums = _re.findall(r"\d+", papers_match.group(1))
            indices = [int(n) - 1 for n in nums if 0 < int(n) <= paper_count]  # 1-based → 0-based
            if indices:
                categories[current_cat] = indices
            current_cat = None

    # Validate: every paper should be assigned. If >30% unassigned, categorization may have failed.
    assigned = set()
    for indices in categories.values():
        assigned.update(indices)
    if len(assigned) < paper_count * 0.3:
        logger.warning("Categorization covered only %d/%d papers", len(assigned), paper_count)
        # Still return what we have — better than nothing

    return categories


def _build_tiered_corpus(papers: list, token_budget: int = 15000) -> str:
    """Build a token-budgeted corpus with progressive compression (Claude Code pattern).

    Level 1 (top papers by citations): Full entry with abstract (~250 tokens each)
    Level 2 (middle): One-line entry (~30 tokens each)
    Level 3 (tail): Counted as a group
    """
    _CHARS_PER_TOKEN = 4

    sorted_papers = sorted(papers, key=lambda p: (-(p.citation_count or 0), -(p.year or 0)))
    lines = []
    tokens_used = 0
    level1_budget = int(token_budget * 0.6)
    level2_budget = int(token_budget * 0.9)
    full_count = 0

    for i, p in enumerate(sorted_papers, 1):
        # Level 1: full entry
        full_entry = _paper_full_entry(i, p)
        full_tokens = len(full_entry) // _CHARS_PER_TOKEN
        if tokens_used + full_tokens < level1_budget:
            lines.append(full_entry)
            tokens_used += full_tokens
            full_count += 1
            continue

        # Level 2: one-line entry
        short_entry = _paper_short_entry(i, p)
        short_tokens = len(short_entry) // _CHARS_PER_TOKEN
        if tokens_used + short_tokens < level2_budget:
            lines.append(short_entry)
            tokens_used += short_tokens
            continue

        # Level 3: count the rest
        remaining = len(sorted_papers) - i + 1
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
        abstract = p.abstract[:250]
        if len(p.abstract) > 250:
            cut = abstract.rfind(". ")
            abstract = abstract[:cut + 1] if cut > 150 else abstract + "..."
        parts.append(f"   Abstract: {abstract}")
    return "\n".join(parts)


def _paper_short_entry(idx: int, p) -> str:
    """One-line compressed entry for tier-2 papers."""
    author = p.authors[0].split()[-1] if p.authors else "Unknown"
    year = p.year or "n.d."
    cites = f", {p.citation_count} cites" if p.citation_count else ""
    return f"[{idx}] {p.title} ({author}, {year}{cites})"


def _is_relevant(paper, query: str, method_terms: list[str] = None, domain_terms: list[str] = None) -> bool:
    """Relevance check using LLM-extracted term groups.

    If method_terms and domain_terms are provided (extracted at research start),
    a paper must match at least one term from EACH group.
    If not provided, falls back to basic phrase matching.
    """
    paper_text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    if not paper_text.strip():
        return True  # Keep papers with no text (can't judge)

    # Primary: use extracted term groups (METHOD + DOMAIN must both match)
    if method_terms and domain_terms:
        has_method = any(term in paper_text for term in method_terms)
        has_domain = any(term in paper_text for term in domain_terms)
        return has_method and has_domain

    # Fallback: basic phrase matching from query words
    import re as _re
    query_words = _re.findall(r"[a-z]{3,}", query.lower())
    for n in (3, 2):
        for i in range(len(query_words) - n + 1):
            phrase = " ".join(query_words[i:i + n])
            if phrase in paper_text:
                return True
    return False


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s
