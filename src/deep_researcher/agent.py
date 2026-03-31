from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config import Config
from deep_researcher.llm import LLMClient
from deep_researcher.models import Paper
from deep_researcher.report import save_report
from deep_researcher.tools import build_tool_registry


def _build_system_prompt(config: Config) -> str:
    return f"""\
You are a research analyst that maps academic landscapes. Given a research question, \
you systematically search databases, track how papers connect, and produce insight-driven \
analysis — not textbook summaries.

## How to Research — Three Phases

### Phase 1: DISCOVERY (first ~{config.breadth * 2} tool calls)
- Break the question into {config.breadth} search queries using different terminology
- Search at least 3 databases per query variant
- Aim for 20-40 candidate papers

### Phase 2: DEEP DIVE (next ~{config.depth * 3} tool calls)
- Follow citation chains on the top {config.depth * 3} most-cited papers
- Get details on papers that appear across multiple searches
- Look for survey/review papers — they map the field for you
- Check open access for key papers

### Phase 3: SYNTHESIS (final response — no more tool calls)
- Write the analysis (see format below)
- Stop searching. Write.

## When to Stop Searching
- You have 15-30 relevant papers
- New searches return papers you already found
- You covered 3+ databases and followed citation chains

## Final Report Format

Be direct. No filler. No "In recent years..." introductions. Write like a researcher \
briefing a colleague, not like a textbook.

### [Topic]

#### What's Been Done
2-3 paragraphs mapping the landscape. What approaches exist? What are the main \
camps or schools of thought? Where is consensus, where is debate?

#### Key Methods & Findings

| Paper | Approach | Key Result | Limitation |
|-------|----------|------------|------------|
| Author (Year) [N] | Method used | What they found | What's missing |

(Include 10-15 most important papers in this table)

#### How Papers Connect
Which papers build on each other? What are the intellectual lineages? \
Who disagrees with whom? What triggered shifts in the field? \
Think Connected Papers — show the web of relationships.

#### Gaps & Opportunities
What hasn't been tried? What's the low-hanging fruit? Where would a new \
paper have the most impact? Be specific — name concrete research questions.

#### Open Access
List papers that have free full-text versions available.

#### References
[N] Authors (Year). Title. *Journal*. DOI: xxx

## Writing Rules

DO:
- Be direct and specific
- Show connections between papers
- Highlight contradictions and debates
- Name concrete methods (not "various machine learning techniques")
- Use the findings table for structured comparison
- Identify what's actually missing, not what "could be explored further"

DO NOT:
- Write long introductions explaining what the field is
- Give chronological history lessons ("In 2015, Smith et al...")
- Summarize each paper one-by-one in paragraphs
- Use hedge phrases ("It is worth noting...", "Further research is needed...")
- Pad with generic statements about the importance of the topic
- Hallucinate papers — ONLY cite papers you found via tools

## Available Databases
- **arXiv**: Preprints — CS, physics, math, engineering, biology
- **Semantic Scholar**: 200M+ papers, citation counts, TLDR summaries
- **OpenAlex**: 250M+ works, fully open metadata
- **CrossRef**: 150M+ records from Elsevier, Springer, IEEE, Wiley
- **PubMed**: 36M+ biomedical and life sciences
- **CORE**: 300M+ open access articles
"""


# Maximum tool result messages before compressing old ones
_COMPACT_THRESHOLD = 30


def _compact_messages(messages: list[dict]) -> list[dict]:
    """Compress old tool results to keep context manageable (Claude Code autoCompact pattern)."""
    tool_msgs = [(i, m) for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_msgs) <= _COMPACT_THRESHOLD:
        return messages

    # Keep the system prompt, first user message, and recent messages
    # Truncate old tool results to just their first 200 chars
    cutoff = len(tool_msgs) - _COMPACT_THRESHOLD // 2
    old_indices = {i for i, _ in tool_msgs[:cutoff]}

    compacted = []
    for i, msg in enumerate(messages):
        if i in old_indices:
            content = msg["content"]
            if len(content) > 200:
                # Keep first line (summary) + truncation notice
                first_line = content.split("\n")[0]
                compacted.append({**msg, "content": f"{first_line}\n[Earlier results truncated to save context]"})
            else:
                compacted.append(msg)
        else:
            compacted.append(msg)
    return compacted


class ResearchAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.registry = build_tool_registry(config)
        self.papers: dict[str, Paper] = {}
        self.console = Console()
        self._databases_used: set[str] = set()
        self._tool_call_count = 0

    def research(self, query: str) -> str:
        self.console.print(Panel(
            f"[bold]{query}[/bold]\n[dim]breadth={self.config.breadth}  depth={self.config.depth}  max_iter={self.config.max_iterations}[/dim]",
            title="Deep Researcher",
            border_style="blue",
        ))

        system_prompt = _build_system_prompt(self.config)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research this question and produce an insight-driven literature analysis:\n\n{query}"},
        ]
        tool_schemas = self.registry.schemas()

        for iteration in range(1, self.config.max_iterations + 1):
            self.console.print(f"\n[dim]--- Iteration {iteration}/{self.config.max_iterations} | {len(self.papers)} papers | {len(self._databases_used)} databases ---[/dim]")

            # Context compression — truncate old tool results (Claude Code autoCompact pattern)
            messages = _compact_messages(messages)

            try:
                response = self.llm.chat(messages, tools=tool_schemas)
            except Exception as e:
                self.console.print(f"[red]LLM error: {e}[/red]")
                break

            # No tool calls = LLM is done, final report
            if not response.tool_calls:
                report = response.content or ""
                self._print_summary()
                self._save(query, report)
                return report

            messages.append(_message_to_dict(response))

            # Execute tool calls concurrently (Claude Code pattern)
            tc_list = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.tool_calls
            ]

            if len(tc_list) > 1:
                self.console.print(f"  [cyan]Executing {len(tc_list)} tools concurrently...[/cyan]")
                results = self.registry.execute_concurrent(tc_list)
            else:
                results = []
                for tc in tc_list:
                    result = self.registry.execute(tc["name"], tc["arguments"])
                    results.append((tc["id"], result))

            for call_id, result in results:
                tc_info = next(tc for tc in tc_list if tc["id"] == call_id)
                self._tool_call_count += 1
                self.console.print(f"  [cyan]{tc_info['name']}[/cyan] -> {_truncate(result.text, 100)}")

                for paper in result.papers:
                    self._track_paper(paper)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.text,
                })

            self.console.print(f"  [yellow]Total: {len(self.papers)} unique papers from {len(self._databases_used)} databases[/yellow]")

        # Max iterations — force synthesis
        self.console.print("\n[yellow]Max iterations reached — synthesizing...[/yellow]")
        messages = _compact_messages(messages)
        messages.append({
            "role": "user",
            "content": (
                f"You have found {len(self.papers)} papers across {len(self._databases_used)} databases. "
                "Stop searching. Write your analysis now using the format specified."
            ),
        })
        try:
            response = self.llm.chat(messages)
            report = response.content or ""
        except Exception as e:
            report = f"Error generating final report: {e}"

        self._print_summary()
        self._save(query, report)
        return report

    def _track_paper(self, paper: Paper) -> None:
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
            paths = save_report(query, report, self.papers, self.config.output_dir)
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


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s
