<h1 align="center">
  <img src="assets/logo.svg" width="64" alt="Deep Researcher" align="absmiddle">&nbsp;&nbsp;Deep Researcher
</h1>

<p align="center">
  <strong>An academic research assistant that searches Google Scholar, enriches via OpenAlex, and writes structured literature reviews — all running locally.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/dependencies-4-brightgreen.svg?style=flat-square" alt="Dependencies: 4">
  <img src="https://img.shields.io/badge/version-0.4.0-blue.svg?style=flat-square" alt="Version: 0.4.0">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#sample-output">Sample Output</a> &middot;
  <a href="#configuration">Configuration</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

Most "AI research tools" do web search and write you a story with a few citations. Deep Researcher does what an actual researcher does: **searches Google Scholar for semantically-ranked academic papers, enriches them with full metadata from OpenAlex, and builds a structured literature review with proper citations.**

**4 dependencies. No LangChain. Runs 100% locally.**

> [!NOTE]
> Deep Researcher searches **Google Scholar** — the same academic search engine researchers use. This means semantic ranking (it understands "transformer" means the neural network, not the electrical device), academic-only results (no blog posts), and access to papers across all major publishers. Metadata is enriched via OpenAlex and CrossRef for full abstracts, DOIs, and journal information.

---

## Why This Exists

Existing tools like GPT Researcher, STORM, Gemini Deep Research, and ChatGPT all rely on **web search**. They write a narrative essay with a few citations. That's useful for a quick overview, but not for a literature review.

We tried the alternative — searching academic database APIs directly (Scopus, IEEE, PubMed, arXiv). The problem: these APIs use **loose keyword matching**. Search "transformer models for structural health monitoring" and you get papers about electrical power transformers, mental health monitoring, and hundreds of irrelevant results. We built relevance filters, term extraction, and keyword matching to compensate — and it kept breaking on semantic ambiguity.

**The insight: Google Scholar already solves this.** It has decades of semantic ranking built in. A single query returns 100 relevant, academically-ranked papers with near-zero noise. No keyword filters needed. No term extraction. No relevance hacks.

Deep Researcher combines **Google Scholar's semantic search** with **OpenAlex's structured metadata** (full abstracts, DOIs, journal names, citation counts) and a **local LLM for multi-step synthesis**. The result:

- **100 relevant papers** in ~30 seconds (vs 5+ minutes with database APIs and hundreds of rejected results)
- **Full metadata** — DOIs, abstracts, journal names, citation counts, open access URLs
- **Concurrent enrichment** — 8 parallel workers enrich papers via OpenAlex/CrossRef
- **Structured synthesis** — categorized by theme, with per-category analysis and cross-category patterns
- **Globally consistent citations** — `[N]` in the text matches `[N]` in the reference list
- **No hallucinated sources** — every paper comes from Google Scholar, every claim cites a real abstract
- **Principled tool-based architecture** — modeled after [Claude Code](https://github.com/anthropics/claude-code)'s agentic framework
- **BibTeX + CSV output** ready for LaTeX/Overleaf or Excel
- **Runs 100% locally** with Ollama — your queries never leave your machine

Built for **academics, grad students, and researchers** who need comprehensive literature reviews, not AI-generated essays.

---

## Quick Start

### Install

```bash
git clone https://github.com/jackswl/deep-researcher.git
cd deep-researcher
pip install -e .
```

### Run with Ollama (local, free, private)

```bash
ollama pull qwen3.5:9b
deep-researcher "large language models for automated code compliance in BIM"
```

### Run with any provider (one flag)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
deep-researcher "machine learning for drug discovery" --provider openai

# Groq (fast, free tier available)
export OPENAI_API_KEY="gsk_..."
deep-researcher "CRISPR gene editing" --provider groq

# DeepSeek (affordable)
export OPENAI_API_KEY="sk-..."
deep-researcher "quantum computing algorithms" --provider deepseek

# Anthropic
export OPENAI_API_KEY="sk-ant-..."
deep-researcher "climate modeling" --provider anthropic

# OpenRouter (100+ models, 29 free)
export OPENAI_API_KEY="sk-or-..."
deep-researcher "protein folding" --provider openrouter

# LMStudio (local GUI)
deep-researcher "robotics control" --provider lmstudio
```

<details>
<summary><strong>All 8 supported providers</strong></summary>

| Provider | Flag | Default Model | API Key Required |
|---|---|---|---|
| Ollama | `--provider ollama` | `qwen3.5:9b` | No (local) |
| LMStudio | `--provider lmstudio` | auto-detect | No (local) |
| OpenAI | `--provider openai` | `gpt-5.4-mini` | Yes |
| Anthropic | `--provider anthropic` | `claude-sonnet-4-6` | Yes |
| Groq | `--provider groq` | `qwen/qwen3-32b` | Yes (free tier) |
| DeepSeek | `--provider deepseek` | `deepseek-chat` | Yes |
| OpenRouter | `--provider openrouter` | `claude-sonnet-4-6` | Yes (free models) |
| Together | `--provider together` | `Llama-4-Maverick` | Yes |

</details>

---

## What You Get

Each session produces five files:

```
output/2026-04-02-161823-large-language-models-for-automated-code/
├── report.md        # Categorized literature review with synthesis
├── references.bib   # Deduplicated BibTeX (import into LaTeX/Overleaf)
├── papers.json      # Full metadata for every paper found
├── papers.csv       # Same data as CSV (open in Excel/Sheets)
└── metadata.json    # Research stats: sources, coverage, year range
```

> `papers.json` is saved as a **checkpoint** during search, so even interrupted runs preserve partial results.

---

## Sample Output

Here's what the synthesis looks like. **Not a list of summaries. Structured analysis:**

```markdown
### Large Language Models for Automated Code Compliance in BIM

#### Coverage
100 papers found via Google Scholar, enriched via OpenAlex. Years 2010-2026. 96 with DOIs.

#### Categories

##### Automated Compliance Checking & Reasoning (13 papers)

### What this group does
This group shifts BIM compliance checking from manual review to automated systems
that use LLMs to interpret regulatory texts and validate design models. Researchers
focus on transforming building codes into machine-readable formats, leveraging
knowledge graphs to represent compliance rules, and employing instruction-tuned
LLMs to reason over design schemes and detect violations.

### Key methods
Regulatory information transformation with ruleset expansion [18]. Knowledge Graph
+ LLM fusion for intelligent evaluation [19]. Instruction-tuning for building
code analysis (Qwen-AEC) [52]. Multi-agent BIM-to-Text paradigm [80].

### Main findings
Automated compliance checking significantly addresses manual inefficiency.
Integration of LLMs and knowledge graphs enables handling of complex design
specifications and automated identification of design defects [100].

### Limitations & gaps (your analysis)
A common limitation is the heavy reliance on transforming unstructured regulatory
texts into structured formats. Most frameworks focus on detection but lack
protocols for automatic repair of violations without human intervention.

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
| [18] | Regulatory information transformation... | 2022 | Ruleset expansion | Promising upgrade to manual compliance | 36 |
| [19] | Intelligent checking method via KG+LLM | 2024 | KG + LLM fusion | Professional evaluation ensures safety | 35 |
| ... | ... | ... | ... | ... | ... |

#### Cross-Category Patterns
...

#### Gaps & Opportunities
...

#### References
[1] Author et al. (2021). Title. *Journal*. DOI: 10.xxxx
[2] Author et al. (2024). Title. *Journal*. DOI: 10.xxxx
...
```

> Every `[N]` in the text matches `[N]` in the reference list. Limitations are the model's own synthesis — not fake-attributed to papers. Findings come strictly from abstracts.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  YOUR RESEARCH QUESTION                 │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Phase 1: GOOGLE SCHOLAR    │
          │                             │
          │  Your query → Google        │
          │  Scholar → up to 100        │  Semantic ranking:
          │  academically-ranked        │  "transformer" means
          │  papers with titles,        │  the neural network,
          │  authors, citations,        │  not the electrical
          │  and abstract snippets      │  device.
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Phase 2: ENRICHMENT        │
          │  (concurrent — 8 workers)   │
          │                             │
          │  For each paper (parallel): │
          │  1. OpenAlex title search   │  Full abstracts,
          │     → full abstract, DOI,   │  DOIs, journal
          │       journal, citations    │  names, open
          │  2. CrossRef fallback       │  access URLs
          │     → DOI → OpenAlex        │
          │                             │
          │  Title similarity check     │
          │  prevents wrong-paper       │
          │  enrichment                 │
          └──────────────┬──────────────┘
                         │
              Enriched paper corpus
              (100 papers, ~60-70% full abstracts)
                         │
          ┌──────────────▼──────────────┐
          │  Phase 3: SYNTHESIS         │
          │  (each step is a Tool)      │
          │                             │
          │  Step 1: CategorizeTool     │
          │    → 4-6 themes (batched)   │
          │  Step 2: SynthesisTool      │
          │    → per-category analysis  │
          │    (no-think mode, 2.6x     │
          │     faster on local models) │
          │  Step 3: CrossAnalysisTool  │
          │    → patterns & gaps        │
          │  Step 4: Assemble report    │
          │    with programmatic refs   │
          │                             │
          │  Fallback: FallbackTool     │
          │    if categorization fails  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │ report.md + references.bib  │
          │ papers.json/csv + metadata  │
          └─────────────────────────────┘
```

### Date Range Filtering: `--start-year` and `--end-year`

Focus your research on a specific time period:

```bash
# Only papers from 2020 onward
deep-researcher "federated learning" --start-year 2020

# Specific window
deep-researcher "attention mechanisms" --start-year 2017 --end-year 2023
```

### Interactive Mode: `--interactive`

Ask the LLM to generate clarifying questions before starting the search:

```bash
deep-researcher "machine learning in healthcare" --interactive
```

```
Generating clarifying questions...

1. Are you focused on diagnostics, treatment planning, or administrative applications?
2. Any specific medical domain (cardiology, oncology, radiology, etc.)?
3. Are you interested in deep learning specifically, or ML methods broadly?

Answer each question (press Enter to skip):

  1. > diagnostics and radiology
  2. > oncology
  3. >

Enhanced query ready.
```

---

## How It Compares

| | GPT Researcher | STORM | Gemini/ChatGPT | **Deep Researcher** |
|---|---|---|---|---|
| Search method | Web (Tavily) | Web | Web | **Google Scholar** |
| Academic-only results | No | No | No | **Yes** |
| Semantic search ranking | Depends on Tavily | No | Yes | **Yes (Google Scholar)** |
| Full metadata enrichment | No | No | No | **Yes (OpenAlex)** |
| Paper deduplication | No | No | No | **Yes** |
| BibTeX output | No | No | No | **Yes** |
| CSV export | No | No | No | **Yes** |
| Categorized synthesis | No | No | No | **Yes** |
| Consistent [N] citations | No | No | No | **Yes** |
| Date range filtering | No | No | No | **Yes** |
| Interactive query refinement | No | No | No | **Yes** |
| Runs fully local | Partial | Partial | No | **Yes** |
| Paid APIs required | Yes (Tavily) | Yes | Yes (subscription) | **No** |
| Dependencies | ~50+ | ~30+ | N/A | **4** |

---

## Usage

```
deep-researcher "your research question" [options]

Options:
  --provider PROVIDER    LLM provider (ollama, openai, groq, etc.)
  --model MODEL          LLM model name
  --base-url URL         OpenAI-compatible API URL
  --api-key KEY          API key
  --start-year YEAR      Filter papers published on or after this year
  --end-year YEAR        Filter papers published on or before this year
  --interactive          Ask clarifying questions before researching
  --output DIR           Output directory (default: ./output)
  --email EMAIL          Email for polite API access to OpenAlex/CrossRef
  --verbose              Enable debug logging
  --version              Show version
```

### Examples

```bash
# Basic usage
deep-researcher "transformer models in structural health monitoring"

# Focus on recent work only
deep-researcher "large language models" --start-year 2023

# Narrow to a specific time window
deep-researcher "CRISPR gene editing" --start-year 2020 --end-year 2024

# Interactive mode: refine your question first
deep-researcher "machine learning in healthcare" --interactive

# Use a cloud provider for faster synthesis
deep-researcher "quantum computing" --provider groq --start-year 2022
```

---

## Configuration

### Config File (recommended)

Create `~/.deep-researcher/config.json`:

```json
{
  "model": "qwen3.5:9b",
  "base_url": "http://localhost:11434/v1",
  "api_key": "ollama",
  "email": "you@university.edu",
  "output_dir": "~/research/output",
  "start_year": 2020,
  "end_year": 2026
}
```

Priority: CLI args > environment variables > config file > defaults.

<details>
<summary><strong>Environment variables</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `qwen3.5:9b` | LLM model name |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | API endpoint |
| `OPENAI_API_KEY` | `ollama` | API key |
| `DEEP_RESEARCH_EMAIL` | - | Email for polite API pool |
| `DEEP_RESEARCH_START_YEAR` | - | Filter: papers from this year onward |
| `DEEP_RESEARCH_END_YEAR` | - | Filter: papers up to this year |

</details>

---

## Models & Compatibility

> [!IMPORTANT]
> Deep Researcher uses an LLM only for **synthesis** (categorization and writing). Search is handled by Google Scholar — no LLM involved. This means even smaller models work well, since they only need to write, not reason about what to search.

<details>
<summary><strong>Local models (Ollama / LMStudio)</strong></summary>

| Model | Ollama ID | Notes |
|---|---|---|
| Qwen 3.5 9B | `qwen3.5:9b` | **Default.** Good quality/size ratio |
| Qwen 3.5 27B | `qwen3.5:27b` | Higher quality, needs 16GB+ VRAM |
| Qwen3-Coder 30B | `qwen3-coder:30b` | Strong writing |
| Llama 4 Scout | `llama4:scout` | 10M context |
| DeepSeek V3.2 | `deepseek-v3.2` | Strong reasoning |

> All synthesis calls use **no-think mode** (suppresses reasoning tokens), making local models 2.6x faster with equal quality for writing tasks.

</details>

<details>
<summary><strong>Cloud models</strong></summary>

| Provider | Model | Notes |
|---|---|---|
| OpenAI | `gpt-5.4-mini` | Fast, great quality |
| Anthropic | `claude-sonnet-4-6` | Excellent synthesis |
| Groq | `qwen/qwen3-32b` | Ultra-fast, free tier |
| DeepSeek | `deepseek-chat` | Affordable |
| OpenRouter | Browse 40+ models | 29 free models |

</details>

---

## Architecture

Deep Researcher v0.4 was built by studying **Claude Code's source code** (Anthropic's CLI for Claude) and applying its 7 agentic framework principles:

| Principle | Claude Code | Deep Researcher |
|---|---|---|
| Tools as unit of action | Every interaction is a `Tool` with `call()`, `validateInput()` | 7 pipeline tools with `execute()`, `validate_input()`, `safe_execute()` |
| Orchestration separate from execution | `query.ts` only dispatches tools | `orchestrator.py` makes zero raw API calls — only `safe_execute()` |
| Immutable state flow | Messages never mutated, `contextModifier` returns new context | `PipelineState.evolve()` returns new state each phase |
| Layered error recovery | `withRetry` → model fallback → context compact → tool errors | API retry → `safe_execute()` wrapping → per-category skip → fallback synthesis |
| Validation at boundaries | Zod schemas on every tool input | `validate_input()` called on every `safe_execute()` path |
| Declarative concurrency | `isConcurrencySafe()` per tool, `StreamingToolExecutor` | `is_read_only` per tool, `EnrichmentTool` uses `ThreadPoolExecutor` |
| Physical separation | Each tool in its own file, orchestration/execution/permissions separate | Each module has one responsibility, 12 source files |

### Project Structure

```
src/deep_researcher/
  __main__.py          # CLI entry point + provider presets
  orchestrator.py      # Pipeline loop — calls tools only, manages state flow
  prompts.py           # All LLM prompt templates
  parsing.py           # LLM output parsing utilities
  display.py           # Display and output helpers (separated from orchestration)
  llm.py               # OpenAI-compatible client with retry/recovery
  config.py            # Config file + env var + CLI loading with validation
  constants.py         # All tunable thresholds in one place
  errors.py            # Domain-specific error types with structured context
  models.py            # Paper, PipelineState (immutable state flow), ToolResult
  report.py            # Report + BibTeX + JSON + CSV + checkpoint generation
  tools/
    base.py            # Tool base class with validate_input() + safe_execute()
    scholar_search.py  # Google Scholar search (pipeline Phase 1)
    enrichment.py      # OpenAlex + CrossRef enrichment with concurrent HTTP (Phase 2)
    categorize.py      # LLM paper categorization (Phase 3, Step 1)
    synthesize.py      # Per-category LLM synthesis (Phase 3, Step 2)
    cross_analysis.py  # Cross-category LLM analysis (Phase 3, Step 3)
    clarify.py         # Interactive query clarification
    fallback_synthesis.py  # Single-pass fallback when multi-step fails
    arxiv_search.py    # arXiv API (database tool)
    openalex.py        # OpenAlex API (database tool)
    crossref.py        # CrossRef API (database tool)
    pubmed.py          # PubMed E-utilities (database tool)
    scopus.py          # Scopus API (database tool)
    ieee_xplore.py     # IEEE Xplore API (database tool)
    ...
tests/
  test_models.py       # Paper dedup, merge, BibTeX, PipelineState
  test_parsing.py      # Category parsing, corpus building, title matching
  test_pipeline_tools.py  # All 7 pipeline tools with mocked dependencies
  test_orchestrator.py # Pipeline phases, state flow, fallback paths
  test_tool_base.py    # Tool validation, safe_execute, registry
  test_config.py       # Config validation
```

---

## License

MIT

---

<p align="center">
  <sub>Built by studying how the best agentic systems work, then applying those patterns to academic research.</sub>
</p>
