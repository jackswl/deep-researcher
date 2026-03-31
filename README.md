<p align="center">
  <h1 align="center">Deep Researcher</h1>
  <p align="center">
    <strong>An agentic academic research assistant that searches 6 databases and writes literature reviews.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#how-it-works">How It Works</a> &middot;
    <a href="#configuration">Configuration</a> &middot;
    <a href="#how-this-was-built">Origin Story</a>
  </p>
</p>

---

Most "AI research tools" do a web search and summarize. Deep Researcher does what an actual researcher does: **searches multiple academic databases, reads abstracts, follows citation chains, refines search queries based on findings, and only writes up when it has enough material.**

It runs a real agentic loop — the LLM decides what to search next, when to dig deeper, and when to stop. Not a pipeline. Not a single prompt. An autonomous research agent.

**3 dependencies. ~1,500 lines. No LangChain.**

---

## Why This Exists

Existing tools like GPT Researcher and STORM are powerful but rely on **general web search** — great for current events, not for academic research. They miss the databases that matter: Semantic Scholar, OpenAlex, CrossRef, PubMed. They can't follow citation chains. They don't output BibTeX.

Deep Researcher was built for **academics, grad students, and researchers** who need:
- Proper coverage across real academic databases (not just Google)
- Citation chains to find foundational and recent work
- BibTeX output they can actually import into LaTeX/Overleaf
- Open access detection for paywalled papers
- A tool they can run locally with their own models

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
ollama pull llama3.1
deep-researcher "applications of transformer models in structural health monitoring"
```

### Run with any provider (one flag)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
deep-researcher "machine learning for drug discovery" --provider openai

# Groq (fast, free tier)
export OPENAI_API_KEY="gsk_..."
deep-researcher "CRISPR gene editing" --provider groq

# DeepSeek
export OPENAI_API_KEY="sk-..."
deep-researcher "quantum computing algorithms" --provider deepseek

# Anthropic
export OPENAI_API_KEY="sk-ant-..."
deep-researcher "climate modeling" --provider anthropic

# OpenRouter (access 100+ models)
export OPENAI_API_KEY="sk-or-..."
deep-researcher "protein folding" --provider openrouter

# LMStudio (local)
deep-researcher "robotics control" --provider lmstudio

# Together AI
export OPENAI_API_KEY="..."
deep-researcher "NLP transformers" --provider together
```

Supported providers: `ollama`, `lmstudio`, `openai`, `anthropic`, `groq`, `deepseek`, `openrouter`, `together`

---

## What You Get

Each session produces four files in `./output/<timestamp>-<topic>/`:

```
output/2026-03-31-142315-transformer-structural-health/
  report.md        # Structured literature review with thematic analysis
  references.bib   # Deduplicated BibTeX — import directly into LaTeX
  papers.json      # Full metadata for all papers found
  metadata.json    # Research stats: databases, coverage, year range
```

---

## How It Works

```
                        YOUR RESEARCH QUESTION
                                |
                                v
                 +------------------------------+
                 |     Phase 1: DISCOVERY        |
                 |  Break query into subqueries  |
                 |  Search 3-6 databases         |
                 |  Run searches concurrently    |
                 |  Collect 20-40 candidates     |
                 +------------------------------+
                                |
                                v
                 +------------------------------+
                 |     Phase 2: DEEP DIVE        |
                 |  Follow citation chains       |
                 |  Get details on key papers    |
                 |  Check open access (Unpaywall)|
                 |  Find survey/review papers    |
                 +------------------------------+
                                |
                                v
                 +------------------------------+
                 |     Phase 3: SYNTHESIS        |
                 |  Thematic analysis            |
                 |  Chronological development    |
                 |  Research gaps identified     |
                 |  Numbered inline citations    |
                 +------------------------------+
                                |
                                v
              report.md + references.bib + papers.json
```

The LLM drives the entire process. It decides which databases to search, what queries to use, when to follow citations, and when it has enough to synthesize. You control the intensity with two knobs:

```bash
# Quick scan — fast, surface-level
deep-researcher "topic" --breadth 1 --depth 0

# Standard research (default)
deep-researcher "topic"

# Comprehensive — wide search, deep citation chains
deep-researcher "topic" --breadth 5 --depth 4
```

### Academic Databases

| Tool | Source | What It Covers |
|---|---|---|
| `search_arxiv` | arXiv | Preprints: CS, physics, math, engineering, biology |
| `search_semantic_scholar` | Semantic Scholar | 200M+ papers, all fields, citation graphs |
| `search_openalex` | OpenAlex | 250M+ works, fully open metadata |
| `search_crossref` | CrossRef | 150M+ records from Elsevier, Springer, IEEE, Wiley |
| `search_pubmed` | PubMed | 36M+ biomedical and life sciences |
| `search_core` | CORE | 300M+ open access full texts |
| `get_paper_details` | Semantic Scholar | Deep lookup on a specific paper by DOI |
| `get_citations` | Semantic Scholar | Papers that cite this / papers this cites |
| `find_open_access` | Unpaywall | Legal free copies of paywalled papers |

---

## Usage

```
deep-researcher "your research question" [options]

Options:
  --model MODEL          LLM model name (default: llama3.1)
  --base-url URL         OpenAI-compatible API URL
  --api-key KEY          API key
  --breadth N            Search breadth: query variations (1-5, default: 3)
  --depth N              Search depth: citation rounds (0-5, default: 2)
  --max-iterations N     Max agentic loop iterations (default: 20)
  --output DIR           Output directory (default: ./output)
  --email EMAIL          Email for polite API access (recommended)
  --version              Show version
```

---

## Configuration

### Config File (recommended)

Create `~/.deep-researcher/config.json`:

```json
{
  "model": "gpt-4o",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "email": "you@university.edu",
  "output_dir": "~/research/output"
}
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEEP_RESEARCH_MODEL` | `llama3.1` | LLM model name |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | API endpoint |
| `OPENAI_API_KEY` | `ollama` | API key |
| `DEEP_RESEARCH_MAX_ITER` | `20` | Max iterations |
| `DEEP_RESEARCH_EMAIL` | — | Email for polite API pool |
| `CORE_API_KEY` | — | Free key from [CORE](https://core.ac.uk/api-keys/register) |

Priority: CLI args > environment variables > config file > defaults.

### Models & Compatibility

Deep Researcher requires models with **function/tool calling** support. This is how the agent decides which databases to search.

**Local models (Ollama / LMStudio):**

| Model | Tool Calling | Quality | Notes |
|---|---|---|---|
| `llama3.1` | Yes | Good | Default. Solid all-around |
| `llama3.3:70b` | Yes | Excellent | Best local quality (needs 48GB+ VRAM) |
| `qwen2.5:14b` | Yes | Excellent | Best quality/size ratio for local |
| `qwen2.5-coder:32b` | Yes | Excellent | Strong structured output |
| `gemma3:12b` | Yes | Good | Lightweight alternative |
| `mistral-nemo` | Yes | Good | Fast |
| `deepseek-r1` | **No** | - | Does NOT support tool calling |
| `llama3.2:3b` | **No** | - | Too small for reliable tool use |

**Cloud providers:**

| Provider | Recommended Model | Notes |
|---|---|---|
| OpenAI | `gpt-4o` | Best overall quality |
| OpenAI | `gpt-4o-mini` | Good speed/cost balance |
| Anthropic | `claude-sonnet-4-20250514` | Excellent synthesis |
| Groq | `llama-3.3-70b-versatile` | Fast, free tier available |
| DeepSeek | `deepseek-chat` | Good quality, affordable |
| Together | `Llama-3.3-70B-Instruct-Turbo` | Fast inference |
| OpenRouter | Any tool-calling model | Access 100+ models |

> **If you get a "function calling not supported" error**, your model doesn't support tool use. Switch to one of the models listed above.

---

## How It Compares

| | GPT Researcher | STORM | local-deep-research | **Deep Researcher** |
|---|---|---|---|---|
| Academic databases | 0 (web only) | 0 (web only) | 3 | **6** |
| Citation chains | No | No | No | **Yes** |
| Open access check | No | No | No | **Yes** |
| Paper deduplication | No | No | No | **Yes** |
| BibTeX output | No | No | No | **Yes** |
| Dependencies | ~50+ | ~30+ | ~50+ | **3** |
| Lines of code | ~15K | ~10K | ~15K | **~1.5K** |
| Runs locally | Partial | Partial | Yes | **Yes** |

---

## Extending

Adding a new database is one file:

```python
from deep_researcher.tools.base import Tool
from deep_researcher.models import Paper, ToolResult

class MyDatabaseTool(Tool):
    name = "search_my_database"
    description = "Search My Database for ..."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }

    def execute(self, query: str) -> ToolResult:
        papers = call_my_api(query)
        text = format_results(papers)
        return ToolResult(text=text, papers=papers)
```

Register it in `src/deep_researcher/tools/__init__.py` and it's immediately available to the agent.

---

## How This Was Built

This project started as a study of **Claude Code's source code** — Anthropic's CLI for Claude, which became [publicly accessible](https://x.com/Fried_rice/status/2038894956459290963) on March 31, 2026 through a source map exposure in the npm distribution (~512K lines of TypeScript).

While analyzing the architecture, several patterns stood out as broadly applicable beyond coding assistants:

1. **The agentic tool-call loop** (`queryLoop()` in `query.ts`) — a while loop where the LLM calls tools, gets results, and decides what to do next. Simple but powerful.
2. **Concurrent tool execution** (`partitionToolCalls()` in `toolOrchestration.ts`) — read-only tools run in parallel, write tools run serially. Cuts latency dramatically.
3. **Structured tool results** (`ToolResult<T>` in `Tool.ts`) — tools return typed data alongside human-readable text, enabling proper tracking without parsing.
4. **Retry with exponential backoff** — every external call handles rate limits and transient failures gracefully.

The question was: **what if you applied these production-tested patterns to academic research instead of code editing?**

The result is Deep Researcher — a clean-room Python implementation (~1,500 lines, 3 dependencies) that uses the same architectural DNA as Claude Code but pointed at academic databases instead of file systems. No code was copied; just the design patterns that make agentic systems reliable in production.

### Architecture Mapping

| Claude Code (TypeScript) | Deep Researcher (Python) |
|---|---|
| `queryLoop()` in `query.ts` | `research()` in `agent.py` |
| `buildTool()` with schema + execute | `Tool` class + `execute()` → `ToolResult` |
| `partitionToolCalls()` batching | `execute_concurrent()` via ThreadPoolExecutor |
| `ToolResult<T>` with data + messages | `ToolResult` with text + papers |
| `ToolRegistry` with deny rules | `ToolRegistry` with schema export |
| Exponential backoff on API errors | Same, across all 9 tools |

### Built With

The entire project — from initial architecture study through implementation to this README — was built in a single session using [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6). The agentic patterns we studied in Claude Code's source were implemented by Claude Code itself.

---

## Project Structure

```
src/deep_researcher/
  __main__.py          # CLI entry point
  agent.py             # Agentic research loop (core)
  llm.py               # OpenAI-compatible LLM client
  config.py            # Config file + env var loading
  models.py            # Paper, ToolResult data models
  report.py            # Report + BibTeX + JSON generation
  tools/
    base.py            # Tool base class + concurrent registry
    arxiv_search.py    # arXiv API
    semantic_scholar.py # Semantic Scholar + citation chains
    openalex.py        # OpenAlex API
    crossref.py        # CrossRef API
    pubmed.py          # PubMed E-utilities
    core_search.py     # CORE API
    paper_details.py   # Paper lookup by DOI
    open_access.py     # Unpaywall open access check
```

---

## License

MIT

---

<p align="center">
  <sub>Built by studying how the best agentic systems work, then applying those patterns where they're needed most.</sub>
</p>
