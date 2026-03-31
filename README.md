<p align="center">
  <br>
  <img src="assets/logo.svg" width="48" alt="">&nbsp;&nbsp;&nbsp;&nbsp;<img src="assets/logo-right.svg" width="48" alt="">
  <h1 align="center">Deep Researcher</h1>
  <p align="center">
    <strong>An agentic academic research assistant that searches 8 databases (open and paywalled), follows citation chains, and writes structured literature reviews.</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/dependencies-3-brightgreen.svg" alt="Dependencies: 3">
    <img src="https://img.shields.io/badge/lines_of_code-~1.5K-blue.svg" alt="LOC: ~1.5K">
    <img src="https://img.shields.io/badge/databases-8-orange.svg" alt="Databases: 8">
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#how-it-works">How It Works</a> &middot;
    <a href="#sample-output">Sample Output</a> &middot;
    <a href="#configuration">Configuration</a> &middot;
    <a href="#how-this-was-built">Origin Story</a>
  </p>
</p>

---

Most "AI research tools" search the web and write you a story. Deep Researcher does what an actual researcher does: **searches academic databases (including paywalled publishers), reads abstracts, follows citation chains, and builds a structured corpus before writing anything.**

It runs a real agentic loop where the LLM decides what to search next, when to dig deeper, and when to stop. Not a pipeline. Not a single prompt. An autonomous research agent.

**3 dependencies. ~1,500 lines. No LangChain.**

> [!NOTE]
> Deep Researcher searches **real academic databases** (Semantic Scholar, OpenAlex, arXiv, CrossRef, PubMed, CORE, Scopus, IEEE Xplore), not the open web. It accesses both open access and paywalled sources through their free APIs (abstracts and metadata are always available, no subscription needed). Every paper it finds actually exists. No hallucinated sources.

---

## Why This Exists

Existing tools like GPT Researcher, STORM, Gemini Deep Research, and ChatGPT all rely on **web search**. They write you a narrative essay with a few citations sprinkled in. That's useful for a quick overview, but not for an actual literature review.

The problem: most published research lives behind publisher paywalls (Elsevier, Springer, IEEE, ASCE, Wiley). Web search tools can't see it. They only find what's freely indexed on the open web.

Deep Researcher takes a different approach: **gather a comprehensive corpus first, then categorize and synthesize.** It searches both open and paywalled academic databases (abstracts and metadata are freely available), follows citation chains, deduplicates across sources, and produces a structured analysis of everything it finds.

Built for **academics, grad students, and researchers** who need:

- **Comprehensive coverage** across 8 academic databases, including paywalled publishers via Scopus and IEEE
- **Literature gathering first, storytelling second.** Every paper found, categorized, and cited.
- Citation chains to find foundational and recent work
- BibTeX output ready for LaTeX/Overleaf
- Open access detection for paywalled papers
- A tool they can run **100% locally** with their own models

> [!TIP]
> Unlike Gemini or ChatGPT deep research, Deep Researcher doesn't give you a narrative with a few sources. It builds a **complete paper corpus**, categorizes by theme, synthesizes across groups, and identifies gaps. Think Connected Papers + automated analysis.

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
deep-researcher "applications of transformer models in structural health monitoring"
```

### Unlock paywalled databases (recommended)

Deep Researcher works out of the box with 5 open databases. But most research tools stop there, missing the majority of published academic work that lives behind publisher paywalls.

**The key insight:** you don't need full text to do a literature review. Abstracts, citation counts, and metadata from Scopus and IEEE Xplore are enough to identify relevant work, and these are available for free through their APIs.

| Database | What it adds | Registration |
|---|---|---|
| **Scopus** (Elsevier) | 90M+ records from Elsevier, Springer, Wiley, ASCE, IEEE, ACM. The largest abstract database in academia. | [dev.elsevier.com](https://dev.elsevier.com/) |
| **IEEE Xplore** | 6M+ IEEE/IET journal and conference papers. Essential for engineering and CS. | [developer.ieee.org](https://developer.ieee.org/) |
| **CORE** | 300M+ open access articles with full text links. | [core.ac.uk/api-keys/register](https://core.ac.uk/api-keys/register) |

**How to set up (5 minutes, all free):**

1. **Scopus**: Go to [dev.elsevier.com](https://dev.elsevier.com/) > Create account > "My API Key" > Create. You get a key instantly. No institutional affiliation required. Free tier gives 20,000 searches/week.

2. **IEEE Xplore**: Go to [developer.ieee.org](https://developer.ieee.org/) > Register > Request API key. Free tier gives 200 calls/day.

3. **CORE** (optional): Go to [core.ac.uk/api-keys/register](https://core.ac.uk/api-keys/register) > Register > Get key.

4. **Save your keys** in `~/.deep-researcher/config.json`:

```json
{
  "scopus_api_key": "your-scopus-key",
  "ieee_api_key": "your-ieee-key",
  "core_api_key": "your-core-key"
}
```

> [!TIP]
> **Scopus is the most impactful addition.** It indexes nearly every major publisher, so even one key dramatically increases coverage. For a topic like "structural health monitoring," Scopus will find ASCE, Springer, and Elsevier journal papers that open databases miss entirely.

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

Each session produces four files:

```
output/2026-03-31-142315-transformer-structural-health/
├── report.md        # Categorized literature review with synthesis
├── references.bib   # Deduplicated BibTeX (import into LaTeX/Overleaf)
├── papers.json      # Full metadata for every paper found
└── metadata.json    # Research stats: databases, coverage, year range
```

---

## Sample Output

Here's what the synthesis looks like. **Not a list of summaries. Structured analysis:**

```markdown
### Transformer Models for Structural Health Monitoring

#### Coverage
32 papers found across 5 databases (2019-2026). 8 open access.

#### Categories

##### 1. Vision-Based Damage Detection (12 papers)
**What this group does:** Uses CNNs and vision transformers on structural images
to detect cracks, corrosion, and deformation.
**Key methods:** ResNet-50 transfer learning, ViT fine-tuning, YOLO object detection
**Main findings:** 90-97% accuracy on controlled datasets, drops to 70-82% in field
conditions. The lab-to-field gap is the central unsolved problem.
**Limitations:** Requires high-resolution images; lighting/weather sensitivity.

| Ref | Paper | Year | Method | Key Finding | Citations |
|-----|-------|------|--------|-------------|-----------|
| [1] | Li et al. | 2022 | ResNet-50 | 94% crack detection | 145 |
| [5] | Wang et al. | 2023 | ViT-Base | 97% lab / 72% field | 89 |
| [12] | Park et al. | 2024 | YOLOv8 | Real-time bridge inspection | 34 |
| ... | ... | ... | ... | ... | ... |

##### 2. Vibration-Based Monitoring (9 papers)
...

#### Cross-Category Patterns
Categories 1 and 2 are converging: 3 papers in 2024-2025 combine vision
+ vibration data. Physics-informed approaches appear in both but aren't
dominant yet. Transfer learning is the most common strategy across all groups.

#### Gaps & Opportunities
1. No multi-modal fusion combining vision + vibration + thermal in one model
2. No studies on long-term model drift (models trained on new structures,
   tested years later on degraded ones)
3. ...
```

> Every paper found appears in a category table. Every paper is in the references. No paper is invented.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  YOUR RESEARCH QUESTION                 │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   Phase 1-2: SEARCH AGENT   │
          │                             │
          │  ┌─── arXiv ────────────┐   │
          │  ├─── Semantic Scholar ─┤   │
          │  ├─── OpenAlex ─────────┤   │    The LLM decides
          │  ├─── CrossRef ─────────┤   │    what to search,
          │  ├─── PubMed ───────────┤ ► │    follows citations,
          │  ├─── CORE ─────────────┤   │    and stops when it
          │  ├─── Scopus ───────────┤   │    has enough papers
          │  └─── IEEE Xplore ──────┘   │
          │                             │
          │  + Citation chain following │
          │  + Open access detection    │
          │  + Concurrent execution     │
          └──────────────┬──────────────┘
                         │
              Structured paper corpus
              (deduplicated, merged)
                         │
          ┌──────────────▼──────────────┐
          │  Phase 3: SYNTHESIS AGENT   │
          │                             │
          │  Receives ALL papers as     │
          │  structured data, then:     │
          │                             │
          │  1. Categorizes by theme    │
          │  2. Synthesizes per group   │
          │  3. Finds cross-group       │
          │     patterns                │
          │  4. Identifies gaps         │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │ report.md + references.bib  │
          │ papers.json + metadata.json │
          └─────────────────────────────┘
```

### Research Intensity: `--breadth` and `--depth`

These two flags control how thorough the research is. Think of it like adjusting a microscope:

```
--breadth = how WIDE you search (number of different search angles)
--depth   = how DEEP you dig  (how many citation chains to follow)
```

| Setting | What happens | Papers found | Time |
|---|---|---|---|
| `--breadth 1 --depth 0` | One search query, no citation following | ~5-10 | ~1 min |
| `--breadth 3 --depth 2` | **(default)** 3 query angles, follows citations for top papers | ~15-30 | ~3-5 min |
| `--breadth 5 --depth 4` | 5 query angles with synonyms, deep citation chains | ~30-60 | ~8-15 min |

**Breadth example:** For "machine learning in structural engineering", breadth=3 means the agent searches:
1. "machine learning structural engineering"
2. "deep learning civil infrastructure"
3. "AI-based structural health monitoring"

**Depth example:** depth=2 means for the most-cited papers found, the agent:
1. Checks who cites them (find newer follow-up work)
2. Checks what they cite (find foundational papers)

> [!TIP]
> Start with defaults. If the report feels thin, increase breadth. If it's missing foundational papers, increase depth.

---

## Academic Databases

| Tool | Source | Papers | What It Covers |
|---|---|---|---|
| `search_arxiv` | arXiv | 2.4M+ | Preprints: CS, physics, math, engineering, biology |
| `search_semantic_scholar` | Semantic Scholar | 200M+ | All fields, citation graphs, TLDR summaries |
| `search_openalex` | OpenAlex | 250M+ | Fully open metadata, excellent coverage |
| `search_crossref` | CrossRef | 150M+ | Elsevier, Springer, IEEE, Wiley, ACM |
| `search_pubmed` | PubMed | 36M+ | Biomedical and life sciences |
| `search_core` | CORE | 300M+ | Open access full texts worldwide |
| `search_scopus` | Scopus (Elsevier) | 90M+ | Most major publishers; abstracts of paywalled papers |
| `search_ieee` | IEEE Xplore | 6M+ | IEEE/IET journals, conferences, standards |
| `get_paper_details` | Semantic Scholar | - | Deep lookup by DOI with TLDR |
| `get_citations` | Semantic Scholar | - | "Who cites this" / "What this cites" |
| `find_open_access` | Unpaywall | - | Find free legal copies of paywalled papers |

> All APIs are **free** (some require a free registration for an API key). No Tavily, no SearXNG, no paid search APIs required.

---

## Usage

```
deep-researcher "your research question" [options]

Options:
  --provider PROVIDER    LLM provider (ollama, openai, groq, etc.)
  --model MODEL          LLM model name
  --base-url URL         OpenAI-compatible API URL
  --api-key KEY          API key
  --breadth N            Search breadth: query angles (1-5, default: 3)
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
  "model": "qwen3.5:9b",
  "base_url": "http://localhost:11434/v1",
  "api_key": "ollama",
  "email": "you@university.edu",
  "output_dir": "~/research/output",
  "scopus_api_key": "your-scopus-key",
  "ieee_api_key": "your-ieee-key",
  "core_api_key": "your-core-key"
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
| `DEEP_RESEARCH_MAX_ITER` | `20` | Max iterations |
| `DEEP_RESEARCH_EMAIL` | - | Email for polite API pool |
| `CORE_API_KEY` | - | Free key from [CORE](https://core.ac.uk/api-keys/register) |
| `SCOPUS_API_KEY` | - | Free key from [Elsevier Dev](https://dev.elsevier.com/) |
| `IEEE_API_KEY` | - | Free key from [IEEE Developer](https://developer.ieee.org/) |

</details>

---

## Models & Compatibility

> [!IMPORTANT]
> Deep Researcher requires models with **function/tool calling** support. If your model doesn't support this, you'll get a clear error message with recommendations.

<details>
<summary><strong>Local models (Ollama / LMStudio)</strong></summary>

| Model | Ollama ID | Tool Calling | Notes |
|---|---|---|---|
| Qwen 3.5 9B | `qwen3.5:9b` | Yes | **Default.** Best quality/size ratio, 256K context |
| Qwen 3.5 27B | `qwen3.5:27b` | Yes | Higher quality, needs 16GB+ VRAM |
| Qwen3-Coder 30B | `qwen3-coder:30b` | Yes | Best agentic tool calling locally |
| Qwen 3 32B | `qwen3:32b` | Yes | Strong general purpose |
| Llama 4 Scout | `llama4:scout` | Yes | 10M context, multimodal |
| Llama 4 Maverick | `llama4:maverick` | Yes | 128 experts, highest local quality |
| DeepSeek V3.2 | `deepseek-v3.2` | Yes | Strong reasoning |
| GLM-4.7-Flash | `glm-4.7-flash` | Yes | Fast, balanced |
| DeepSeek R1 | `deepseek-r1` | **No** | Does NOT support tool calling |

</details>

<details>
<summary><strong>Cloud models</strong></summary>

| Provider | Model | Model ID | Notes |
|---|---|---|---|
| OpenAI | GPT-5.4 | `gpt-5.4` | Best overall, 1M context |
| OpenAI | GPT-5.4 Mini | `gpt-5.4-mini` | Great speed/cost balance |
| OpenAI | GPT-OSS 120B | `gpt-oss-120b` | Open-weight, Apache 2.0 |
| Anthropic | Claude Opus 4.6 | `claude-opus-4-6` | Top synthesis, 1M context |
| Anthropic | Claude Sonnet 4.6 | `claude-sonnet-4-6` | Fast, excellent |
| Groq | Qwen3 32B | `qwen/qwen3-32b` | Ultra-fast LPU inference |
| DeepSeek | V3.2 | `deepseek-chat` | Affordable, strong |
| Together | Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Fast |
| OpenRouter | Any | Browse 40+ models | 29 free models |

</details>

---

## How It Compares

| | GPT Researcher | STORM | local-deep-research | **Deep Researcher** |
|---|---|---|---|---|
| Academic databases | 0 (web only) | 0 (web only) | 3 | **8** |
| Paywalled sources | No | No | No | **Yes (abstracts)** |
| Citation chains | No | No | No | **Yes** |
| Open access check | No | No | No | **Yes** |
| Paper deduplication | No | No | No | **Yes** |
| BibTeX output | No | No | No | **Yes** |
| Categorized synthesis | No | No | No | **Yes** |
| Dependencies | ~50+ | ~30+ | ~50+ | **3** |
| Lines of code | ~15K | ~10K | ~15K | **~1.5K** |
| Runs fully local | Partial | Partial | Yes | **Yes** |
| Paid APIs required | Yes (Tavily) | Yes | Yes (SearXNG) | **No** |

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

This project started as a study of **Claude Code's source code** (Anthropic's CLI for Claude), which became [publicly accessible](https://x.com/Fried_rice/status/2038894956459290963) on March 31, 2026 through a source map exposure in the npm distribution (~512K lines of TypeScript).

While analyzing the architecture, several patterns stood out as broadly applicable beyond coding assistants:

1. **The agentic tool-call loop** (`queryLoop()` in `query.ts`): a while loop where the LLM calls tools, gets results, and decides what to do next
2. **Partitioned concurrent execution** (`partitionToolCalls()` in `toolOrchestration.ts`): read-only tools run in parallel, write tools run serially
3. **Structured tool results** (`ToolResult<T>` in `Tool.ts`): tools return typed data alongside human-readable text
4. **Token-aware context compression** (`autoCompact` in `compact.ts`): keeps conversations within context limits
5. **Multi-level retry/recovery**: exponential backoff, reactive compaction, graceful degradation

The question was: **what if you applied these production-tested patterns to academic research instead of code editing?**

The result is Deep Researcher: a clean-room Python implementation that uses the same architectural DNA as Claude Code but pointed at academic databases. No code was copied; just the design patterns that make agentic systems reliable in production.

<details>
<summary><strong>Architecture mapping: Claude Code → Deep Researcher</strong></summary>

| Claude Code (TypeScript) | Deep Researcher (Python) |
|---|---|
| `queryLoop()` in `query.ts` | `_search_phase()` in `agent.py` |
| `buildTool()` with schema + execute | `Tool` class + `execute()` → `ToolResult` |
| `partitionToolCalls()` batching | `execute_partitioned()` via ThreadPoolExecutor |
| `ToolResult<T>` with data + messages | `ToolResult` with text + papers |
| `autoCompact` token-aware compression | `_compact_messages` with token estimation |
| Exponential backoff + reactive recovery | Same, at both tool and LLM level |

</details>

### Built With

The entire project (architecture study, implementation, this README) was built in a single session using [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6). The agentic patterns we studied in Claude Code's source were implemented by Claude Code itself.

---

## Project Structure

```
src/deep_researcher/
  __main__.py          # CLI entry point + provider presets
  agent.py             # Two-phase research: search agent → synthesis agent
  llm.py               # OpenAI-compatible client with retry/recovery
  config.py            # Config file + env var + CLI loading
  models.py            # Paper (with dedup + merge) + ToolResult
  report.py            # Report + BibTeX + JSON + metadata generation
  tools/
    base.py            # Tool base class + partitioned concurrent registry
    arxiv_search.py    # arXiv API
    semantic_scholar.py # Semantic Scholar + citation chains
    openalex.py        # OpenAlex API
    crossref.py        # CrossRef API
    pubmed.py          # PubMed E-utilities
    core_search.py     # CORE API
    scopus.py          # Scopus (Elsevier) API
    ieee_xplore.py     # IEEE Xplore API
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
