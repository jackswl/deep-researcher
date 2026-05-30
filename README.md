<h1 align="center">
  <img src="assets/logo.svg" width="64" alt="Deep Researcher" align="absmiddle">&nbsp;&nbsp;Deep Researcher
</h1>

<p align="center">
  <strong>Search Google Scholar, enrich with OpenAlex, and extract a structured literature matrix, all locally.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/version-0.6.0-blue.svg?style=flat-square" alt="Version: 0.6.0">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#sample-output">Sample Output</a> &middot;
  <a href="#configuration">Configuration</a>
</p>

---

Deep Researcher searches **Google Scholar** for academically-ranked papers, enriches them with full metadata from **OpenAlex** and **CrossRef**, and uses a local LLM to **extract a structured matrix** of each paper (method and key finding), grouped by theme.

- **100 papers** from Google Scholar's semantic search, no keyword hacks, no irrelevant results
- **Full metadata**: DOIs, abstracts, journal names, citation counts, open access URLs
- **Structured extraction**: papers grouped by theme, each reduced to the method and key finding stated in its abstract
- **Consistent `[N]` references**: every row in the tables matches the reference list
- **BibTeX + CSV output** ready for LaTeX/Overleaf or Excel
- **Runs locally** with Ollama. Your queries never leave your machine
- **Tool-based agentic architecture** inspired by [Claude Code](https://github.com/anthropics/claude-code)
- **4 dependencies**, no LangChain

---

## Quick Start

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

### Run with a cloud provider

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
deep-researcher "machine learning for drug discovery" --provider openai

# Groq (fast, free tier available)
export OPENAI_API_KEY="gsk_..."
deep-researcher "CRISPR gene editing" --provider groq

# DeepSeek
export OPENAI_API_KEY="sk-..."
deep-researcher "quantum computing algorithms" --provider deepseek
```

<details>
<summary><strong>All supported providers</strong></summary>

| Provider | Flag | Default Model | API Key |
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

## How It Works

1. **Search**: Queries Google Scholar for up to 100 semantically-ranked academic papers
2. **Enrich**: Concurrent workers (8 threads) look up each paper in OpenAlex/CrossRef for full abstracts, DOIs, and journal metadata
3. **Extract**: LLM groups papers into themes, then extracts a table per theme (method and key finding for each paper, taken only from the abstract)

Each run produces:

```
output/2026-04-02-161823-large-language-models-for-automated-code/
├── report.md        # Themed extraction tables
├── references.bib   # BibTeX (import into LaTeX/Overleaf)
├── papers.json      # Full metadata for every paper
├── papers.csv       # Same data as CSV
└── metadata.json    # Search stats
```

---

## Sample Output

<details open>
<summary><strong>First theme shown in full, remaining themes truncated for brevity</strong></summary>

```markdown
### Literature search: large language models for automated code compliance in BIM

#### Coverage
100 papers found via Google Scholar, enriched via OpenAlex/CrossRef. Years 2010-2026. 96 with DOIs.

#### Themes

##### Automated IFC-Based Compliance Processing (8 papers)

| Ref | Paper | Year | Method | Key finding (as stated) | Cites |
|-----|-------|------|--------|-------------------------|-------|
| [2] | Automated code compliance checking based on a visual language and BIM | 2015 | Visual language + IFC | feasibility of checking designs against codes | 51 |
| [3] | Automated compliance checking using building information models | 2010 | IFC schema checking | reduces time and ambiguity in manual reviews | not stated |
| [4] | A review on BIM-based automated code compliance checking | 2017 | Literature review | maps the evolution of automated compliance | 57 |
| [5] | Knowledge-informed semantic alignment and rule interpretation | 2022 | Semantic alignment | handles complex, changing regulations | 118 |
| [6] | Automated BIM data validation integrating open-standard schema | 2019 | Visual programming + IFC | validates BIM data against open standards | 37 |
| [7] | BIM: automated code checking and compliance processes | 2018 | Rule-based checking | objective fire-safety assessment | 35 |
| [75] | AI-driven IFC processing for automated IBS scoring | 2026 | AI-driven IFC processing | replaces manual, inconsistent IBS assessment | not stated |

##### NLP-Driven Regulatory Interpretation (20 papers)
| Ref | Paper | Year | Method | Key finding (as stated) | Cites |
| ... truncated ... |

##### Generative LLM Methodologies (10 papers)
| ... truncated ... |

#### References
[1] R Amor et al. (2021). The promise of automated compliance checking. *Developments in the Built Environment*. DOI: 10.1016/j.dibe.2020.100039
[2] C Preidel et al. (2015). Automated code compliance checking based on a visual language and building information modeling. *Proceedings of the ISARC*. DOI: 10.22260/isarc2015/0033
[3] D Greenwood (2010). Automated compliance checking using building information models. *Northumbria Research Link*.
[4] AS Ismail et al. (2017). A review on BIM-based automated code compliance checking system. *ICRIIS 2017*. DOI: 10.1109/icriis.2017.8002486
[5] Z Zheng et al. (2022). Knowledge-informed semantic alignment and rule interpretation for automated compliance checking. *Automation in Construction*. DOI: 10.1016/j.autcon.2022.104524
...
[100] Orchestrating LLM-Powered Workflows for Autodesk Revit via Model Context Protocol.
```

</details>

> Every `[N]` in the tables matches `[N]` in the reference list. Every row is extracted from a real abstract: when an abstract does not state the method or finding, the cell reads "not stated" rather than inventing one.

---

## Usage

```
deep-researcher "your research question" [options]

Options:
  --provider PROVIDER    LLM provider (ollama, openai, groq, etc.)
  --model MODEL          LLM model name
  --base-url URL         OpenAI-compatible API URL
  --api-key KEY          API key
  --start-year YEAR      Filter papers from this year onward
  --end-year YEAR        Filter papers up to this year
  --interactive          Ask clarifying questions before researching
  --output DIR           Output directory (default: ./output)
  --email EMAIL          Email for polite API access to OpenAlex/CrossRef
  --verbose              Enable debug logging
```

```bash
# Recent papers only
deep-researcher "federated learning" --start-year 2020

# Specific time window
deep-researcher "attention mechanisms" --start-year 2017 --end-year 2023

# Interactive mode: refine your question first
deep-researcher "machine learning in healthcare" --interactive

# Cloud provider for faster extraction
deep-researcher "quantum computing" --provider groq --start-year 2022
```

---

## Configuration

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

## Models

The LLM is only used to **group papers into themes** and **extract** the per-paper table. Search is handled by Google Scholar, no LLM involved. Even smaller models work well.

Any OpenAI-compatible model works. Use `--model` to override the default:

```bash
deep-researcher "your query" --model gemma4
```

| Model | ID | Notes |
|---|---|---|
| Qwen 3.5 9B | `qwen3.5:9b` | **Default.** Good quality/size ratio |
| Gemma 4 | `gemma4` | 128K context, reliable extraction |
| Qwen 3.5 27B | `qwen3.5:27b` | Higher quality, needs 16GB+ VRAM |
| Llama 4 Scout | `llama4:scout` | 10M context |
| DeepSeek V3.2 | `deepseek-v3.2` | Strong reasoning |

---

## License

MIT
