# Refine a literature query with Parallel Search MCP

Before running a literature search, use web sources to identify terminology,
datasets, and related research questions. [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
adds web search and page fetching to Claude Code. Its anonymous endpoint is free
for light use, with no Parallel account or API key required. Anonymous search uses
Fast mode.

This optional workflow runs in Claude Code before you pass a query to Deep
Researcher. Deep Researcher still searches Google Scholar, enriches metadata with
OpenAlex/CrossRef, and extracts its literature matrix from paper abstracts. Web
excerpts are background context, not paper abstracts or input to that matrix.

## Connect Claude Code

Start with [Deep Researcher installed](../README.md#quick-start) and
[Claude Code installed and signed in](https://github.com/anthropics/claude-code#get-started).
Claude Code has its own account and usage requirements; the anonymous Parallel
endpoint does not provide model access.

From the `deep-researcher` directory, register the HTTP server for your local
Claude Code project configuration:

If a server named `parallel-search` is already configured, inspect it with
`claude mcp get parallel-search` before making changes.

```bash
claude mcp add --transport http --scope local parallel-search https://search.parallel.ai/mcp \
  --header "User-Agent: deep-researcher"
claude mcp get parallel-search
claude
```

Inside Claude Code, run `/mcp` to check the connection. The server exposes
`web_search` for finding sources and `web_fetch` for reading selected URLs. Allow
the tools when prompted. If they do not appear, restart Claude Code and check
`/mcp` again.

These requests send your search terms and selected URLs to Parallel. Claude Code
also processes the conversation with its configured model. Use public research
topics and source links you are comfortable sharing with those services.

## Explore, then choose a query

For example, give Claude Code this prompt:

```text
Use parallel-search's web_search to investigate terminology for automated
building-code compliance using building information modeling. Find relevant
research groups, standards, and survey papers. Use web_fetch on promising URLs
when the search excerpts are insufficient. Cite source URLs for the terminology
you suggest, distinguish research papers from other web pages, and propose three
focused literature-search queries. Do not write a literature review.
```

Inspect the linked sources and choose or edit one query. Then run Deep Researcher
in a terminal, using your usual model setup. For example, with Ollama running and
the default model downloaded as in the Quick Start:

```bash
deep-researcher "building information modeling automated building code compliance" --start-year 2020
```

Deep Researcher's output includes `report.md`, `papers.json`, `papers.csv`, and
`references.bib`. Keep any web notes separately so that the matrix and its
references remain grounded in the retrieved papers. Check the original papers
before citing them; a web result or excerpt does not establish peer review.

## Limits and removal

Anonymous access has rate limits. If a tool reports a quota limit, wait as
instructed or consult the [Parallel setup and authentication options](https://docs.parallel.ai/integrations/mcp/search-mcp#installation).
No Parallel credentials belong in Deep Researcher's `config.json` for this
workflow.

To remove the local Claude Code server configuration, run this from the same
directory:

```bash
claude mcp remove --scope local parallel-search
```
