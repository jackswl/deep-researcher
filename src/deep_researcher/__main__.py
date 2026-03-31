from __future__ import annotations

import argparse
import sys

from rich.console import Console

from deep_researcher import __version__
from deep_researcher.agent import ResearchAgent
from deep_researcher.config import Config

# Provider presets — saves users from looking up base URLs
PROVIDERS: dict[str, dict[str, str]] = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "default_model": "llama3.1"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio", "default_model": "default"},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key": "", "default_model": "gpt-4o"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "api_key": "", "default_model": "claude-sonnet-4-20250514"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "", "default_model": "llama-3.3-70b-versatile"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key": "", "default_model": "deepseek-chat"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "", "default_model": "anthropic/claude-sonnet-4"},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key": "", "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deep-researcher",
        description="An agentic academic research assistant that searches multiple databases and produces literature reviews.",
    )
    parser.add_argument("query", nargs="?", help="Research question to investigate")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), help="LLM provider (auto-configures base URL and model)")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum research iterations (default: 20)")
    parser.add_argument("--output", default=None, help="Output directory (default: ./output)")
    parser.add_argument("--email", default=None, help="Email for polite API access to OpenAlex/CrossRef/Unpaywall")
    parser.add_argument("--breadth", type=int, default=None, help="Search breadth: query variations (1-5, default: 3)")
    parser.add_argument("--depth", type=int, default=None, help="Search depth: citation rounds (0-5, default: 2)")
    parser.add_argument("--version", action="version", version=f"deep-researcher {__version__}")
    args = parser.parse_args()

    console = Console()

    if not args.query:
        console.print("[bold]Deep Researcher[/bold] — Academic Literature Review Agent\n")
        console.print("Usage: deep-researcher \"your research question here\"\n")
        console.print("Examples:")
        console.print('  deep-researcher "transformer models in structural health monitoring"')
        console.print('  deep-researcher "machine learning drug discovery" --provider openai')
        console.print('  deep-researcher "CRISPR gene editing" --provider groq')
        console.print('  deep-researcher "deep learning NLP" --provider ollama --model qwen2.5:14b')
        console.print("")
        console.print("Providers: " + ", ".join(PROVIDERS.keys()))
        console.print("Config file: ~/.deep-researcher/config.json")
        console.print("Run deep-researcher --help for all options.")
        sys.exit(0)

    # Apply provider preset first, then overrides
    config = Config()
    if args.provider:
        preset = PROVIDERS[args.provider]
        config.base_url = preset["base_url"]
        config.api_key = preset["api_key"]
        config.model = preset["default_model"]

    # Explicit args override provider preset
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.output:
        config.output_dir = args.output
    if args.email:
        config.email = args.email
    if args.breadth is not None:
        config.breadth = max(1, min(args.breadth, 5))
    if args.depth is not None:
        config.depth = max(0, min(args.depth, 5))

    # Check for missing API key on cloud providers
    if not config.api_key or config.api_key in ("ollama", "lm-studio"):
        if args.provider and args.provider not in ("ollama", "lmstudio"):
            console.print(f"[red]Error: --provider {args.provider} requires an API key.[/red]")
            console.print(f"Set it with: --api-key YOUR_KEY  or  export OPENAI_API_KEY=YOUR_KEY")
            sys.exit(1)

    console.print(f"[dim]Model: {config.model} @ {config.base_url}[/dim]")
    console.print(f"[dim]Settings: breadth={config.breadth} depth={config.depth} max_iter={config.max_iterations}[/dim]")

    agent = ResearchAgent(config)
    try:
        report = agent.research(args.query)
        if report:
            console.print("\n")
            try:
                from rich.markdown import Markdown
                console.print(Markdown(report))
            except Exception:
                console.print(report)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/yellow]")
        if agent.papers:
            console.print(f"[yellow]Saving {len(agent.papers)} papers collected so far...[/yellow]")
            agent._save(args.query, "# Research Interrupted\n\nPartial results — research was interrupted before synthesis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
