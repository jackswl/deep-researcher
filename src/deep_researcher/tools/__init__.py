from __future__ import annotations

from deep_researcher.config import Config
from deep_researcher.llm import LLMClient
from deep_researcher.tools.arxiv_search import ArxivSearchTool
from deep_researcher.tools.base import ToolRegistry
from deep_researcher.tools.categorize import CategorizeTool
from deep_researcher.tools.core_search import CoreSearchTool
from deep_researcher.tools.cross_analysis import CrossAnalysisTool
from deep_researcher.tools.crossref import CrossrefSearchTool
from deep_researcher.tools.enrichment import EnrichmentTool
from deep_researcher.tools.ieee_xplore import IEEEXploreSearchTool
from deep_researcher.tools.open_access import OpenAccessTool
from deep_researcher.tools.openalex import OpenAlexSearchTool
from deep_researcher.tools.pubmed import PubMedSearchTool
from deep_researcher.tools.scholar_search import ScholarSearchTool
from deep_researcher.tools.scopus import ScopusSearchTool
from deep_researcher.tools.synthesize import SynthesisTool


def build_tool_registry(config: Config, llm: LLMClient | None = None) -> ToolRegistry:
    """Build tool registry with all available tools."""
    registry = ToolRegistry()

    # Pipeline tools
    registry.register(ScholarSearchTool())
    registry.register(EnrichmentTool())
    if llm:
        registry.register(CategorizeTool(llm=llm))
        registry.register(SynthesisTool(llm=llm))
        registry.register(CrossAnalysisTool(llm=llm))

    # Database tools
    database_tools = [
        ArxivSearchTool(),
        OpenAlexSearchTool(email=config.email),
        CrossrefSearchTool(email=config.email),
        PubMedSearchTool(),
        CoreSearchTool(api_key=config.core_api_key),
        ScopusSearchTool(api_key=config.scopus_api_key),
        IEEEXploreSearchTool(api_key=config.ieee_api_key),
        OpenAccessTool(email=config.email),
    ]
    for tool in database_tools:
        tool.set_year_range(config.start_year, config.end_year)
        registry.register(tool)

    return registry
