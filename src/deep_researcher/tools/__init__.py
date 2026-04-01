from __future__ import annotations

from deep_researcher.config import Config
from deep_researcher.tools.arxiv_search import ArxivSearchTool
from deep_researcher.tools.base import ToolRegistry
from deep_researcher.tools.core_search import CoreSearchTool
from deep_researcher.tools.crossref import CrossrefSearchTool
from deep_researcher.tools.ieee_xplore import IEEEXploreSearchTool
from deep_researcher.tools.scopus import ScopusSearchTool
from deep_researcher.tools.open_access import OpenAccessTool
from deep_researcher.tools.openalex import OpenAlexSearchTool
from deep_researcher.tools.pubmed import PubMedSearchTool

# Semantic Scholar tools (search, citations, paper details) are available but
# not registered by default — S2 aggressively rate-limits without an API key.
# To enable: uncomment below and set S2 API key or accept rate limits.
# from deep_researcher.tools.semantic_scholar import GetCitationsTool, SemanticScholarSearchTool
# from deep_researcher.tools.paper_details import PaperDetailsTool


def build_tool_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    tools = [
        ArxivSearchTool(),
        OpenAlexSearchTool(email=config.email),
        CrossrefSearchTool(email=config.email),
        PubMedSearchTool(),
        CoreSearchTool(api_key=config.core_api_key),
        ScopusSearchTool(api_key=config.scopus_api_key),
        IEEEXploreSearchTool(api_key=config.ieee_api_key),
        OpenAccessTool(email=config.email),
    ]
    for tool in tools:
        tool.set_year_range(config.start_year, config.end_year)
        registry.register(tool)
    return registry
