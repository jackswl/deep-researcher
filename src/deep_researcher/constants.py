"""Centralized constants (extracted from inline magic numbers).

All tunable thresholds in one place for easy adjustment.
"""

# --- Token estimation ---
CHARS_PER_TOKEN = 4  # Rough approximation for English text

# --- Google Scholar search ---
SCHOLAR_RESULTS_PER_QUERY = 15     # Results per query variation
NUM_QUERY_VARIATIONS = 4           # LLM-generated query variations (+ original = 5 total)

# --- Synthesis phase ---
MAX_SYNTHESIS_PAPERS = 200          # Cap on papers sent to synthesis
MAX_FINAL_CATEGORIES = 6            # Target number of categories after merging
CATEGORIZE_BATCH_SIZE = 20          # Papers per categorization LLM call
CATEGORY_SYNTHESIS_TIMEOUT = 300    # Seconds before skipping a category (5 min)
CATEGORY_TOKEN_BUDGET = 15_000      # Token budget per category corpus
FALLBACK_TOKEN_BUDGET = 8_000       # Token budget for single-pass fallback
FALLBACK_MAX_PAPERS = 20            # Max papers for fallback synthesis

# --- Tier-1 sources (peer-reviewed / curated) ---
TIER1_SOURCES = frozenset({"scopus", "ieee", "pubmed"})

# --- Concurrency ---
MAX_TOOL_CONCURRENCY = 8            # Max parallel tool executions

# --- Display / truncation ---
ABSTRACT_MAX_CHARS = 250            # Max abstract length in corpus entries
ABSTRACT_MIN_CUT = 150              # Minimum cut point for sentence-boundary truncation

# --- Relevance filtering ---
MIN_CATEGORIZATION_COVERAGE = 0.3   # Warn if <30% of papers categorized
