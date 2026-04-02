# src/deep_researcher/errors.py
"""Custom error types (Claude Code utils/errors.ts pattern).

Each error carries structured context for diagnostics.
"""
from __future__ import annotations


class DeepResearcherError(Exception):
    """Base error for all deep-researcher exceptions."""


class ToolCallingNotSupported(DeepResearcherError):
    """Model does not support function/tool calling."""

    def __init__(self, model: str, original_error: Exception | None = None) -> None:
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"Model '{model}' may not support function calling.\n"
            f"Recommended models: qwen3.5:9b (local), gpt-5.4-mini (OpenAI), claude-sonnet-4-6 (Anthropic)"
            + (f"\nOriginal error: {original_error}" if original_error else "")
        )


class ConfigValidationError(DeepResearcherError):
    """Invalid configuration value."""

    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        super().__init__(f"Invalid config '{field}={value}': {reason}")


class SearchError(DeepResearcherError):
    """A search tool failed after retries."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Search tool '{tool_name}' failed: {reason}")


class SynthesisError(DeepResearcherError):
    """Synthesis phase failed."""


class ContextOverflowError(DeepResearcherError):
    """Context window exceeded and recovery failed."""


class PhaseError(DeepResearcherError):
    """A pipeline phase failed but may be recoverable."""
    def __init__(self, phase: str, reason: str, recoverable: bool = True) -> None:
        self.phase = phase
        self.recoverable = recoverable
        super().__init__(f"Phase '{phase}' failed: {reason}")
