from __future__ import annotations

import logging
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessage

from deep_researcher.config import Config

logger = logging.getLogger("deep_researcher")


class ToolCallingNotSupported(Exception):
    """Raised when the model doesn't support function/tool calling."""

# Rough token estimate: ~4 chars per token for English text
_CHARS_PER_TOKEN = 4


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=float(config.timeout),
            max_retries=0,  # We handle retries ourselves for better control
        )
        self.model = config.model
        self._max_retries = 3

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatCompletionMessage:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Retry with exponential backoff (Claude Code recovery pattern)
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError("LLM returned no choices")
                return response.choices[0].message
            except RateLimitError as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited (attempt %d/%d), waiting %ds", attempt + 1, self._max_retries, wait)
                time.sleep(wait)
            except APITimeoutError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("API timeout (attempt %d/%d), waiting %ds", attempt + 1, self._max_retries, wait)
                time.sleep(wait)
            except APIError as e:
                err_str = str(e).lower()
                # Detect models that don't support function calling
                if any(hint in err_str for hint in ("tool", "function", "not supported", "invalid param")):
                    raise ToolCallingNotSupported(
                        f"Model '{self.model}' may not support function calling.\n"
                        f"Recommended models: qwen3.5:9b (local), gpt-5.4-mini (OpenAI), claude-sonnet-4-6 (Anthropic)\n"
                        f"Original error: {e}"
                    ) from e
                # Retry on server errors (5xx), fail fast on client errors (4xx)
                if e.status_code and 500 <= e.status_code < 600:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning("Server error %s (attempt %d/%d)", e.status_code, attempt + 1, self._max_retries)
                    time.sleep(wait)
                else:
                    raise
        raise last_error  # type: ignore[misc]

    def chat_no_think(self, messages: list[dict]) -> str:
        """Chat without thinking mode (for Ollama local models).

        Uses Ollama's native API with think:false to suppress reasoning tokens.
        Falls back to the standard OpenAI-compatible chat if not running locally.
        """
        # Only use native API for local Ollama/LMStudio
        if "localhost" not in self.client.base_url.host and "127.0.0.1" not in self.client.base_url.host:
            response = self.chat(messages)
            return response.content or ""

        import httpx as _httpx
        # Convert to Ollama native format
        base = str(self.client.base_url).replace("/v1", "").rstrip("/")
        ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        last_error = None
        for attempt in range(self._max_retries):
            try:
                resp = _httpx.post(
                    f"{base}/api/chat",
                    json={"model": self.model, "messages": ollama_messages, "stream": False, "think": False},
                    timeout=float(self.client.timeout),
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
            except _httpx.TimeoutException as e:
                last_error = e
                time.sleep(2 ** attempt)
            except _httpx.HTTPError as e:
                last_error = e
                if hasattr(e, 'response') and e.response and e.response.status_code < 500:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"chat_no_think failed after {self._max_retries} retries: {last_error}")

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate for message list (Claude Code proactive blocking pattern)."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content") or ""
            total_chars += len(content)
            # Tool calls add ~50 tokens overhead each
            for tc in msg.get("tool_calls", []):
                total_chars += len(tc.get("function", {}).get("arguments", ""))
                total_chars += 200  # Schema overhead
        return total_chars // _CHARS_PER_TOKEN
