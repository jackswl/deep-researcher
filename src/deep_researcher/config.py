from __future__ import annotations

import json
import os
from dataclasses import dataclass


CONFIG_LOCATIONS = [
    os.path.expanduser("~/.deep-researcher/config.json"),
    "./deep-researcher.json",
]


def _load_config_file() -> dict:
    for path in CONFIG_LOCATIONS:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _get(file_cfg: dict, key: str, env_var: str, default: str) -> str:
    return os.getenv(env_var) or file_cfg.get(key) or default


@dataclass
class Config:
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    max_iterations: int = 20
    output_dir: str = "./output"
    email: str = ""
    core_api_key: str = ""
    scopus_api_key: str = ""
    ieee_api_key: str = ""
    exa_api_key: str = ""
    timeout: int = 500  # ~8 min — local models need time for multi-step synthesis
    start_year: int | None = None
    end_year: int | None = None
    interactive: bool = False

    def __post_init__(self) -> None:
        file_cfg = _load_config_file()

        if not self.model:
            self.model = _get(file_cfg, "model", "DEEP_RESEARCH_MODEL", "qwen3.5:9b")
        if not self.base_url:
            self.base_url = _get(file_cfg, "base_url", "OPENAI_BASE_URL", "http://localhost:11434/v1")
        if not self.api_key:
            self.api_key = _get(file_cfg, "api_key", "OPENAI_API_KEY", "ollama")
        if not self.email:
            self.email = _get(file_cfg, "email", "DEEP_RESEARCH_EMAIL", "")
        if not self.core_api_key:
            self.core_api_key = _get(file_cfg, "core_api_key", "CORE_API_KEY", "")
        if not self.scopus_api_key:
            self.scopus_api_key = _get(file_cfg, "scopus_api_key", "SCOPUS_API_KEY", "")
        if not self.ieee_api_key:
            self.ieee_api_key = _get(file_cfg, "ieee_api_key", "IEEE_API_KEY", "")
        if not self.exa_api_key:
            self.exa_api_key = _get(file_cfg, "exa_api_key", "EXA_API_KEY", "")

        iter_str = os.getenv("DEEP_RESEARCH_MAX_ITER") or str(file_cfg.get("max_iterations", ""))
        if iter_str:
            try:
                self.max_iterations = int(iter_str)
            except ValueError:
                pass

        output = os.getenv("DEEP_RESEARCH_OUTPUT") or file_cfg.get("output_dir") or ""
        if output:
            self.output_dir = output

        # Year range from env vars / config file
        for attr, env_key in [("start_year", "DEEP_RESEARCH_START_YEAR"), ("end_year", "DEEP_RESEARCH_END_YEAR")]:
            if getattr(self, attr) is None:
                raw = os.getenv(env_key) or str(file_cfg.get(attr, ""))
                if raw:
                    try:
                        setattr(self, attr, int(raw))
                    except ValueError:
                        pass

        self.max_iterations = max(1, min(self.max_iterations, 50))

        self.validate()

    def validate(self) -> None:
        """Validate configuration values (Claude Code boundary-validation pattern)."""
        from deep_researcher.errors import ConfigValidationError

        if self.start_year is not None and self.end_year is not None:
            if self.start_year > self.end_year:
                raise ConfigValidationError("start_year", self.start_year,
                    f"must be <= end_year ({self.end_year})")
        if self.start_year is not None and self.start_year < 1900:
            raise ConfigValidationError("start_year", self.start_year, "must be >= 1900")
        if self.end_year is not None and self.end_year > 2100:
            raise ConfigValidationError("end_year", self.end_year, "must be <= 2100")
        if self.max_iterations < 1:
            raise ConfigValidationError("max_iterations", self.max_iterations, "must be >= 1")
        if self.timeout < 1:
            raise ConfigValidationError("timeout", self.timeout, "must be >= 1")
