"""Runtime configuration, loaded from environment / .env via pydantic-settings.

Every field has a safe default so the package imports and unit-tests run with **no**
environment configured (and never touches the network at import time).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM judge (OpenAI-compatible client; Groq free tier by default) ---
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    judge_model: str = "llama-3.3-70b-versatile"

    # --- Storage ---
    database_url: str = "postgresql://evalgate:evalgate@localhost:5432/evalgate"

    # --- Calibration + CI gate ---
    anchor_set_path: str = "./anchors.jsonl"
    min_pass_rate: float = 0.9
    min_judge_kappa: float = 0.7


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so config is parsed once per process."""
    return Settings()
