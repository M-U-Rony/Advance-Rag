"""Application settings, loaded from environment variables (or a ``.env`` file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. See ``.env.example`` for documentation of each variable."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["none", "openai", "openai_compatible", "anthropic"] = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    embedding_backend: Literal["auto", "dense", "tfidf"] = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    data_dir: Path = Field(default=Path("./data"))
    static_dir: Path = Field(default=REPO_ROOT / "backend" / "static")
    samples_dir: Path = Field(default=REPO_ROOT / "samples")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 50

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
