"""
Application settings loaded from environment variables.

Uses pydantic-settings to validate and type-check all config values.
Every configurable parameter is centralized here. no scattered strings
across the codebase.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """All application configuration in one place."""

    # ── API Keys ────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")

    # ── Model Config ────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM for generation (development)"
    )
    llm_model_eval: str = Field(
        default="gpt-4o",
        description="LLM for evaluation runs (higher quality)"
    )

    # ── Vector Store ────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_data",
        description="Directory for ChromaDB persistent storage"
    )
    chroma_collection_name: str = Field(
        default="research_papers",
        description="ChromaDB collection name"
    )

    # ── Prompt Versioning ───────────────────────────────
    active_prompt_version: str = Field(
        default="v1",
        description="Which prompt version to use (e.g., v1, v2)"
    )
    prompts_dir: Path = Field(
        default=Path("prompts"),
        description="Root directory for versioned prompts"
    )

    # ── Ingestion Defaults ──────────────────────────────
    # These are defaults; the active prompt config can override them.
    default_chunk_size: int = Field(default=500, description="Chunk size in tokens")
    default_chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens")

    # ── Retrieval Defaults ──────────────────────────────
    default_top_k: int = Field(default=5, description="Number of chunks to retrieve")

    # ── API ─────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # ignore extra envs
    }


# Singleton instance
settings = Settings()