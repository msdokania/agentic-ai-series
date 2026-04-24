"""
Versioned prompt configuration loader.

Why version prompts?
In RAG systems, prompt wording has a huge impact on output quality.
Small changes, like adding "cite your sources" or changing the system role,
can shift faithfulness scores by 20-30%. Without versioning, you can't
reproduce results or track which change caused which improvement.

Every query logs which prompt version was used, making evaluation
results fully reproducible.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

from src.config.settings import settings

console = Console()


@dataclass
class PromptConfig:
    """
    A complete prompt configuration for one version.
    
    Contains all prompts, model parameters, and retrieval settings
    so that a single version string fully defines the system's behavior.
    """
    version: str
    description: str

    # Prompts
    system_prompt: str
    retrieval_template: str
    no_context_response: str = "I don't have enough information in the provided documents to answer this question."

    # Model parameters
    temperature: float = 0.1
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 1024

    # Retrieval parameters
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Metadata
    embedding_model: str = "text-embedding-3-small"
    notes: str = ""


def load_prompt_config(version: Optional[str] = None) -> PromptConfig:
    """
    Load a prompt configuration by version.
    
    Args:
        version: Prompt version to load (e.g., "v1"). 
                 Defaults to the ACTIVE_PROMPT_VERSION env var.
    
    Returns:
        PromptConfig with all prompts and parameters for this version.
    
    Raises:
        FileNotFoundError: If the version directory or config.yaml doesn't exist.
        ValueError: If the YAML is malformed or missing required fields.
    """
    version = version or settings.active_prompt_version
    config_path = settings.prompts_dir / version / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Prompt config not found: {config_path}\n"
            f"Create {config_path} with your prompt configuration."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(raw)}")

    # Validate required fields
    required = ["version", "description", "system_prompt", "retrieval_template"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing required fields in {config_path}: {missing}")

    config = PromptConfig(**raw)
    console.print(f"[dim]Loaded prompt config: {config.version} — {config.description}[/dim]")
    return config