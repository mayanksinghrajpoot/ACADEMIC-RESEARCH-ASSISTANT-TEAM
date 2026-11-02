"""Environment-backed settings for the research assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration. Paths are resolved relative to the project root."""

    openai_api_key: str = ""
    openai_base_url: str | None = None
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    corpus_dir: Path = Field(default_factory=lambda: Path("corpus"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))
    top_k: int = 4
    revision_cap: int = 1
    chunk_size: int = 800
    chunk_overlap: int = 100

    def require_api_key(self) -> None:
        if not self.openai_api_key.strip():
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to the environment or a .env file "
                "before running the assistant. The pipeline will not invent sources."
            )


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


def load_settings(project_root: Path | None = None) -> Settings:
    """Load settings from the environment, with a project ``.env`` as fallback."""
    root = project_root or Path.cwd()
    load_dotenv(root / ".env", override=False)

    base_url = os.getenv("OPENAI_BASE_URL") or None
    corpus = Path(os.getenv("CORPUS_DIR", "corpus"))
    output = Path(os.getenv("OUTPUT_DIR", "outputs"))
    if not corpus.is_absolute():
        corpus = root / corpus
    if not output.is_absolute():
        output = root / output

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=base_url,
        chat_model=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        corpus_dir=corpus,
        output_dir=output,
        top_k=_positive_int("TOP_K", 4),
        revision_cap=_positive_int("REVISION_CAP", 1),
        chunk_size=_positive_int("CHUNK_SIZE", 800),
        chunk_overlap=_non_negative_int("CHUNK_OVERLAP", 100),
    )


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ConfigurationError(f"{name} must be >= 1, got {value}.")
    return value


def _non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ConfigurationError(f"{name} must be >= 0, got {value}.")
    return value
