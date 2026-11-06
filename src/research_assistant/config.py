"""Environment-backed settings for the research assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration. Paths are resolved relative to the project root."""

    gemini_api_key: str = ""
    chat_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-2"
    corpus_dir: Path = Field(default_factory=lambda: Path("corpus"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))
    top_k: int = 4
    revision_cap: int = 1
    chunk_size: int = 800
    chunk_overlap: int = 100

    def require_api_key(self) -> None:
        if not self.gemini_api_key.strip():
            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Add it to the environment or a .env file "
                "(GEMINI_API_KEY is also accepted) before running the assistant. "
                "The pipeline will not invent sources."
            )


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


def load_settings(project_root: Path | None = None) -> Settings:
    """Load settings from the environment, with a project ``.env`` as fallback."""
    root = project_root or Path.cwd()
    load_dotenv(root / ".env", override=False)

    corpus = Path(os.getenv("CORPUS_DIR", "corpus"))
    output = Path(os.getenv("OUTPUT_DIR", "outputs"))
    if not corpus.is_absolute():
        corpus = root / corpus
    if not output.is_absolute():
        output = root / output

    chunk_size = _positive_int("CHUNK_SIZE", 800)
    chunk_overlap = _non_negative_int("CHUNK_OVERLAP", 100)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError(
            f"CHUNK_OVERLAP must be smaller than CHUNK_SIZE, got overlap {chunk_overlap} and size {chunk_size}."
        )

    return Settings(
        gemini_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "",
        chat_model=os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_CHAT_MODEL") or "gemini-3.8-flash",
        embedding_model=(
            os.getenv("GOOGLE_EMBEDDING_MODEL")
            or os.getenv("GEMINI_EMBEDDING_MODEL")
            or "gemini-embedding-2"
        ),
        corpus_dir=corpus,
        output_dir=output,
        top_k=_positive_int("TOP_K", 4),
        revision_cap=_positive_int("REVISION_CAP", 1),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _positive_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value < 1:
        raise ConfigurationError(f"{name} must be >= 1, got {value}.")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value < 0:
        raise ConfigurationError(f"{name} must be >= 0, got {value}.")
    return value


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
