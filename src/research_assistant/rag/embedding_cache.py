"""Remember passage embeddings so the next question does not embed the corpus again."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research_assistant.memory.dossier import Passage


def corpus_fingerprint(directory: Path) -> tuple[tuple[str, int, int], ...]:
    """File name, mtime, and size. A changed paper misses the cache."""
    files = sorted([*directory.glob("*.md"), *directory.glob("*.txt")])
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in files)


def cache_path(output_dir: Path, model: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-._" else "_" for character in model)
    return output_dir / f"embedding-cache-{safe}.json"


def lookup_vectors(path: Path, passages: list[Passage]) -> list[list[float] | None]:
    """Return a vector for each passage, or None where this cache has no match."""
    stored = _read(path)
    return [stored.get(_key(passage)) for passage in passages]


def store_vectors(path: Path, passages: list[Passage], vectors: list[list[float] | None]) -> None:
    stored = _read(path)
    for passage, vector in zip(passages, vectors, strict=True):
        if vector is not None:
            stored[_key(passage)] = vector
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored), encoding="utf-8")


def _key(passage: Passage) -> str:
    digest = hashlib.sha256(passage.text.encode("utf-8")).hexdigest()
    return f"{passage.passage_id}:{digest}"


def _read(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    vectors: dict[str, list[float]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, list) and value:
            vectors[key] = [float(item) for item in value]
    return vectors
