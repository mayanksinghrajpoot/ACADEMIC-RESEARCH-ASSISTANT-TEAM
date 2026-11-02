"""Load a directory of markdown or text papers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from research_assistant.memory.dossier import Passage
from research_assistant.rag.chunker import chunk_text


class CorpusError(RuntimeError):
    """Raised when the paper set cannot be read."""


class Paper(BaseModel):
    paper_id: str
    title: str
    text: str


def load_corpus(directory: Path) -> list[Paper]:
    """Load every ``.md`` and ``.txt`` paper, sorted by filename."""
    if not directory.is_dir():
        raise CorpusError(f"Corpus directory does not exist: {directory}")
    files = sorted([*directory.glob("*.md"), *directory.glob("*.txt")])
    if not files:
        raise CorpusError(
            f"No .md or .txt papers found in {directory}. "
            "Add a paper set before asking a research question."
        )
    return [_read_paper(path) for path in files]


def passages_from_papers(
    papers: list[Paper],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Passage]:
    """Chunk each paper and assign stable passage ids."""
    passages: list[Passage] = []
    for paper in papers:
        chunks = chunk_text(paper.text, chunk_size, chunk_overlap)
        if not chunks:
            raise CorpusError(f"Paper produced no chunks: {paper.paper_id}")
        for index, chunk in enumerate(chunks):
            passages.append(
                Passage(
                    passage_id=f"{paper.paper_id}#c{index}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    text=chunk,
                )
            )
    return passages


def _read_paper(path: Path) -> Paper:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise CorpusError(f"Paper is empty: {path.name}")
    lines = raw.splitlines()
    title = path.stem.replace("-", " ")
    body = raw
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip() or title
        body = "\n".join(lines[1:]).strip()
    if not body:
        raise CorpusError(f"Paper has no body: {path.name}")
    return Paper(paper_id=path.stem, title=title, text=body)
