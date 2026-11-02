from pathlib import Path

import pytest

from research_assistant.rag.chunker import chunk_text
from research_assistant.rag.corpus import CorpusError, Paper, load_corpus, passages_from_papers


def test_chunks_overlap_and_stay_within_size() -> None:
    text = " ".join(f"token{i}" for i in range(250))
    chunks = chunk_text(text, size=800, overlap=100)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert chunks[0].startswith("token0")
    assert chunks[1][:24] in chunks[0]


def test_passage_ids_are_stable() -> None:
    paper = Paper(paper_id="demo-paper", title="Demo", text=" ".join(f"token{i}" for i in range(250)))
    passages = passages_from_papers([paper], chunk_size=800, chunk_overlap=100)

    assert len(passages) >= 2
    assert [passage.passage_id for passage in passages[:2]] == ["demo-paper#c0", "demo-paper#c1"]
    assert all(passage.title == "Demo" for passage in passages)


def test_empty_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="No .md or .txt papers"):
        load_corpus(tmp_path)


def test_title_is_read_from_heading(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# A Real Title\n\nBody of the note.\n", encoding="utf-8")
    papers = load_corpus(tmp_path)
    assert papers[0].paper_id == "note"
    assert papers[0].title == "A Real Title"
    assert "Body of the note." in papers[0].text
