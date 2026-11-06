import pytest

from research_assistant.memory.dossier import Passage
from research_assistant.rag.index import PassageIndex, cosine_similarity


def _passage(passage_id: str, text: str) -> Passage:
    return Passage(passage_id=passage_id, paper_id=passage_id, title=passage_id, text=text)


def test_lexical_search_ranks_the_matching_passage_first() -> None:
    index = PassageIndex(
        [
            _passage("other", "passage about battery electrolytes"),
            _passage("audit", "passage identifiers make an audit of a literature claim"),
        ]
    )
    hits = index.search("passage identifiers audit", k=2)
    assert hits[0][0].passage_id == "audit"
    assert hits[0][1] > hits[1][1]


def test_embeddings_can_outrank_a_stronger_lexical_hit() -> None:
    lexical = PassageIndex(
        [
            _passage("aligned", "alpha catalog"),
            _passage("lexical", "alpha beta"),
        ]
    )
    hybrid = PassageIndex(
        lexical.passages,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    query = "alpha beta"

    assert lexical.search(query, k=1)[0][0].passage_id == "lexical"
    assert hybrid.search(query, k=1, query_embedding=[1.0, 0.0])[0][0].passage_id == "aligned"


def test_cosine_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_unrelated_embeddings_do_not_force_a_hit() -> None:
    index = PassageIndex(
        [_passage("a", "catalysis recall bottleneck")],
        embeddings=[[0.0, 1.0]],
    )
    assert index.search("unrelated zebrafish sonnet", k=3, query_embedding=[1.0, 0.0]) == []


def test_zero_overlap_returns_nothing() -> None:
    index = PassageIndex([_passage("a", "catalysis recall bottleneck")])
    assert index.search("unrelated zebrafish sonnet", k=3) == []
