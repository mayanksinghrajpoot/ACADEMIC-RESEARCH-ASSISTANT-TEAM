"""Hybrid passage index: embedding cosine blended with lexical overlap."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from research_assistant.memory.dossier import Passage

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class PassageIndex:
    """Read-only retrieval memory for one run.

    When document vectors are present and the caller supplies a query vector,
    the score is ``0.65 * max(cosine, 0) + 0.35 * lexical``. A negative or
    zero cosine adds nothing, so a passage with no word overlap and no
    positive similarity scores zero and is dropped. Otherwise ranking is
    lexical only, which still works on a small corpus.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.passages = list(passages)
        if embeddings is not None and len(embeddings) != len(self.passages):
            raise ValueError(
                f"Expected {len(self.passages)} embeddings, received {len(embeddings)}."
            )
        self.embeddings = [list(vector) for vector in embeddings] if embeddings is not None else None

    def search(
        self,
        query: str,
        k: int,
        query_embedding: Sequence[float] | None = None,
    ) -> list[tuple[Passage, float]]:
        """Return up to ``k`` passages with a positive score, best first."""
        if k < 1:
            raise ValueError("k must be >= 1")
        ranked = [
            (passage, score)
            for passage, score in self._scored(query, query_embedding)
            if score > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].passage_id))
        return ranked[:k]

    def _scored(
        self,
        query: str,
        query_embedding: Sequence[float] | None,
    ) -> list[tuple[Passage, float]]:
        query_tokens = tokenize(query)
        use_vectors = (
            self.embeddings is not None
            and query_embedding is not None
            and len(query_embedding) > 0
        )
        scored: list[tuple[Passage, float]] = []
        for index, passage in enumerate(self.passages):
            lexical = lexical_overlap(query_tokens, tokenize(passage.text))
            if use_vectors:
                assert self.embeddings is not None
                cosine = cosine_similarity(query_embedding, self.embeddings[index])
                score = 0.65 * max(cosine, 0.0) + 0.35 * lexical
            else:
                score = lexical
            scored.append((passage, score))
        return scored


def corpus_covers(question: str, index: PassageIndex) -> bool:
    """True when the question shares a content word with a stored passage.

    This runs before any chat call. A question the papers never mention stops
    here instead of paying for the agent chain.
    """
    query_tokens = tokenize(question)
    if not query_tokens:
        return False
    for passage in index.passages:
        if query_tokens & tokenize(f"{passage.title} {passage.text}"):
            return True
    return False


def tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1}


def lexical_overlap(query_tokens: set[str], passage_tokens: set[str]) -> float:
    """Fraction of query terms that appear in the passage."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right, strict=True):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
