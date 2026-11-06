"""Retrieval agent: the only step that reads the passage index."""

from __future__ import annotations

import asyncio
from typing import Protocol

from agent_framework import Executor, WorkflowContext, handler, tool

from research_assistant.blackboard import get_bindings, invoke, use_shared
from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.rag.index import PassageIndex


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float] | None: ...

    async def embed_queries(self, texts: list[str]) -> list[list[float] | None]: ...


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot ground the question in the corpus."""


@tool
async def embed_subqueries() -> str:
    """Embed every sub-query. On failure, leave lexical ranking in place."""
    bindings = get_bindings()
    queries = list(bindings.dossier.sub_queries)
    if not queries:
        bindings.query_vectors = []
        return "no sub-queries"
    embedder = bindings.embedder
    if embedder is None:
        bindings.query_vectors = [None] * len(queries)
        return "lexical fallback"
    try:
        embed_queries = getattr(embedder, "embed_queries", None)
        if embed_queries is not None:
            vectors = await embed_queries(queries)
            if len(vectors) != len(queries):
                vectors = list(await asyncio.gather(*(embedder.embed_query(query) for query in queries)))
        else:
            vectors = list(await asyncio.gather(*(embedder.embed_query(query) for query in queries)))
    except Exception:
        bindings.query_vectors = [None] * len(queries)
        return "lexical fallback"
    bindings.query_vectors = list(vectors)
    if all(vector is None for vector in vectors):
        return "lexical fallback"
    ready = sum(vector is not None for vector in vectors)
    return f"embedded {ready} of {len(queries)}"


@tool
def store_retrieved_passages() -> str:
    """Search the passage index and write the hits onto the shared dossier."""
    bindings = get_bindings()
    dossier = bindings.dossier
    if not dossier.sub_queries:
        raise RetrievalError("The query agent produced no sub-queries, so retrieval cannot run.")
    if bindings.query_vectors is None:
        raise RetrievalError("embed_subqueries must run before the passage search.")
    index = bindings.index
    if not isinstance(index, PassageIndex):
        raise RetrievalError("No passage index is loaded.")
    vectors = bindings.query_vectors
    if len(vectors) != len(dossier.sub_queries):
        vectors = [None] * len(dossier.sub_queries)
    best: dict[str, tuple[Passage, float]] = {}
    for query, vector in zip(dossier.sub_queries, vectors, strict=True):
        for passage, score in index.search(query, bindings.top_k, vector):
            current = best.get(passage.passage_id)
            if current is None or score > current[1]:
                best[passage.passage_id] = (passage, score)
    if not best:
        raise RetrievalError(
            "Retrieval found no passages for the sub-queries. "
            "The assistant will not invent sources."
        )
    ordered = [item[0] for item in sorted(best.values(), key=lambda item: (-item[1], item[0].passage_id))]
    dossier.passages = ordered
    return f"{len(ordered)} passages: " + ", ".join(passage.passage_id for passage in ordered)


class RetrievalAgent(Executor):
    """Fetches relevant passages for each sub-query and stores them on the dossier."""

    def __init__(
        self,
        index: PassageIndex,
        *,
        top_k: int = 4,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        super().__init__(id="retrieval")
        self._index = index
        self._top_k = top_k
        self._embedder = embedder

    @handler
    async def retrieve(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        dossier = use_shared(
            dossier,
            index=self._index,
            embedder=self._embedder,
            top_k=self._top_k,
            query_vectors=None,
        )
        await invoke("retrieval", embed_subqueries)
        await invoke("retrieval", store_retrieved_passages)
        await ctx.send_message(dossier)
