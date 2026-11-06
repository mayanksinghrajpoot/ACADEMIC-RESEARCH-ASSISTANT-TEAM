"""Retrieval agent: the only step that reads the passage index."""

from __future__ import annotations

import asyncio
from typing import Protocol

from agent_framework import Executor, WorkflowContext, handler

from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.rag.index import PassageIndex


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float] | None: ...

    async def embed_queries(self, texts: list[str]) -> list[list[float] | None]: ...


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot ground the question in the corpus."""


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
        if not dossier.sub_queries:
            raise RetrievalError(
                "The query agent produced no sub-queries, so retrieval cannot run."
            )
        best: dict[str, tuple[Passage, float]] = {}
        vectors = await self._query_vectors(dossier.sub_queries)
        for query, vector in zip(dossier.sub_queries, vectors, strict=True):
            for passage, score in self._index.search(query, self._top_k, vector):
                current = best.get(passage.passage_id)
                if current is None or score > current[1]:
                    best[passage.passage_id] = (passage, score)
        if not best:
            raise RetrievalError(
                "Retrieval found no passages for the sub-queries. "
                "The assistant will not invent sources."
            )
        ordered = [item[0] for item in sorted(best.values(), key=lambda item: (-item[1], item[0].passage_id))]
        await ctx.send_message(dossier.model_copy(update={"passages": ordered}))

    async def _query_vectors(self, queries: list[str]) -> list[list[float] | None]:
        """Embed every sub-query in one request when the client supports it."""
        if self._embedder is None:
            return [None] * len(queries)
        embed_queries = getattr(self._embedder, "embed_queries", None)
        if embed_queries is not None:
            vectors = await embed_queries(queries)
            if len(vectors) == len(queries):
                return list(vectors)
        return list(await asyncio.gather(*(self._embedder.embed_query(query) for query in queries)))
