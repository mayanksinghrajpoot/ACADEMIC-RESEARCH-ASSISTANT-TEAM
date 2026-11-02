"""Retrieval agent: the only step that reads the passage index."""

from __future__ import annotations

from typing import Protocol

from agent_framework import Executor, WorkflowContext, handler

from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.rag.index import PassageIndex


class QueryEmbedder(Protocol):
    async def embed_query(self, text: str) -> list[float] | None: ...


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
        for query in dossier.sub_queries:
            vector = await self._embedder.embed_query(query) if self._embedder is not None else None
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
