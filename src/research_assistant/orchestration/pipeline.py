"""Pipeline with retrieval and a fact-check critic that may retry the summary once."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import uuid4

from agent_framework import Case, Default, Workflow, WorkflowBuilder
from agent_framework.gemini import GeminiChatClient, GeminiEmbeddingClient
from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions

from research_assistant.agents import (
    ContradictionAgent,
    FactCheckAgent,
    QueryAgent,
    RetrievalAgent,
    SummaryAgent,
    WriterAgent,
)
from research_assistant.agents.opening_agent import OpeningDraft
from research_assistant.agents.retrieval_agent import RetrievalError
from research_assistant.blackboard import RunBindings, bind_run, record
from research_assistant.config import Settings
from research_assistant.llm import Completer, FrameworkCompleter
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.rag.corpus import Paper, load_corpus, passages_from_papers
from research_assistant.rag.embedding_cache import cache_path, corpus_fingerprint, lookup_vectors, store_vectors
from research_assistant.rag.index import PassageIndex, corpus_covers

logger = logging.getLogger(__name__)

_INDEX_CACHE: dict[tuple[object, ...], tuple[list[Paper], PassageIndex]] = {}
_INDEX_LOCK = threading.Lock()


def clear_index_cache() -> None:
    """Drop the in-memory passage index. Tests use this between cases."""
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()


class EmbeddingVector(Protocol):
    vector: Sequence[float]


class EmbeddingClient(Protocol):
    """Subset of the Gemini embedding client used by retrieval."""

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: Mapping[str, str] | None = None,
    ) -> Sequence[EmbeddingVector]: ...


class QueryEmbedder:
    """Embeds a sub-query. A failure falls back to lexical retrieval."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self.embed_queries([text])
        return vectors[0]

    async def embed_queries(self, texts: list[str]) -> list[list[float] | None]:
        """One embedding request for every sub-query."""
        if not texts:
            return []
        try:
            result = await self._client.get_embeddings(list(texts), options={"task_type": "RETRIEVAL_QUERY"})
            return [[float(value) for value in item.vector] for item in result]
        except Exception as exc:
            logger.warning("Query embedding failed (%s); using lexical overlap.", exc)
            return [None] * len(texts)


def build_clients(settings: Settings) -> tuple[GeminiChatClient, GeminiEmbeddingClient]:
    settings.require_api_key()
    # One attempt per model. The SDK's default retries sit on a busy model for
    # minutes, and the completer already moves to the next model after a 503.
    chat_transport = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=1)),
    )
    return (
        GeminiChatClient(model=settings.chat_model, client=chat_transport),
        GeminiEmbeddingClient(api_key=settings.gemini_api_key, model=settings.embedding_model),
    )


def build_workflow(
    *,
    completer: Completer,
    papers: list[Paper],
    index: PassageIndex,
    top_k: int,
    revision_cap: int,
    embedder: QueryEmbedder | None = None,
) -> Workflow:
    """Query, retrieval, then summary and contradiction together, then fact-check.

    Fact-check sends the dossier back to summary alone when unsupported claims
    remain. That retry does not repeat the contradiction call. The writer
    publishes from the supported claims and does not call the model.
    """
    query = QueryAgent(completer, papers)
    retrieval = RetrievalAgent(index, top_k=top_k, embedder=embedder)
    summary = SummaryAgent(completer)
    opening = OpeningDraft(summary, ContradictionAgent(completer))
    factcheck = FactCheckAgent(completer, revision_cap=revision_cap)
    writer = WriterAgent()
    return (
        WorkflowBuilder(start_executor=query, output_from=[writer])
        .add_edge(query, retrieval)
        .add_edge(retrieval, opening)
        .add_edge(opening, factcheck)
        .add_edge(summary, factcheck)
        .add_switch_case_edge_group(
            factcheck,
            [
                Case(condition=lambda dossier: dossier.needs_revision, target=summary),
                Default(target=writer),
            ],
        )
        .build()
    )


async def prepare_corpus(
    settings: Settings,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[list[Paper], PassageIndex]:
    """Load papers and reuse embeddings when the corpus and model are unchanged."""
    key = (
        str(settings.corpus_dir.resolve()),
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        corpus_fingerprint(settings.corpus_dir),
        embedding_client is not None,
    )
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        papers, index, remember = await _load_index(settings, embedding_client)
        if remember:
            _INDEX_CACHE[key] = (papers, index)
        return papers, index


async def _load_index(
    settings: Settings,
    embedding_client: EmbeddingClient | None,
) -> tuple[list[Paper], PassageIndex, bool]:
    papers = load_corpus(settings.corpus_dir)
    passages = passages_from_papers(
        papers,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if embedding_client is None:
        return papers, PassageIndex(passages), True

    path = cache_path(settings.output_dir, settings.embedding_model)
    stored = lookup_vectors(path, passages)
    missing = [index for index, vector in enumerate(stored) if vector is None]
    if missing:
        try:
            result = await embedding_client.get_embeddings(
                [passages[index].text for index in missing],
                options={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            for index, item in zip(missing, result, strict=True):
                stored[index] = [float(value) for value in item.vector]
            store_vectors(path, passages, stored)
        except Exception as exc:
            logger.warning("Passage embeddings failed (%s); using lexical overlap.", exc)
            return papers, PassageIndex(passages), False
    embeddings = [vector for vector in stored if vector is not None]
    if len(embeddings) != len(passages):
        return papers, PassageIndex(passages), False
    return papers, PassageIndex(passages, embeddings), True


async def run_research(
    question: str,
    settings: Settings,
    *,
    completer: Completer | None = None,
    papers: list[Paper] | None = None,
    index: PassageIndex | None = None,
    embedder: QueryEmbedder | None = None,
) -> ResearchDossier:
    """Run the workflow. Tests inject a completer and index so no network is used."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("A research question is required.")

    embedding_client: EmbeddingClient | None = None
    if completer is None or papers is None or index is None:
        if completer is None:
            chat_client, embedding_client = build_clients(settings)
            completer = FrameworkCompleter(chat_client)
            embedder = QueryEmbedder(embedding_client)
        if papers is None or index is None:
            papers, index = await prepare_corpus(settings, embedding_client)

    dossier = ResearchDossier(question=cleaned)
    bindings = RunBindings(
        dossier=dossier,
        papers=papers,
        index=index,
        embedder=embedder,
        top_k=settings.top_k,
        revision_cap=settings.revision_cap,
        log_path=settings.log_path,
        run_id=uuid4().hex[:8],
    )
    with bind_run(bindings):
        if index is None or not corpus_covers(cleaned, index):
            record(
                "pipeline",
                "corpus_covers",
                "These papers do not cover that question. The assistant will not invent sources.",
                ok=False,
            )
            raise RetrievalError(
                "These papers do not cover that question. The assistant will not invent sources."
            )

        workflow = build_workflow(
            completer=completer,
            papers=papers,
            index=index,
            top_k=settings.top_k,
            revision_cap=settings.revision_cap,
            embedder=embedder,
        )
        result = await workflow.run(dossier)
    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("The workflow finished without a literature summary.")
    return outputs[0]
