"""Pipeline with retrieval and a fact-check critic that may retry the summary once."""

from __future__ import annotations

import logging

from agent_framework import Case, Default, Workflow, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient, OpenAIEmbeddingClient

from research_assistant.agents import (
    ContradictionAgent,
    FactCheckAgent,
    QueryAgent,
    RetrievalAgent,
    SummaryAgent,
    WriterAgent,
)
from research_assistant.config import Settings
from research_assistant.llm import Completer, FrameworkCompleter
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.rag.corpus import Paper, load_corpus, passages_from_papers
from research_assistant.rag.index import PassageIndex

logger = logging.getLogger(__name__)


class OpenAIQueryEmbedder:
    """Embeds a sub-query. A failure falls back to lexical retrieval."""

    def __init__(self, client: OpenAIEmbeddingClient) -> None:
        self._client = client

    async def embed_query(self, text: str) -> list[float] | None:
        try:
            result = await self._client.get_embeddings([text])
            return [float(value) for value in result[0].vector]
        except Exception as exc:
            logger.warning("Query embedding failed (%s); using lexical overlap.", exc)
            return None


def build_clients(settings: Settings) -> tuple[OpenAIChatClient, OpenAIEmbeddingClient]:
    settings.require_api_key()
    chat_kwargs: dict[str, str] = {
        "model": settings.chat_model,
        "api_key": settings.openai_api_key,
    }
    embed_kwargs: dict[str, str] = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        chat_kwargs["base_url"] = settings.openai_base_url
        embed_kwargs["base_url"] = settings.openai_base_url
    return OpenAIChatClient(**chat_kwargs), OpenAIEmbeddingClient(**embed_kwargs)


def build_workflow(
    *,
    completer: Completer,
    papers: list[Paper],
    index: PassageIndex,
    top_k: int,
    revision_cap: int,
    embedder: OpenAIQueryEmbedder | None = None,
) -> Workflow:
    """Query, retrieval, summary, contradiction, fact-check, then writer.

    Fact-check sends the dossier back to summary when unsupported claims remain
    and the revision cap has not been used. The next pass goes to the writer.
    """
    query = QueryAgent(completer, papers)
    retrieval = RetrievalAgent(index, top_k=top_k, embedder=embedder)
    summary = SummaryAgent(completer)
    contradiction = ContradictionAgent(completer)
    factcheck = FactCheckAgent(completer, revision_cap=revision_cap)
    writer = WriterAgent(completer)
    return (
        WorkflowBuilder(start_executor=query, output_from=[writer])
        .add_edge(query, retrieval)
        .add_edge(retrieval, summary)
        .add_edge(summary, contradiction)
        .add_edge(contradiction, factcheck)
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
    embedding_client: OpenAIEmbeddingClient | None = None,
) -> tuple[list[Paper], PassageIndex]:
    """Load papers, chunk them, and attach embeddings when the client succeeds."""
    papers = load_corpus(settings.corpus_dir)
    passages = passages_from_papers(
        papers,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings: list[list[float]] | None = None
    if embedding_client is not None:
        try:
            result = await embedding_client.get_embeddings([passage.text for passage in passages])
            embeddings = [[float(value) for value in item.vector] for item in result]
        except Exception as exc:
            logger.warning("Passage embeddings failed (%s); using lexical overlap.", exc)
            embeddings = None
    return papers, PassageIndex(passages, embeddings)


async def run_research(
    question: str,
    settings: Settings,
    *,
    completer: Completer | None = None,
    papers: list[Paper] | None = None,
    index: PassageIndex | None = None,
    embedder: OpenAIQueryEmbedder | None = None,
) -> ResearchDossier:
    """Run the workflow. Tests inject a completer and index so no network is used."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("A research question is required.")

    embedding_client: OpenAIEmbeddingClient | None = None
    if completer is None or papers is None or index is None:
        if completer is None:
            chat_client, embedding_client = build_clients(settings)
            completer = FrameworkCompleter(chat_client)
            embedder = OpenAIQueryEmbedder(embedding_client)
        if papers is None or index is None:
            papers, index = await prepare_corpus(settings, embedding_client)

    workflow = build_workflow(
        completer=completer,
        papers=papers,
        index=index,
        top_k=settings.top_k,
        revision_cap=settings.revision_cap,
        embedder=embedder,
    )
    result = await workflow.run(ResearchDossier(question=cleaned))
    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("The workflow finished without a literature summary.")
    return outputs[0]
