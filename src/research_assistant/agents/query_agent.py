"""Query agent: turn a research question into corpus sub-queries."""

from __future__ import annotations

import json

from agent_framework import Executor, WorkflowContext, handler, tool
from pydantic import BaseModel, Field

from research_assistant.blackboard import get_bindings, invoke, use_shared
from research_assistant.llm import Completer
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.rag.corpus import Paper

_INSTRUCTIONS = """You write search sub-queries for a closed set of research papers.
Use only the research question and the paper titles. Do not answer the question.
Do not search the open web. Return between 3 and 5 short sub-queries that could
match passages in this paper set, including wording that might surface a disagreement.
"""


class QueryDraft(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


@tool
def list_corpus_titles() -> str:
    """Read the closed paper titles for this run. Does not search the web."""
    papers = get_bindings().papers
    if not papers:
        return "(no papers)"
    return "\n".join(f"- {paper.title}" for paper in papers)


@tool
def commit_subqueries(raw_json: str) -> str:
    """Normalize 3 to 5 sub-queries and write them on the shared dossier."""
    dossier = get_bindings().dossier
    raw = json.loads(raw_json)
    if not isinstance(raw, list):
        raise ValueError("sub-queries must be a JSON list.")
    dossier.sub_queries = normalize_subqueries(dossier.question, [str(item) for item in raw])
    return " | ".join(dossier.sub_queries)


class QueryAgent(Executor):
    """Produces 3–5 sub-queries over the provided paper set."""

    def __init__(self, completer: Completer, papers: list[Paper]) -> None:
        super().__init__(id="query")
        self._completer = completer
        self._papers = papers

    @handler
    async def expand(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        dossier = use_shared(dossier, papers=self._papers)
        titles = await invoke("query", list_corpus_titles)
        prompt = (
            f"Research question:\n{dossier.question}\n\n"
            f"Paper titles:\n{titles}\n\n"
            "Return JSON with a sub_queries array of 3 to 5 strings."
        )
        draft = await self._completer.complete(
            name="query",
            instructions=_INSTRUCTIONS,
            prompt=prompt,
            response_format=QueryDraft,
        )
        await invoke("query", commit_subqueries, raw_json=json.dumps(draft.sub_queries))
        await ctx.send_message(dossier)


def normalize_subqueries(question: str, raw: list[str]) -> list[str]:
    """Keep at most five unique queries, and pad to three with corpus-local fallbacks."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = " ".join(item.split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == 5:
            break
    fallbacks = [
        question.strip(),
        f"evidence that supports: {question.strip()}",
        f"evidence that disagrees: {question.strip()}",
    ]
    for item in fallbacks:
        if len(cleaned) >= 3:
            break
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)
    if not cleaned:
        cleaned.append(question.strip() or "research question")
    return cleaned[:5]
