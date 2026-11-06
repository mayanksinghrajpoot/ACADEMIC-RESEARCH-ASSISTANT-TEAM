"""Contradiction agent: report where two retrieved sources disagree."""

from __future__ import annotations

import json

from agent_framework import Executor, WorkflowContext, handler, tool
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
from research_assistant.blackboard import get_bindings, invoke, use_shared
from research_assistant.grounding import grounded_contradictions
from research_assistant.llm import Completer
from research_assistant.memory.dossier import Contradiction, ResearchDossier

_INSTRUCTIONS = """You compare retrieved passages and report direct disagreements.
A disagreement means two passages take opposed positions on the same point.
Each position must cite passage ids from the prompt, and the two sides must use
different ids. Copy quote_a from a passage cited by side A and quote_b from a
passage cited by side B. Each quote must be a contiguous span of at least 20
characters. If the passages do not disagree, return an empty list.
Do not invent a dispute, and do not use knowledge from outside the passages.
"""


class ContradictionDraft(BaseModel):
    topic: str
    position_a: str
    passage_ids_a: list[str] = Field(default_factory=list)
    quote_a: str = ""
    position_b: str
    passage_ids_b: list[str] = Field(default_factory=list)
    quote_b: str = ""


class ContradictionReport(BaseModel):
    contradictions: list[ContradictionDraft] = Field(default_factory=list)


@tool
def read_passages_for_contradictions() -> str:
    """Read the passages stored on the shared dossier."""
    return format_passages(get_bindings().dossier.passages) or "(no passages)"


@tool
def commit_contradictions(proposed_json: str) -> str:
    """Keep only disagreements whose quotes are copied from the cited passages."""
    dossier = get_bindings().dossier
    raw = json.loads(proposed_json)
    if not isinstance(raw, list):
        raise ValueError("contradictions must be a JSON list.")
    proposed: list[Contradiction] = []
    for item in raw:
        position_a = " ".join(str(item.get("position_a", "")).split())
        position_b = " ".join(str(item.get("position_b", "")).split())
        if not position_a or not position_b:
            continue
        proposed.append(
            Contradiction(
                topic=str(item.get("topic", "")).strip() or "Disagreement",
                position_a=position_a,
                passage_ids_a=list(dict.fromkeys(str(value) for value in item.get("passage_ids_a") or [])),
                quote_a=" ".join(str(item.get("quote_a", "")).split()),
                position_b=position_b,
                passage_ids_b=list(dict.fromkeys(str(value) for value in item.get("passage_ids_b") or [])),
                quote_b=" ".join(str(item.get("quote_b", "")).split()),
            )
        )
    kept, note = grounded_contradictions(dossier, proposed)
    dossier.contradictions = kept
    dossier.contradiction_note = note
    if not kept:
        return note or "no contradiction"
    return f"{len(kept)} contradictions"


class ContradictionAgent(Executor):
    """Finds opposed positions and drops any pair that is not grounded."""

    def __init__(self, completer: Completer) -> None:
        super().__init__(id="contradiction")
        self._completer = completer

    async def apply(self, dossier: ResearchDossier) -> ResearchDossier:
        """Find disagreements on the shared dossier. Depends only on retrieved passages."""
        dossier = use_shared(dossier)
        passages = await invoke("contradiction", read_passages_for_contradictions)
        draft = await self._completer.complete(
            name="contradiction",
            instructions=_INSTRUCTIONS,
            prompt=(
                f"Research question:\n{dossier.question}\n\n"
                f"Retrieved passages:\n{passages}\n\n"
                "Return JSON with a contradictions array. Each item has topic, position_a, "
                "passage_ids_a, quote_a, position_b, passage_ids_b, and quote_b. "
                "Copy each quote from the passage ids on that side."
            ),
            response_format=ContradictionReport,
        )
        payload = [item.model_dump() for item in draft.contradictions]
        await invoke("contradiction", commit_contradictions, proposed_json=json.dumps(payload))
        return dossier

    @handler
    async def compare(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        await ctx.send_message(await self.apply(dossier))
