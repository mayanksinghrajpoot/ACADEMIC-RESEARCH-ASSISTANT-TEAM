"""Contradiction agent: report where two retrieved sources disagree."""

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
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


class ContradictionAgent(Executor):
    """Finds opposed positions and drops any pair that is not grounded."""

    def __init__(self, completer: Completer) -> None:
        super().__init__(id="contradiction")
        self._completer = completer

    async def apply(self, dossier: ResearchDossier) -> ResearchDossier:
        """Find disagreements. Depends only on the retrieved passages."""
        prompt = (
            f"Research question:\n{dossier.question}\n\n"
            f"Retrieved passages:\n{format_passages(dossier.passages)}\n\n"
            "Return JSON with a contradictions array. Each item has topic, position_a, "
            "passage_ids_a, quote_a, position_b, passage_ids_b, and quote_b. "
            "Copy each quote from the passage ids on that side."
        )
        draft = await self._completer.complete(
            name="contradiction",
            instructions=_INSTRUCTIONS,
            prompt=prompt,
            response_format=ContradictionReport,
        )
        proposed = [
            Contradiction(
                topic=item.topic.strip() or "Disagreement",
                position_a=" ".join(item.position_a.split()),
                passage_ids_a=list(dict.fromkeys(item.passage_ids_a)),
                quote_a=" ".join(item.quote_a.split()),
                position_b=" ".join(item.position_b.split()),
                passage_ids_b=list(dict.fromkeys(item.passage_ids_b)),
                quote_b=" ".join(item.quote_b.split()),
            )
            for item in draft.contradictions
            if item.position_a.strip() and item.position_b.strip()
        ]
        kept, note = grounded_contradictions(dossier, proposed)
        return dossier.model_copy(update={"contradictions": kept, "contradiction_note": note})

    @handler
    async def compare(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        await ctx.send_message(await self.apply(dossier))
