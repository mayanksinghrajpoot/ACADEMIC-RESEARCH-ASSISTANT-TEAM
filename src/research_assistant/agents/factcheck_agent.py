"""Fact-check agent: the critic. Verdicts are gated by retrieved passages."""

from __future__ import annotations

from typing import Literal

from agent_framework import Executor, WorkflowContext, handler
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
from research_assistant.grounding import mark_revision, merge_verdicts, partition_claims
from research_assistant.llm import Completer
from research_assistant.memory.dossier import Claim, ResearchDossier, Verdict

_INSTRUCTIONS = """You fact-check literature claims against the retrieved passages only.
Mark a claim supported only when a cited passage entails it, and copy a contiguous
quote of at least 20 characters from that passage into supporting_quote.
If the passage does not entail the claim, mark it unsupported and explain why.
Do not use outside knowledge. Do not invent a quote.
"""


class VerdictDraft(BaseModel):
    claim_id: str
    verdict: Literal["supported", "unsupported"]
    reason: str
    supporting_quote: str = ""


class FactCheckDraft(BaseModel):
    verdicts: list[VerdictDraft] = Field(default_factory=list)


class FactCheckAgent(Executor):
    """Verifies each claim and flags a single summary revision when needed."""

    def __init__(self, completer: Completer, *, revision_cap: int = 1) -> None:
        super().__init__(id="factcheck")
        self._completer = completer
        self._revision_cap = revision_cap

    @handler
    async def check(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        eligible, pre_rejected = partition_claims(dossier)
        model_verdicts: list[Verdict] = []
        if eligible:
            draft = await self._completer.complete(
                name="factcheck",
                instructions=_INSTRUCTIONS,
                prompt=_prompt(dossier, eligible),
                response_format=FactCheckDraft,
            )
            model_verdicts = [
                Verdict(
                    claim_id=item.claim_id,
                    verdict=item.verdict,
                    reason=item.reason.strip() or "No reason given.",
                    supporting_quote=item.supporting_quote.strip(),
                )
                for item in draft.verdicts
            ]
        verdicts = merge_verdicts(dossier, model_verdicts, pre_rejected)
        updated = mark_revision(
            dossier.model_copy(update={"verdicts": verdicts}),
            self._revision_cap,
        )
        await ctx.send_message(updated)


def _prompt(dossier: ResearchDossier, claims: list[Claim]) -> str:
    claim_lines = "\n".join(
        f"- {claim.claim_id} [{', '.join(claim.passage_ids)}] ({claim.theme}): {claim.text}"
        for claim in claims
    )
    return (
        f"Research question:\n{dossier.question}\n\n"
        f"Retrieved passages:\n{format_passages(dossier.passages)}\n\n"
        f"Claims to check:\n{claim_lines}\n\n"
        "Return JSON with a verdicts array. Use the claim_id values above. "
        "supporting_quote must be copied from a cited passage when the verdict is supported."
    )
