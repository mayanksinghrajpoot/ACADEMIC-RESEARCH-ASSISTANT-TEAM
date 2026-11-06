"""Fact-check agent: the critic. Verdicts are gated by retrieved passages."""

from __future__ import annotations

import json
from typing import Literal

from agent_framework import Executor, WorkflowContext, handler, tool
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
from research_assistant.blackboard import get_bindings, invoke, use_shared
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


@tool
def select_checkable_claims() -> str:
    """Split claims into those a model may judge and those the passage gate already rejects."""
    eligible, rejected = partition_claims(get_bindings().dossier)
    return json.dumps(
        {
            "eligible": [claim.model_dump() for claim in eligible],
            "pre_rejected": [verdict.model_dump() for verdict in rejected],
        }
    )


@tool
def commit_verdicts(review_json: str, verdicts_json: str) -> str:
    """Apply the quote gate and write verdicts on the shared dossier."""
    bindings = get_bindings()
    dossier = bindings.dossier
    review = json.loads(review_json)
    proposed = json.loads(verdicts_json)
    model_verdicts = [Verdict.model_validate(item) for item in proposed]
    pre_rejected = [Verdict.model_validate(item) for item in review.get("pre_rejected", [])]
    dossier.verdicts = merge_verdicts(dossier, model_verdicts, pre_rejected)
    mark_revision(dossier, bindings.revision_cap)
    supported = sum(verdict.verdict == "supported" for verdict in dossier.verdicts)
    state = "revision requested" if dossier.needs_revision else "accepted"
    return f"{state}; {supported} supported"


class FactCheckAgent(Executor):
    """Verifies each claim and flags a single summary revision when needed."""

    def __init__(self, completer: Completer, *, revision_cap: int = 1) -> None:
        super().__init__(id="factcheck")
        self._completer = completer
        self._revision_cap = revision_cap

    @handler
    async def check(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        dossier = use_shared(dossier, revision_cap=self._revision_cap)
        review_json = await invoke("factcheck", select_checkable_claims)
        review = json.loads(review_json)
        eligible = [Claim.model_validate(item) for item in review["eligible"]]
        model_verdicts: list[dict[str, str]] = []
        if eligible:
            draft = await self._completer.complete(
                name="factcheck",
                instructions=_INSTRUCTIONS,
                prompt=_prompt(dossier, eligible),
                response_format=FactCheckDraft,
            )
            model_verdicts = [
                {
                    "claim_id": item.claim_id,
                    "verdict": item.verdict,
                    "reason": item.reason.strip() or "No reason given.",
                    "supporting_quote": item.supporting_quote.strip(),
                }
                for item in draft.verdicts
            ]
        await invoke(
            "factcheck",
            commit_verdicts,
            review_json=review_json,
            verdicts_json=json.dumps(model_verdicts),
        )
        await ctx.send_message(dossier)


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
