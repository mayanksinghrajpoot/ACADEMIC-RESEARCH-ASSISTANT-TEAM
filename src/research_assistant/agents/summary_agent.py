"""Summary agent: themed claims that must cite retrieved passage ids."""

from __future__ import annotations

import json

from agent_framework import Executor, WorkflowContext, handler, tool
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
from research_assistant.blackboard import get_bindings, invoke, use_shared
from research_assistant.llm import Completer
from research_assistant.memory.dossier import Claim, ResearchDossier

_INSTRUCTIONS = """You write a short literature summary as discrete claims grouped by theme.
Every claim must be supported by one or more passage ids copied from the retrieved set.
Do not use outside knowledge. Do not cite a passage that is not in the prompt.
If critic notes are present, rewrite the summary: drop rejected claims unless a retrieved
passage now supports a corrected wording, and do not repeat a rejected claim unchanged.
"""


class ClaimDraft(BaseModel):
    theme: str
    text: str
    passage_ids: list[str] = Field(default_factory=list)


class SummaryDraft(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)


@tool
def read_passages_for_summary() -> str:
    """Read the passages stored on the shared dossier."""
    return format_passages(get_bindings().dossier.passages) or "(no passages)"


@tool
def commit_claims(claims_json: str) -> str:
    """Replace the dossier claims with this draft and clear stale verdicts."""
    dossier = get_bindings().dossier
    raw = json.loads(claims_json)
    if not isinstance(raw, list):
        raise ValueError("claims must be a JSON list.")
    claims: list[Claim] = []
    for index, item in enumerate(raw, start=1):
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        passage_ids = item.get("passage_ids") or []
        claims.append(
            Claim(
                claim_id=f"c{index}",
                theme=str(item.get("theme", "")).strip() or "General",
                text=text,
                passage_ids=list(dict.fromkeys(str(passage_id) for passage_id in passage_ids)),
            )
        )
    dossier.claims = claims
    dossier.verdicts = []
    dossier.needs_revision = False
    return f"{len(claims)} claims"


class SummaryAgent(Executor):
    """Writes structured claims. On a critic retry it rewrites from the notes."""

    def __init__(self, completer: Completer) -> None:
        super().__init__(id="summary")
        self._completer = completer

    async def apply(self, dossier: ResearchDossier) -> ResearchDossier:
        """Write claims onto the shared dossier. Safe beside the contradiction check."""
        dossier = use_shared(dossier)
        passages = await invoke("summary", read_passages_for_summary)
        draft = await self._completer.complete(
            name="summary",
            instructions=_INSTRUCTIONS,
            prompt=_prompt(dossier, passages),
            response_format=SummaryDraft,
        )
        payload = [
            {"theme": item.theme, "text": item.text, "passage_ids": item.passage_ids}
            for item in draft.claims
        ]
        await invoke("summary", commit_claims, claims_json=json.dumps(payload))
        return dossier

    @handler
    async def summarize(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        await ctx.send_message(await self.apply(dossier))


def _prompt(dossier: ResearchDossier, passages: str) -> str:
    parts = [
        f"Research question:\n{dossier.question}",
        "",
        "Retrieved passages:",
        passages,
        "",
        "Return JSON claims. Each claim has theme, text, and passage_ids.",
    ]
    if dossier.revision_count > 0 and dossier.critic_notes:
        notes = "\n".join(f"- {note}" for note in dossier.critic_notes)
        parts.extend(
            [
                "",
                "This is a revision. The fact-check critic rejected these claims:",
                notes,
                "Do not restate a rejected claim unless a retrieved passage supports the new wording.",
            ]
        )
    return "\n".join(parts)
