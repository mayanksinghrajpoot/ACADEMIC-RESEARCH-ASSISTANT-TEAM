"""Summary agent: themed claims that must cite retrieved passage ids."""

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler
from pydantic import BaseModel, Field

from research_assistant.agents.context import format_passages
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


class SummaryAgent(Executor):
    """Writes structured claims. On a critic retry it rewrites from the notes."""

    def __init__(self, completer: Completer) -> None:
        super().__init__(id="summary")
        self._completer = completer

    @handler
    async def summarize(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        prompt = _prompt(dossier)
        draft = await self._completer.complete(
            name="summary",
            instructions=_INSTRUCTIONS,
            prompt=prompt,
            response_format=SummaryDraft,
        )
        claims = [
            Claim(
                claim_id=f"c{index}",
                theme=item.theme.strip() or "General",
                text=" ".join(item.text.split()),
                passage_ids=list(dict.fromkeys(item.passage_ids)),
            )
            for index, item in enumerate(draft.claims, start=1)
            if item.text.strip()
        ]
        updated = dossier.model_copy(
            update={
                "claims": claims,
                "verdicts": [],
                "needs_revision": False,
            }
        )
        await ctx.send_message(updated)


def _prompt(dossier: ResearchDossier) -> str:
    parts = [
        f"Research question:\n{dossier.question}",
        "",
        "Retrieved passages:",
        format_passages(dossier.passages),
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
