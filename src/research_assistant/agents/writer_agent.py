"""Writer agent: final summary that keeps only supported, sourced claims."""

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler
from pydantic import BaseModel

from research_assistant.llm import Completer
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.reporting import render_markdown

_INSTRUCTIONS = """You write a short overview paragraph for a literature summary.
Use only the supported claims provided. Do not add facts, numbers, or sources
that are not in those claims. Do not mention claims that were dropped.
If there are no supported claims, say that no claim survived fact-checking.
"""


class OverviewDraft(BaseModel):
    overview: str


class WriterAgent(Executor):
    """Publishes the report. The claim list is rendered from verdicts, not from prose."""

    def __init__(self, completer: Completer) -> None:
        super().__init__(id="writer")
        self._completer = completer

    @handler
    async def write(self, dossier: ResearchDossier, ctx: WorkflowContext[None, ResearchDossier]) -> None:
        supported = dossier.supported_claims()
        if not supported:
            overview = "No claim survived fact-checking against the retrieved passages."
        else:
            claim_lines = "\n".join(f"- ({claim.theme}) {claim.text}" for claim in supported)
            draft = await self._completer.complete(
                name="writer",
                instructions=_INSTRUCTIONS,
                prompt=(
                    f"Research question:\n{dossier.question}\n\n"
                    f"Supported claims:\n{claim_lines}\n\n"
                    "Return JSON with an overview string of one short paragraph."
                ),
                response_format=OverviewDraft,
            )
            overview = " ".join(draft.overview.split())
        report = render_markdown(dossier, overview)
        await ctx.yield_output(dossier.model_copy(update={"report_markdown": report}))
