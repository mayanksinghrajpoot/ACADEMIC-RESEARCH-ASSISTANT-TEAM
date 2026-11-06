"""Writer agent: final summary built from supported claims, without another model call."""

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler

from research_assistant.memory.dossier import ResearchDossier
from research_assistant.reporting import render_markdown


class WriterAgent(Executor):
    """Publishes the report from verdicts. The overview is the supported claims."""

    def __init__(self) -> None:
        super().__init__(id="writer")

    @handler
    async def write(self, dossier: ResearchDossier, ctx: WorkflowContext[None, ResearchDossier]) -> None:
        report = render_markdown(dossier, overview_from_claims(dossier))
        await ctx.yield_output(dossier.model_copy(update={"report_markdown": report}))


def overview_from_claims(dossier: ResearchDossier) -> str:
    """One sentence per supported claim. No model call, so no new facts."""
    supported = dossier.supported_claims()
    if not supported:
        return "No claim survived fact-checking against the retrieved passages."
    return " ".join(f"{claim.text.rstrip('.')}." for claim in supported)
