"""Writer agent: final summary built from supported claims, without another model call."""

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler, tool

from research_assistant.blackboard import get_bindings, invoke, use_shared
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.reporting import render_markdown


@tool
def draft_supported_overview() -> str:
    """Build the overview from supported claims only. Does not call a model."""
    return overview_from_claims(get_bindings().dossier)


@tool
def publish_report(overview: str) -> str:
    """Write the markdown report onto the shared dossier."""
    dossier = get_bindings().dossier
    dossier.report_markdown = render_markdown(dossier, overview)
    return f"{len(dossier.report_markdown)} characters"


class WriterAgent(Executor):
    """Publishes the report from verdicts. The overview is the supported claims."""

    def __init__(self) -> None:
        super().__init__(id="writer")

    @handler
    async def write(self, dossier: ResearchDossier, ctx: WorkflowContext[None, ResearchDossier]) -> None:
        dossier = use_shared(dossier)
        overview = await invoke("writer", draft_supported_overview)
        await invoke("writer", publish_report, overview=overview)
        await ctx.yield_output(dossier)


def overview_from_claims(dossier: ResearchDossier) -> str:
    """One sentence per supported claim. No model call, so no new facts."""
    supported = dossier.supported_claims()
    if not supported:
        return "No claim survived fact-checking against the retrieved passages."
    return " ".join(f"{claim.text.rstrip('.')}." for claim in supported)
