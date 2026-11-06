"""First draft: summary and contradiction run at the same time."""

from __future__ import annotations

import asyncio

from agent_framework import Executor, WorkflowContext, handler

from research_assistant.agents.contradiction_agent import ContradictionAgent
from research_assistant.agents.summary_agent import SummaryAgent
from research_assistant.memory.dossier import ResearchDossier


class OpeningDraft(Executor):
    """Writes claims and looks for disagreements in one step.

    The two model calls start together. Wall-clock time is the slower call,
    not the sum. A later summary retry does not repeat the contradiction call.
    """

    def __init__(self, summary: SummaryAgent, contradiction: ContradictionAgent) -> None:
        super().__init__(id="opening")
        self._summary = summary
        self._contradiction = contradiction

    @handler
    async def draft(self, dossier: ResearchDossier, ctx: WorkflowContext[ResearchDossier]) -> None:
        await asyncio.gather(
            self._summary.apply(dossier),
            self._contradiction.apply(dossier),
        )
        await ctx.send_message(dossier)
