"""In-memory chat completer for pipeline tests. No network calls."""

from __future__ import annotations

from research_assistant.agents.contradiction_agent import ContradictionDraft, ContradictionReport
from research_assistant.agents.factcheck_agent import FactCheckDraft, VerdictDraft
from research_assistant.agents.query_agent import QueryDraft
from research_assistant.agents.summary_agent import ClaimDraft, SummaryDraft

SUPPORTED_QUOTE = "Alpha reduces measured error when the passage is retrieved and shown."
BAD_CLAIM = "Alpha cures every disease."


class ScriptedCompleter:
    """Returns schema objects keyed by agent name."""

    def __init__(
        self,
        *,
        revision_still_bad: bool = False,
        all_supported: bool = False,
        smuggle_overview: bool = False,
    ) -> None:
        self.names: list[str] = []
        self.revision_still_bad = revision_still_bad
        self.all_supported = all_supported
        self.smuggle_overview = smuggle_overview

    async def complete(self, *, name: str, instructions: str, prompt: str, response_format: type):
        self.names.append(name)
        if name == "query":
            return QueryDraft(
                sub_queries=["alpha measured error", "alpha does not reduce", "passage retrieved"]
            )
        if name == "summary":
            return self._summary(prompt)
        if name == "contradiction":
            return ContradictionReport(
                contradictions=[
                    ContradictionDraft(
                        topic="Effect of alpha",
                        position_a="Alpha reduces measured error when the passage is shown.",
                        passage_ids_a=["paper-a#c0"],
                        quote_a=SUPPORTED_QUOTE,
                        position_b="Alpha does not reduce measured error.",
                        passage_ids_b=["paper-b#c0"],
                        quote_b="Alpha does not reduce measured error.",
                    )
                ]
            )
        if name == "factcheck":
            return FactCheckDraft(
                verdicts=[
                    VerdictDraft(
                        claim_id="c1",
                        verdict="supported",
                        reason="The cited passage states the reduction.",
                        supporting_quote=SUPPORTED_QUOTE,
                    ),
                    VerdictDraft(
                        claim_id="c2",
                        verdict="supported",
                        reason="The model tried to keep an uncited claim.",
                        supporting_quote="not a real quote from the corpus",
                    ),
                ]
            )
        raise AssertionError(f"unexpected agent {name}")

    def _summary(self, prompt: str) -> SummaryDraft:
        good = ClaimDraft(
            theme="Measured effect",
            text="Alpha reduces measured error when the passage is shown.",
            passage_ids=["paper-a#c0"],
        )
        bad = ClaimDraft(theme="Measured effect", text=BAD_CLAIM, passage_ids=["missing-passage"])
        if self.all_supported or ("This is a revision" in prompt and not self.revision_still_bad):
            return SummaryDraft(claims=[good])
        return SummaryDraft(claims=[good, bad])
