"""Shared dossier. Every agent reads it and writes only its own fields."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A retrieved chunk that later claims may cite."""

    passage_id: str
    paper_id: str
    title: str
    text: str


class Claim(BaseModel):
    """A single statement in the themed summary."""

    claim_id: str
    theme: str
    text: str
    passage_ids: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """Two positions, each tied to different retrieved passages and a copied quote."""

    topic: str
    position_a: str
    passage_ids_a: list[str]
    quote_a: str = ""
    position_b: str
    passage_ids_b: list[str]
    quote_b: str = ""


class Verdict(BaseModel):
    """Fact-check result for one claim."""

    claim_id: str
    verdict: Literal["supported", "unsupported"]
    reason: str
    supporting_quote: str = ""


class DroppedClaim(BaseModel):
    """A claim removed as unsupported, kept for the final report."""

    text: str
    reason: str
    claim_id: str = ""


class ResearchDossier(BaseModel):
    """Blackboard passed along the pipeline."""

    question: str
    sub_queries: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    contradiction_note: str = ""
    verdicts: list[Verdict] = Field(default_factory=list)
    dropped_log: list[DroppedClaim] = Field(default_factory=list)
    revision_count: int = 0
    needs_revision: bool = False
    critic_notes: list[str] = Field(default_factory=list)
    report_markdown: str = ""

    def passage_index(self) -> dict[str, Passage]:
        return {passage.passage_id: passage for passage in self.passages}

    def verdict_for(self, claim_id: str) -> Verdict | None:
        for verdict in self.verdicts:
            if verdict.claim_id == claim_id:
                return verdict
        return None

    def supported_claims(self) -> list[Claim]:
        return [
            claim
            for claim in self.claims
            if (verdict := self.verdict_for(claim.claim_id)) is not None
            and verdict.verdict == "supported"
        ]

    def dropped_claims(self) -> list[DroppedClaim]:
        """Unsupported claims from the critic retry and from the final verdicts."""
        seen: set[tuple[str, str]] = set()
        dropped: list[DroppedClaim] = []
        for item in self.dropped_log:
            key = (item.text, item.reason)
            if key in seen:
                continue
            seen.add(key)
            dropped.append(item)
        for claim, verdict in self.unsupported_claims():
            key = (claim.text, verdict.reason)
            if key in seen:
                continue
            seen.add(key)
            dropped.append(DroppedClaim(text=claim.text, reason=verdict.reason, claim_id=claim.claim_id))
        return dropped

    def unsupported_claims(self) -> list[tuple[Claim, Verdict]]:
        dropped: list[tuple[Claim, Verdict]] = []
        for claim in self.claims:
            verdict = self.verdict_for(claim.claim_id)
            if verdict is None or verdict.verdict != "supported":
                dropped.append(
                    (
                        claim,
                        verdict
                        or Verdict(
                            claim_id=claim.claim_id,
                            verdict="unsupported",
                            reason="No fact-check verdict was recorded.",
                        ),
                    )
                )
        return dropped
