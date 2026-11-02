"""Deterministic checks that a claim or contradiction cites retrieved passages."""

from __future__ import annotations

from research_assistant.memory.dossier import (
    Claim,
    Contradiction,
    DroppedClaim,
    Passage,
    ResearchDossier,
    Verdict,
)

MIN_QUOTE_CHARS = 20


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def partition_claims(dossier: ResearchDossier) -> tuple[list[Claim], list[Verdict]]:
    """Split claims into those a model may judge and those the gate already rejects.

    A claim is eligible only when every passage id it cites is in the retrieved set.
    Missing or unknown ids are unsupported before the model runs, and stay unsupported
    even if the model later calls them supported.
    """
    known = set(dossier.passage_index())
    eligible: list[Claim] = []
    rejected: list[Verdict] = []
    for claim in dossier.claims:
        if claim.passage_ids and all(passage_id in known for passage_id in claim.passage_ids):
            eligible.append(claim)
            continue
        rejected.append(
            Verdict(
                claim_id=claim.claim_id,
                verdict="unsupported",
                reason="Claim does not cite a passage from the retrieved set.",
            )
        )
    return eligible, rejected


def merge_verdicts(
    dossier: ResearchDossier,
    model_verdicts: list[Verdict],
    pre_rejected: list[Verdict],
) -> list[Verdict]:
    """Apply the gate again after the critic model returns."""
    known = dossier.passage_index()
    rejected_ids = {verdict.claim_id for verdict in pre_rejected}
    by_id = {verdict.claim_id: verdict for verdict in model_verdicts}
    merged: list[Verdict] = []
    for claim in dossier.claims:
        if claim.claim_id in rejected_ids:
            merged.append(next(verdict for verdict in pre_rejected if verdict.claim_id == claim.claim_id))
            continue
        verdict = by_id.get(claim.claim_id)
        if verdict is None or verdict.verdict == "unsupported":
            merged.append(
                verdict
                or Verdict(
                    claim_id=claim.claim_id,
                    verdict="unsupported",
                    reason="Fact-check returned no verdict for this claim.",
                )
            )
            continue
        cited = [known[passage_id] for passage_id in claim.passage_ids]
        if quote_in_passages(verdict.supporting_quote, cited):
            merged.append(verdict.model_copy(update={"verdict": "supported"}))
            continue
        merged.append(
            Verdict(
                claim_id=claim.claim_id,
                verdict="unsupported",
                reason="Supporting quote was not found in a cited retrieved passage.",
                supporting_quote=verdict.supporting_quote,
            )
        )
    return merged


def quote_in_passages(quote: str, passages: list[Passage]) -> bool:
    needle = normalize_text(quote)
    if len(needle) < MIN_QUOTE_CHARS:
        return False
    return any(needle in normalize_text(passage.text) for passage in passages)


def grounded_contradictions(
    dossier: ResearchDossier,
    contradictions: list[Contradiction],
) -> tuple[list[Contradiction], str]:
    """Keep disagreements whose two sides cite different retrieved passages."""
    known = set(dossier.passage_index())
    kept: list[Contradiction] = []
    for item in contradictions:
        side_a = set(item.passage_ids_a)
        side_b = set(item.passage_ids_b)
        if not side_a or not side_b:
            continue
        if not side_a <= known or not side_b <= known:
            continue
        if side_a & side_b:
            continue
        kept.append(item)
    if not kept:
        return [], "No contradiction was found in the retrieved passages."
    return kept, ""


def mark_revision(dossier: ResearchDossier, revision_cap: int) -> ResearchDossier:
    """Request one summary rewrite when unsupported claims remain under the cap."""
    unsupported = dossier.unsupported_claims()
    notes = [f"{claim.text} — {verdict.reason}" for claim, verdict in unsupported]
    revise = bool(notes) and dossier.revision_count < revision_cap
    dropped_log = list(dossier.dropped_log)
    if revise:
        dropped_log.extend(
            DroppedClaim(text=claim.text, reason=verdict.reason, claim_id=claim.claim_id)
            for claim, verdict in unsupported
        )
    return dossier.model_copy(
        update={
            "needs_revision": revise,
            "revision_count": dossier.revision_count + (1 if revise else 0),
            "critic_notes": notes,
            "dropped_log": dropped_log,
        }
    )
