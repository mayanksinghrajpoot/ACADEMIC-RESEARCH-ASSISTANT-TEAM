from research_assistant.grounding import (
    grounded_contradictions,
    mark_revision,
    merge_verdicts,
    partition_claims,
    quote_in_passages,
)
from research_assistant.memory.dossier import (
    Claim,
    Contradiction,
    Passage,
    ResearchDossier,
    Verdict,
)

QUOTE = "Retrieval lowered unsupported claims when the passage was shown."


def _dossier(*claims: Claim) -> ResearchDossier:
    return ResearchDossier(
        question="Does retrieval help?",
        passages=[
            Passage(passage_id="a#c0", paper_id="a", title="A", text=QUOTE + " Further detail follows."),
            Passage(passage_id="b#c0", paper_id="b", title="B", text="A second paper disputes the reduction in the same setting."),
        ],
        claims=list(claims),
    )


def test_missing_passage_ids_are_rejected_before_the_model() -> None:
    dossier = _dossier(
        Claim(claim_id="c1", theme="T", text="A grounded claim.", passage_ids=["a#c0"]),
        Claim(claim_id="c2", theme="T", text="An invented claim.", passage_ids=[]),
        Claim(claim_id="c3", theme="T", text="A bad citation.", passage_ids=["nope"]),
    )
    eligible, rejected = partition_claims(dossier)
    assert [claim.claim_id for claim in eligible] == ["c1"]
    assert {verdict.claim_id for verdict in rejected} == {"c2", "c3"}


def test_model_cannot_restore_a_claim_the_gate_rejected() -> None:
    dossier = _dossier(Claim(claim_id="c1", theme="T", text="Invented.", passage_ids=["missing"]))
    _, rejected = partition_claims(dossier)
    verdicts = merge_verdicts(
        dossier,
        [
            Verdict(
                claim_id="c1",
                verdict="supported",
                reason="model said yes",
                supporting_quote=QUOTE,
            )
        ],
        rejected,
    )
    assert verdicts[0].verdict == "unsupported"
    assert "retrieved set" in verdicts[0].reason


def test_supported_verdict_requires_a_quote_from_the_cited_passage() -> None:
    dossier = _dossier(Claim(claim_id="c1", theme="T", text="Retrieval helped.", passage_ids=["a#c0"]))
    kept = merge_verdicts(
        dossier,
        [Verdict(claim_id="c1", verdict="supported", reason="entailed", supporting_quote=QUOTE)],
        [],
    )
    rejected = merge_verdicts(
        dossier,
        [Verdict(claim_id="c1", verdict="supported", reason="guess", supporting_quote="too short")],
        [],
    )
    assert kept[0].verdict == "supported"
    assert rejected[0].verdict == "unsupported"
    assert quote_in_passages(QUOTE, dossier.passages[:1])


def test_contradiction_must_cite_two_different_passages() -> None:
    dossier = _dossier()
    kept, note = grounded_contradictions(
        dossier,
        [
            Contradiction(
                topic="Same passage",
                position_a="Left",
                passage_ids_a=["a#c0"],
                position_b="Right",
                passage_ids_b=["a#c0"],
            ),
            Contradiction(
                topic="Unknown",
                position_a="Left",
                passage_ids_a=["a#c0"],
                position_b="Right",
                passage_ids_b=["missing"],
            ),
            Contradiction(
                topic="Real dispute",
                position_a="Retrieval helped.",
                passage_ids_a=["a#c0"],
                position_b="The reduction is disputed.",
                passage_ids_b=["b#c0"],
            ),
        ],
    )
    assert [item.topic for item in kept] == ["Real dispute"]
    assert set(kept[0].passage_ids_a).isdisjoint(kept[0].passage_ids_b)
    assert note == ""

    none, empty_note = grounded_contradictions(dossier, [])
    assert none == []
    assert "No contradiction" in empty_note


def test_revision_is_allowed_only_once() -> None:
    dossier = _dossier(Claim(claim_id="c1", theme="T", text="Bad", passage_ids=[])).model_copy(
        update={
            "verdicts": [
                Verdict(claim_id="c1", verdict="unsupported", reason="No passage."),
            ]
        }
    )
    first = mark_revision(dossier, revision_cap=1)
    assert first.needs_revision is True
    assert first.revision_count == 1
    assert first.dropped_log[0].text == "Bad"

    second = mark_revision(first.model_copy(update={"needs_revision": False}), revision_cap=1)
    assert second.needs_revision is False
    assert second.revision_count == 1
