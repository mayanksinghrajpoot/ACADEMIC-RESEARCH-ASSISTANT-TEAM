from research_assistant.memory.dossier import (
    Claim,
    Contradiction,
    Passage,
    ResearchDossier,
    Verdict,
)
from research_assistant.reporting import render_markdown


def test_writer_keeps_supported_claims_and_lists_dropped_ones() -> None:
    dossier = ResearchDossier(
        question="Does alpha help?",
        passages=[
            Passage(
                passage_id="paper-a#c0",
                paper_id="paper-a",
                title="Paper A",
                text="Alpha reduces measured error when the passage is retrieved and shown.",
            )
        ],
        claims=[
            Claim(
                claim_id="c1",
                theme="Effect",
                text="Alpha reduces measured error.",
                passage_ids=["paper-a#c0"],
            ),
            Claim(claim_id="c2", theme="Effect", text="Alpha cures every disease.", passage_ids=[]),
        ],
        verdicts=[
            Verdict(
                claim_id="c1",
                verdict="supported",
                reason="Quoted.",
                supporting_quote="Alpha reduces measured error when the passage is retrieved and shown.",
            ),
            Verdict(claim_id="c2", verdict="unsupported", reason="No retrieved passage."),
        ],
        contradictions=[
            Contradiction(
                topic="Effect of alpha",
                position_a="Alpha reduces measured error.",
                passage_ids_a=["paper-a#c0"],
                position_b="Alpha does not reduce measured error.",
                passage_ids_b=["paper-b#c0"],
            )
        ],
    )
    report = render_markdown(dossier, "Overview stays with supported claims.")
    themes, rest = report.split("## Contradictions")
    contradictions, dropped = rest.split("## Dropped claims")

    assert "Alpha reduces measured error." in themes
    assert "Alpha cures every disease." not in themes
    assert "`paper-a#c0`" in themes
    assert "`paper-a#c0`" in contradictions
    assert "`paper-b#c0`" in contradictions
    assert "paper-a#c0" != "paper-b#c0"
    assert "Alpha cures every disease. — No retrieved passage." in dropped
