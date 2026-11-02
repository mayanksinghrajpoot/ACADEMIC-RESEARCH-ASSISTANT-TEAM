"""Render the final literature summary from supported claims only."""

from __future__ import annotations

from collections import OrderedDict

from research_assistant.memory.dossier import Claim, ResearchDossier


def render_markdown(dossier: ResearchDossier, overview: str) -> str:
    """Build the reader-facing summary.

    Theme bullets come only from claims the fact-check marked supported.
    Unsupported claims appear in the dropped list and nowhere else.
    """
    lines: list[str] = [
        "# Literature summary",
        "",
        f"Question: {dossier.question}",
        "",
        "## Overview",
        "",
        overview.strip() or _default_overview(dossier),
        "",
        "## Themes",
        "",
    ]
    lines.extend(_theme_sections(dossier))
    lines.extend(["## Contradictions", ""])
    lines.extend(_contradiction_section(dossier))
    lines.extend(["## Dropped claims", ""])
    lines.extend(_dropped_section(dossier))
    return "\n".join(lines).rstrip() + "\n"


def _default_overview(dossier: ResearchDossier) -> str:
    supported = dossier.supported_claims()
    if not supported:
        return "Fact-checking left no claim that maps to a retrieved passage."
    themes = list(dict.fromkeys(claim.theme for claim in supported))
    joined = ", ".join(themes)
    return f"The supported claims cover {len(supported)} point(s) under: {joined}."


def _theme_sections(dossier: ResearchDossier) -> list[str]:
    supported = dossier.supported_claims()
    if not supported:
        return ["No supported claims remained after fact-checking.", ""]
    grouped: OrderedDict[str, list[Claim]] = OrderedDict()
    for claim in supported:
        grouped.setdefault(claim.theme, []).append(claim)
    passages = dossier.passage_index()
    lines: list[str] = []
    for theme, claims in grouped.items():
        lines.append(f"### {theme}")
        lines.append("")
        for claim in claims:
            verdict = dossier.verdict_for(claim.claim_id)
            lines.append(f"- {claim.text}")
            if verdict and verdict.supporting_quote:
                lines.append(f"  - Quote: {verdict.supporting_quote}")
            for passage_id in claim.passage_ids:
                passage = passages.get(passage_id)
                if passage is None:
                    continue
                lines.append(f"  - Source: {passage.title} (`{passage.passage_id}`)")
        lines.append("")
    return lines


def _contradiction_section(dossier: ResearchDossier) -> list[str]:
    if not dossier.contradictions:
        note = dossier.contradiction_note or "No contradiction was found in the retrieved passages."
        return [note, ""]
    lines: list[str] = []
    if dossier.contradiction_note:
        lines.extend([dossier.contradiction_note, ""])
    for item in dossier.contradictions:
        lines.append(f"### {item.topic}")
        lines.append("")
        lines.append(f"- Position A ({_ids(item.passage_ids_a)}): {item.position_a}")
        lines.append(f"- Position B ({_ids(item.passage_ids_b)}): {item.position_b}")
        lines.append("")
    return lines


def _dropped_section(dossier: ResearchDossier) -> list[str]:
    dropped = dossier.dropped_claims()
    if not dropped:
        return ["None.", ""]
    lines = [f"- {item.text} — {item.reason}" for item in dropped]
    lines.append("")
    return lines


def _ids(passage_ids: list[str]) -> str:
    return ", ".join(f"`{passage_id}`" for passage_id in passage_ids)
