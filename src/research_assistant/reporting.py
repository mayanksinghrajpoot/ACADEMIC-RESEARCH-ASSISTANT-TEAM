"""Render the final literature summary from supported claims only."""

from __future__ import annotations

from collections import OrderedDict

from research_assistant.memory.dossier import Claim, Passage, ResearchDossier


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


def report_view(dossier: ResearchDossier) -> dict[str, object]:
    """Structured summary for the local page. The same claims the markdown uses."""
    passages = dossier.passage_index()
    grouped: OrderedDict[str, list[Claim]] = OrderedDict()
    for claim in dossier.supported_claims():
        grouped.setdefault(claim.theme, []).append(claim)

    themes: list[dict[str, object]] = []
    for theme, claims in grouped.items():
        items: list[dict[str, object]] = []
        for claim in claims:
            verdict = dossier.verdict_for(claim.claim_id)
            items.append(
                {
                    "text": claim.text,
                    "quote": verdict.supporting_quote if verdict else "",
                    "sources": _source_records(passages, claim.passage_ids),
                }
            )
        themes.append({"theme": theme, "claims": items})

    contradictions: list[dict[str, object]] = []
    for item in dossier.contradictions:
        contradictions.append(
            {
                "topic": item.topic,
                "position_a": item.position_a,
                "quote_a": item.quote_a,
                "sources_a": _source_records(passages, item.passage_ids_a),
                "position_b": item.position_b,
                "quote_b": item.quote_b,
                "sources_b": _source_records(passages, item.passage_ids_b),
            }
        )

    note = dossier.contradiction_note
    if not contradictions and not note:
        note = "No contradiction was found in the retrieved passages."

    return {
        "question": dossier.question,
        "overview": _overview_text(dossier),
        "themes": themes,
        "contradiction_note": note,
        "contradictions": contradictions,
        "dropped": [{"text": item.text, "reason": item.reason} for item in dossier.dropped_claims()],
        "sub_queries": list(dossier.sub_queries),
        "passage_count": len(dossier.passages),
    }


def _overview_text(dossier: ResearchDossier) -> str:
    text = dossier.report_markdown
    if "## Overview" in text and "## Themes" in text:
        return text.split("## Overview", 1)[1].split("## Themes", 1)[0].strip()
    return _default_overview(dossier)


def _source_records(passages: dict[str, Passage], passage_ids: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for passage_id in passage_ids:
        passage = passages.get(passage_id)
        if passage is None:
            records.append({"title": passage_id, "passage_id": passage_id})
            continue
        records.append({"title": passage.title, "passage_id": passage.passage_id})
    return records


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
        lines.append(f"- Position A ({_sources(dossier, item.passage_ids_a)}): {item.position_a}")
        if item.quote_a:
            lines.append(f"  - Quote: {item.quote_a}")
        lines.append(f"- Position B ({_sources(dossier, item.passage_ids_b)}): {item.position_b}")
        if item.quote_b:
            lines.append(f"  - Quote: {item.quote_b}")
        lines.append("")
    return lines


def _dropped_section(dossier: ResearchDossier) -> list[str]:
    dropped = dossier.dropped_claims()
    if not dropped:
        return ["None.", ""]
    lines = [f"- {item.text} — {item.reason}" for item in dropped]
    lines.append("")
    return lines


def _sources(dossier: ResearchDossier, passage_ids: list[str]) -> str:
    passages = dossier.passage_index()
    parts: list[str] = []
    for passage_id in passage_ids:
        passage = passages.get(passage_id)
        if passage is None:
            parts.append(f"`{passage_id}`")
            continue
        parts.append(f"{passage.title} (`{passage.passage_id}`)")
    return ", ".join(parts)
