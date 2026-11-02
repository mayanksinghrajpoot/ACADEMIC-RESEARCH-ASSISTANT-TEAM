"""Shared prompt formatting for passages on the dossier."""

from __future__ import annotations

from research_assistant.memory.dossier import Passage


def format_passages(passages: list[Passage]) -> str:
    blocks: list[str] = []
    for passage in passages:
        blocks.append(f"[{passage.passage_id}] {passage.title}\n{passage.text}")
    return "\n\n".join(blocks)
