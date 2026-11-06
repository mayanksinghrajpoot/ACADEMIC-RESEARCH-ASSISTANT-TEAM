"""Offline demo: the real workflow, a scripted model, and two short papers.

The live command is ``python -m research_assistant``. This module is the path
the notebook uses so the multi-agent trace prints without an API key.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from research_assistant.agents.retrieval_agent import RetrievalError
from research_assistant.config import Settings
from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.orchestration.pipeline import run_research
from research_assistant.rag.corpus import Paper
from research_assistant.rag.index import PassageIndex

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "run.log"
DEMO_QUESTION = "Does alpha reduce error?"


def _papers() -> list[Paper]:
    return [
        Paper(paper_id="paper-a", title="Paper A", text="Alpha reduces measured error."),
        Paper(paper_id="paper-b", title="Paper B", text="Alpha does not reduce measured error."),
    ]


def _index() -> PassageIndex:
    return PassageIndex(
        [
            Passage(
                passage_id="paper-a#c0",
                paper_id="paper-a",
                title="Paper A",
                text=(
                    "Alpha reduces measured error when the passage is retrieved and shown. "
                    "The lab recorded the drop."
                ),
            ),
            Passage(
                passage_id="paper-b#c0",
                paper_id="paper-b",
                title="Paper B",
                text="Alpha does not reduce measured error. The rate stayed flat.",
            ),
        ]
    )


async def run_offline_demo(
    question: str = DEMO_QUESTION,
    *,
    log_path: Path | None = None,
) -> ResearchDossier:
    """Run one question and append the tool trace to the run log."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.fakes import ScriptedCompleter

    settings = Settings(
        gemini_api_key="",
        corpus_dir=ROOT / "corpus",
        output_dir=ROOT / "outputs",
        top_k=2,
        revision_cap=1,
        log_path=log_path or LOG_PATH,
    )
    return await run_research(
        question,
        settings,
        completer=ScriptedCompleter(),
        papers=_papers(),
        index=_index(),
    )


def _print_trace(dossier: ResearchDossier) -> None:
    print("Multi-agent trace")
    for line in dossier.trace:
        print(line)
    print()
    print(dossier.report_markdown)


def main() -> int:
    """Print the trace for the demo question, then the uncovered-question stop."""
    dossier = asyncio.run(run_offline_demo())
    _print_trace(dossier)
    try:
        asyncio.run(run_offline_demo("zebrafish sonnet form"))
    except RetrievalError as exc:
        print(f"error: {exc}")
    print(f"Appended {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
