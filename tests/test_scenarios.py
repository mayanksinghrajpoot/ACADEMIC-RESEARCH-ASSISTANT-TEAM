"""Three scenarios for the shared-dossier workflow.

1. A covered question runs all six agents, and each agent calls two tools on one dossier.
2. The critic rejects an unquoted claim, summary runs once more, and the claim stays dropped.
3. A question outside the papers stops before the agents, and a search with no hits stops
   after the retrieval tools. The run log is only appended to.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from research_assistant.agents.contradiction_agent import (
    commit_contradictions,
    read_passages_for_contradictions,
)
from research_assistant.agents.factcheck_agent import commit_verdicts, select_checkable_claims
from research_assistant.agents.query_agent import commit_subqueries, list_corpus_titles
from research_assistant.agents.retrieval_agent import (
    RetrievalAgent,
    RetrievalError,
    embed_subqueries,
    store_retrieved_passages,
)
from research_assistant.agents.summary_agent import commit_claims, read_passages_for_summary
from research_assistant.agents.writer_agent import draft_supported_overview, publish_report
from research_assistant.blackboard import RunBindings, bind_run
from research_assistant.config import Settings
from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.orchestration.pipeline import run_research
from research_assistant.rag.corpus import Paper
from research_assistant.rag.index import PassageIndex
from tests.fakes import BAD_CLAIM, ScriptedCompleter

AGENT_TOOLS = {
    "query": {"list_corpus_titles", "commit_subqueries"},
    "retrieval": {"embed_subqueries", "store_retrieved_passages"},
    "summary": {"read_passages_for_summary", "commit_claims"},
    "contradiction": {"read_passages_for_contradictions", "commit_contradictions"},
    "factcheck": {"select_checkable_claims", "commit_verdicts"},
    "writer": {"draft_supported_overview", "publish_report"},
}


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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="",
        corpus_dir=tmp_path,
        output_dir=tmp_path / "out",
        top_k=2,
        revision_cap=1,
        log_path=tmp_path / "run.log",
    )


def _log_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "agent" in payload:
            rows.append(payload)
    return rows


def test_tools_are_plain_functions() -> None:
    tools = [
        list_corpus_titles,
        commit_subqueries,
        embed_subqueries,
        store_retrieved_passages,
        read_passages_for_summary,
        commit_claims,
        read_passages_for_contradictions,
        commit_contradictions,
        select_checkable_claims,
        commit_verdicts,
        draft_supported_overview,
        publish_report,
    ]
    assert len(tools) == 12
    for function_tool in tools:
        assert inspect.isfunction(function_tool.func)


@pytest.mark.asyncio
async def test_covered_question_shares_one_dossier(tmp_path: Path) -> None:
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=ScriptedCompleter(all_supported=True),
        papers=_papers(),
        index=_index(),
    )

    rows = _log_rows(tmp_path / "run.log")
    by_agent: dict[str, set[str]] = {}
    for row in rows:
        by_agent.setdefault(str(row["agent"]), set()).add(str(row["tool"]))
    assert by_agent.keys() == AGENT_TOOLS.keys()
    for agent, tools in AGENT_TOOLS.items():
        assert tools <= by_agent[agent]
    assert len({row["dossier_id"] for row in rows}) == 1
    assert "`paper-a#c0`" in dossier.report_markdown
    assert [line.split(" · ")[0] for line in dossier.trace[:4]] == [
        "query",
        "query",
        "retrieval",
        "retrieval",
    ]
    assert [line.split(" · ")[0] for line in dossier.trace[-2:]] == ["writer", "writer"]


@pytest.mark.asyncio
async def test_critic_drops_an_unquoted_claim_after_one_retry(tmp_path: Path) -> None:
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=ScriptedCompleter(),
        papers=_papers(),
        index=_index(),
    )

    rows = _log_rows(tmp_path / "run.log")
    tools = [str(row["tool"]) for row in rows]
    assert tools.count("commit_claims") == 2
    assert tools.count("commit_verdicts") == 2
    assert tools.count("commit_contradictions") == 1
    themes, dropped = dossier.report_markdown.split("## Dropped claims")
    assert BAD_CLAIM in dropped
    assert BAD_CLAIM not in themes
    assert "revision requested" in dossier.trace[0] or any("revision requested" in line for line in dossier.trace)


@pytest.mark.asyncio
async def test_uncovered_question_and_empty_search_only_append_the_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text('{"kept": true}\n', encoding="utf-8")
    settings = _settings(tmp_path)

    with pytest.raises(RetrievalError, match="do not cover"):
        await run_research(
            "zebrafish sonnet form",
            settings,
            completer=ScriptedCompleter(),
            papers=_papers(),
            index=_index(),
        )

    dossier = ResearchDossier(question="q", sub_queries=["zebrafish sonnet form"])
    index = PassageIndex(
        [Passage(passage_id="a#c0", paper_id="a", title="A", text="catalysis recall only")]
    )

    class Ctx:
        async def send_message(self, message: ResearchDossier) -> None:
            raise AssertionError("an empty search must not continue")

    with bind_run(RunBindings(dossier=dossier, index=index, top_k=2, log_path=log_path, run_id="miss")):
        with pytest.raises(RetrievalError, match="will not invent sources"):
            await RetrievalAgent(index, top_k=2).retrieve(dossier, Ctx())  # type: ignore[arg-type]

    text = log_path.read_text(encoding="utf-8")
    assert text.startswith('{"kept": true}\n')
    rows = _log_rows(log_path)
    covered = [row for row in rows if row["tool"] == "corpus_covers"]
    assert covered and covered[0]["ok"] is False
    assert "list_corpus_titles" not in {row["tool"] for row in rows}
    miss = [row for row in rows if row["run_id"] == "miss"]
    assert [row["tool"] for row in miss] == ["embed_subqueries", "store_retrieved_passages"]
    assert miss[-1]["ok"] is False
