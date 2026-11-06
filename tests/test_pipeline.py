import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_assistant.agents.retrieval_agent import RetrievalAgent, RetrievalError
from research_assistant.config import ConfigurationError, Settings, load_settings
from research_assistant.memory.dossier import Passage, ResearchDossier
from research_assistant.orchestration.pipeline import clear_index_cache, prepare_corpus, run_research
from research_assistant.rag.index import PassageIndex
from research_assistant.rag.corpus import Paper
from research_assistant.__main__ import main, write_outputs
from tests.fakes import BAD_CLAIM, ScriptedCompleter


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
    )


@pytest.mark.asyncio
async def test_critic_retries_summary_once_then_writes(tmp_path: Path) -> None:
    completer = ScriptedCompleter()
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=completer,
        papers=_papers(),
        index=_index(),
    )

    assert completer.names.count("summary") == 2
    assert completer.names.count("factcheck") == 2
    assert completer.names.count("writer") == 0
    assert completer.names.count("contradiction") == 1
    assert dossier.revision_count == 1
    assert dossier.needs_revision is False

    themes, dropped = dossier.report_markdown.split("## Dropped claims")
    assert BAD_CLAIM in dropped
    assert BAD_CLAIM not in themes
    assert "`paper-a#c0`" in themes
    contradiction = dossier.report_markdown.split("## Contradictions")[1].split("## Dropped claims")[0]
    assert "`paper-a#c0`" in contradiction
    assert "`paper-b#c0`" in contradiction
    assert dossier.contradictions[0].passage_ids_a != dossier.contradictions[0].passage_ids_b


@pytest.mark.asyncio
async def test_second_failure_does_not_loop(tmp_path: Path) -> None:
    completer = ScriptedCompleter(revision_still_bad=True)
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=completer,
        papers=_papers(),
        index=_index(),
    )
    assert completer.names.count("summary") == 2
    assert completer.names.count("writer") == 0
    assert completer.names.count("contradiction") == 1
    assert dossier.needs_revision is False
    assert BAD_CLAIM in dossier.report_markdown.split("## Dropped claims")[1]


def test_outputs_round_trip_the_dossier(tmp_path: Path) -> None:
    dossier = ResearchDossier(question="Does alpha reduce error?", report_markdown="# Literature summary\n")
    markdown_path, json_path = write_outputs(dossier, tmp_path, "run-1")
    assert markdown_path.read_text(encoding="utf-8").startswith("# Literature summary")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["question"] == "Does alpha reduce error?"


def test_missing_api_key_fails_before_agents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings(tmp_path)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        settings.require_api_key()
    assert main(["Does retrieval reduce unsupported claims?"]) == 1


def test_invalid_setting_prints_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert main(["Does retrieval reduce unsupported claims?"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "TOP_K" in captured.err


@pytest.mark.asyncio
async def test_supported_summary_goes_straight_to_the_writer(tmp_path: Path) -> None:
    completer = ScriptedCompleter(all_supported=True)
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=completer,
        papers=_papers(),
        index=_index(),
    )
    assert completer.names.count("summary") == 1
    assert completer.names.count("factcheck") == 1
    assert completer.names.count("writer") == 0
    assert completer.names.count("contradiction") == 1
    assert dossier.revision_count == 0
    assert dossier.needs_revision is False
    assert BAD_CLAIM not in dossier.report_markdown
    assert "`paper-a#c0`" in dossier.report_markdown


@pytest.mark.asyncio
async def test_ungrounded_overview_is_replaced(tmp_path: Path) -> None:
    completer = ScriptedCompleter(smuggle_overview=True)
    dossier = await run_research(
        "Does alpha reduce error?",
        _settings(tmp_path),
        completer=completer,
        papers=_papers(),
        index=_index(),
    )
    overview = dossier.report_markdown.split("## Overview")[1].split("## Themes")[0]
    assert "Alpha reduces measured error when the passage is shown." in overview
    assert BAD_CLAIM not in overview
    assert BAD_CLAIM in dossier.report_markdown.split("## Dropped claims")[1]


@pytest.mark.asyncio
async def test_unrelated_question_stops_before_the_model(tmp_path: Path) -> None:
    completer = ScriptedCompleter()
    with pytest.raises(RetrievalError, match="do not cover"):
        await run_research(
            "zebrafish sonnet form",
            _settings(tmp_path),
            completer=completer,
            papers=_papers(),
            index=_index(),
        )
    assert completer.names == []


@pytest.mark.asyncio
async def test_passage_embeddings_are_saved_and_reused(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Title\n\nAlpha reduces measured error in the lab.\n", encoding="utf-8")
    settings = Settings(
        gemini_api_key="test-key",
        corpus_dir=corpus,
        output_dir=tmp_path / "out",
        embedding_model="gemini-embedding-2",
    )

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def get_embeddings(self, values: list[str], *, options: object = None) -> list[SimpleNamespace]:
            self.calls += 1
            return [SimpleNamespace(vector=[1.0, 0.0]) for _ in values]

    clear_index_cache()
    client = Client()
    await prepare_corpus(settings, client)
    await prepare_corpus(settings, client)
    assert client.calls == 1
    clear_index_cache()
    await prepare_corpus(settings, client)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_retrieval_with_no_hits_raises() -> None:
    index = PassageIndex(
        [Passage(passage_id="a#c0", paper_id="a", title="A", text="catalysis recall only")]
    )

    class Ctx:
        async def send_message(self, message: ResearchDossier) -> None:
            raise AssertionError("should not send")

    agent = RetrievalAgent(index, top_k=2)
    with pytest.raises(RetrievalError, match="will not invent sources"):
        await agent.retrieve(
            ResearchDossier(question="q", sub_queries=["zebrafish sonnet form"]),
            Ctx(),  # type: ignore[arg-type]
        )
