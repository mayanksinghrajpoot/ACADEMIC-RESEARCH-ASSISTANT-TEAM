import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from research_assistant.llm import ModelServiceError, public_error
from research_assistant.memory.dossier import Claim, Passage, ResearchDossier, Verdict
from research_assistant.reporting import report_view
from research_assistant.ui.server import ResearchServer


def _start(tmp_path: Path, runner) -> tuple[ResearchServer, str]:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "note.md").write_text("# A Real Title\n\nBody of the note.\n", encoding="utf-8")
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=test-key\nCORPUS_DIR=corpus\nOUTPUT_DIR=outputs\n", encoding="utf-8")
    server = ResearchServer(("127.0.0.1", 0), tmp_path, runner=runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


def test_report_view_uses_supported_claims_only() -> None:
    dossier = ResearchDossier(
        question="Does alpha help?",
        sub_queries=["alpha error"],
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
        report_markdown=(
            "# Literature summary\n\n## Overview\n\nAlpha reduces measured error.\n\n## Themes\n\n"
        ),
    )
    view = report_view(dossier)
    assert view["overview"] == "Alpha reduces measured error."
    assert view["themes"] == [
        {
            "theme": "Effect",
            "claims": [
                {
                    "text": "Alpha reduces measured error.",
                    "quote": "Alpha reduces measured error when the passage is retrieved and shown.",
                    "sources": [{"title": "Paper A", "passage_id": "paper-a#c0"}],
                }
            ],
        }
    ]
    assert view["dropped"] == [{"text": "Alpha cures every disease.", "reason": "No retrieved passage."}]


def test_provider_errors_become_one_sentence() -> None:
    raw = (
        "Gemini chat request failed: 503 UNAVAILABLE. "
        "{'error': {'code': 503, 'message': 'This model is currently experiencing high demand. "
        "Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
    )
    assert public_error(RuntimeError(raw)) == (
        "Gemini is busy right now. Wait a moment and run the question again."
    )
    assert public_error(ModelServiceError("A research question is required.")) == (
        "A research question is required."
    )


def test_page_shows_a_busy_model_as_a_sentence(tmp_path: Path) -> None:
    async def runner(question: str, settings: object) -> ResearchDossier:
        raise ModelServiceError("Gemini is busy right now. Wait a moment and run the question again.")

    server, base = _start(tmp_path, runner=runner)
    try:
        request = urllib.request.Request(
            base + "/api/research",
            data=json.dumps({"question": "Does alpha help?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        body = json.load(caught.value)
        assert "Gemini is busy right now" in body["error"]
        assert "ServerError" not in body["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_page_lists_papers_and_rejects_an_empty_question(tmp_path: Path) -> None:
    server, base = _start(tmp_path, runner=None)
    try:
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert "Academic Research Assistant" in page
        corpus = json.load(urllib.request.urlopen(base + "/api/corpus"))
        assert corpus["papers"] == [{"id": "note", "title": "A Real Title"}]

        request = urllib.request.Request(
            base + "/api/research",
            data=json.dumps({"question": "   "}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        body = json.load(caught.value)
        assert caught.value.code == 400
        assert "required" in body["error"].lower()
    finally:
        server.shutdown()
        server.server_close()


def test_page_renders_a_finished_dossier(tmp_path: Path) -> None:
    async def runner(question: str, settings: object) -> ResearchDossier:
        return ResearchDossier(
            question=question.strip(),
            sub_queries=["alpha error"],
            passages=[
                Passage(passage_id="paper-a#c0", paper_id="paper-a", title="Paper A", text="Alpha reduces error.")
            ],
            claims=[
                Claim(claim_id="c1", theme="Effect", text="Alpha reduces error.", passage_ids=["paper-a#c0"])
            ],
            verdicts=[
                Verdict(claim_id="c1", verdict="supported", reason="Quoted.", supporting_quote="Alpha reduces error.")
            ],
            report_markdown="# Literature summary\n\n## Overview\n\nAlpha reduces error.\n\n## Themes\n",
        )

    server, base = _start(tmp_path, runner=runner)
    try:
        request = urllib.request.Request(
            base + "/api/research",
            data=json.dumps({"question": "Does alpha help?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.load(urllib.request.urlopen(request))
        assert payload["overview"] == "Alpha reduces error."
        assert payload["themes"][0]["theme"] == "Effect"
        assert payload["markdown_path"].endswith(".md")
        assert Path(payload["markdown_path"]).is_file()
    finally:
        server.shutdown()
        server.server_close()
