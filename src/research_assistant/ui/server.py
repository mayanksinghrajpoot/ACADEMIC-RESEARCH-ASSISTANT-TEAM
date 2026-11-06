"""Local HTTP server for the research assistant page."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_assistant.__main__ import write_outputs
from research_assistant.agents.retrieval_agent import RetrievalError
from research_assistant.config import ConfigurationError, Settings, load_settings
from research_assistant.llm import ModelServiceError, public_error
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.orchestration.pipeline import run_research
from research_assistant.rag.corpus import CorpusError, load_corpus
from research_assistant.reporting import report_view

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_BODY = 32_768

_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ResearchServer(ThreadingHTTPServer):
    """Serves the page and one research request at a time per connection."""

    def __init__(
        self,
        address: tuple[str, int],
        project_root: Path,
        runner: Any = run_research,
    ) -> None:
        self.project_root = project_root
        self.runner = runner
        super().__init__(address, ResearchHandler)


class ResearchHandler(BaseHTTPRequestHandler):
    server: ResearchServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in _FILES:
            name, content_type = _FILES[path]
            self._send_bytes((STATIC_DIR / name).read_bytes(), content_type)
            return
        if path == "/api/corpus":
            self._send_corpus()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/research":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > _MAX_BODY:
            self._send_json(400, {"error": "The question is too long."})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "The request was not valid JSON."})
            return
        question = payload.get("question", "")
        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"error": "A research question is required."})
            return
        try:
            settings = load_settings(self.server.project_root)
            settings.require_api_key()
            dossier = asyncio.run(self.server.runner(question, settings))
        except (ConfigurationError, CorpusError, RetrievalError, ValueError, ModelServiceError) as exc:
            self._send_json(400, {"error": public_error(exc)})
            return
        except Exception as exc:
            logger.exception("Research run failed")
            self._send_json(500, {"error": public_error(exc)})
            return
        if not isinstance(dossier, ResearchDossier):
            self._send_json(500, {"error": "The research run did not return a dossier."})
            return
        self._send_json(200, _result_payload(dossier, settings))

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_corpus(self) -> None:
        try:
            settings = load_settings(self.server.project_root)
            papers = load_corpus(settings.corpus_dir)
        except (ConfigurationError, CorpusError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(
            200,
            {"papers": [{"id": paper.paper_id, "title": paper.title} for paper in papers]},
        )

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _result_payload(dossier: ResearchDossier, settings: Settings) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    markdown_path, json_path = write_outputs(dossier, settings.output_dir, run_id)
    view = report_view(dossier)
    view["markdown_path"] = str(markdown_path)
    view["json_path"] = str(json_path)
    return view


def serve(host: str, port: int, project_root: Path) -> None:
    """Serve the page until the process is stopped."""
    try:
        server = ResearchServer((host, port), project_root)
    except OSError as exc:
        raise SystemExit(f"error: could not listen on {host}:{port} ({exc})") from exc
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Research assistant at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
