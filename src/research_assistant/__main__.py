"""Command line entry: one research question in, a sourced summary out."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from research_assistant.agents.retrieval_agent import RetrievalError
from research_assistant.config import ConfigurationError, load_settings
from research_assistant.llm import ModelServiceError, public_error
from research_assistant.memory.dossier import ResearchDossier
from research_assistant.orchestration.pipeline import run_research
from research_assistant.rag.corpus import CorpusError


def write_outputs(dossier: ResearchDossier, output_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Write the markdown summary and the full dossier JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{run_id}.md"
    json_path = output_dir / f"{run_id}.json"
    markdown_path.write_text(dossier.report_markdown, encoding="utf-8")
    json_path.write_text(dossier.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Produce a sourced literature summary for a research question.",
    )
    parser.add_argument("question", nargs="+", help="Research question to investigate in the local corpus.")
    parser.add_argument("--corpus", type=Path, default=None, help="Directory of .md or .txt papers.")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(Path.cwd())
        if args.corpus is not None:
            corpus = args.corpus if args.corpus.is_absolute() else Path.cwd() / args.corpus
            settings = settings.model_copy(update={"corpus_dir": corpus})
        settings.require_api_key()
        dossier = asyncio.run(run_research(" ".join(args.question), settings))
    except (ConfigurationError, CorpusError, RetrievalError, ValueError, ModelServiceError) as exc:
        print(f"error: {public_error(exc)}", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    markdown_path, json_path = write_outputs(dossier, settings.output_dir, run_id)
    print(dossier.report_markdown)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    if settings.log_path is not None:
        print(f"Appended {settings.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
