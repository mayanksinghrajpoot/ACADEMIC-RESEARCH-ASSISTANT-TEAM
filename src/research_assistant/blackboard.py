"""One shared dossier per run, plus the append-only tool log."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agent_framework import FunctionTool

from research_assistant.memory.dossier import ResearchDossier

_CURRENT: ContextVar[RunBindings | None] = ContextVar("research_blackboard", default=None)
_LOG_LOCK = threading.Lock()
_DETAIL_LIMIT = 180


@dataclass
class RunBindings:
    """Everything the tools can see for the run that is in progress.

    ``dossier`` is the object every agent reads and writes. Tools must not
    replace it.
    """

    dossier: ResearchDossier
    papers: list[Any] = field(default_factory=list)
    index: Any = None
    embedder: Any = None
    top_k: int = 4
    revision_cap: int = 1
    log_path: Path | None = None
    run_id: str = ""
    query_vectors: list[list[float] | None] | None = None


def get_bindings() -> RunBindings:
    bindings = _CURRENT.get()
    if bindings is None:
        raise RuntimeError("No shared dossier is bound for this run.")
    return bindings


@contextmanager
def bind_run(bindings: RunBindings) -> Iterator[RunBindings]:
    """Bind the shared dossier for the current task and every tool it calls."""
    token = _CURRENT.set(bindings)
    try:
        yield bindings
    finally:
        _CURRENT.reset(token)


def use_shared(message: ResearchDossier, **fields: Any) -> ResearchDossier:
    """Return the run's dossier, creating a binding when a test calls one agent."""
    current = _CURRENT.get()
    if current is None:
        current = RunBindings(dossier=message)
        _CURRENT.set(current)
    for key, value in fields.items():
        setattr(current, key, value)
    return current.dossier


async def invoke(agent: str, function_tool: FunctionTool, **kwargs: Any) -> Any:
    """Run one Agent Framework tool and append the call to the run log."""
    try:
        result = await function_tool.invoke(skip_parsing=True, **kwargs)
    except Exception as exc:
        _record(agent, function_tool.name, str(exc), ok=False)
        raise
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    _record(agent, function_tool.name, text, ok=True)
    return result


def record(agent: str, tool: str, detail: str, *, ok: bool) -> None:
    """Append one line for a step that is not a tool, such as the corpus check."""
    _record(agent, tool, detail, ok=ok)


def _record(agent: str, tool: str, detail: str, *, ok: bool) -> None:
    bindings = _CURRENT.get()
    if bindings is None:
        return
    flat = " ".join(detail.split())
    if len(flat) > _DETAIL_LIMIT:
        flat = flat[: _DETAIL_LIMIT - 1].rstrip() + "…"
    status = "ok" if ok else "failed"
    line = f"{agent} · {tool} · {status} · {flat}"
    with _LOG_LOCK:
        bindings.dossier.trace.append(line)
        path = bindings.log_path
        if path is None:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": bindings.run_id,
            "agent": agent,
            "tool": tool,
            "ok": ok,
            "detail": flat,
            "dossier_id": id(bindings.dossier),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
