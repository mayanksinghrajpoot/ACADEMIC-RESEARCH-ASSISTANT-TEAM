"""Chat completion through a Microsoft Agent Framework agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from agent_framework import Agent, ChatOptions
from agent_framework.exceptions import ChatClientException
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_QUOTED_MESSAGE = re.compile(r"""['"]message['"]:\s*['"]([^'"]+)['"]""")
# gemini-3.8-flash often returns 503/504. These models answered on the same key.
_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)


class ModelServiceError(RuntimeError):
    """A model call failed with a message that can be shown to the reader."""


class Completer(Protocol):
    """Structured completion used by every language-model agent."""

    async def complete(
        self,
        *,
        name: str,
        instructions: str,
        prompt: str,
        response_format: type[ModelT],
    ) -> ModelT: ...


class FrameworkCompleter:
    """Runs one Agent Framework agent per call and parses a schema."""

    def __init__(self, client: object) -> None:
        self._client = client

    async def complete(
        self,
        *,
        name: str,
        instructions: str,
        prompt: str,
        response_format: type[ModelT],
    ) -> ModelT:
        agent = Agent(
            client=self._client,
            name=name,
            instructions=instructions,
            default_options=ChatOptions(temperature=0),
        )
        primary = str(getattr(self._client, "model", "") or "")
        response = None
        last_error: ChatClientException | None = None
        for model in chat_models(primary):
            options: ChatOptions = {"temperature": 0, "response_format": response_format}
            if model:
                options["model"] = model
            try:
                response = await agent.run(prompt, options=options)
                break
            except ChatClientException as exc:
                last_error = exc
                if not is_temporary(exc):
                    raise ModelServiceError(public_error(exc)) from exc
                logger.warning(
                    "Gemini model %s is busy; trying another model for %s.",
                    model or "default",
                    name,
                )
        if response is None:
            error = last_error or RuntimeError("model request failed")
            raise ModelServiceError(public_error(error)) from error
        try:
            value = response.value
        except (ValidationError, ValueError) as exc:
            logger.warning("Structured parse failed for %s (%s); reading the message text.", name, exc)
            value = _parse_json(response.text, response_format)
        if isinstance(value, response_format):
            return value
        return response_format.model_validate(value)


def chat_models(primary: str) -> list[str]:
    """Primary model first, then models that still answer when the newest flash is busy."""
    models: list[str] = []
    for model in (primary, *_FALLBACK_MODELS):
        if model and model not in models:
            models.append(model)
    return models


def public_error(exc: BaseException) -> str:
    """Turn a provider exception into one sentence for the page or the terminal."""
    text = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
    match = _QUOTED_MESSAGE.search(text)
    message = match.group(1) if match else ""
    lowered = f"{message} {text}".lower()
    if any(token in lowered for token in ("high demand", "503", "504", "unavailable", "deadline")):
        return "Gemini is busy right now. Wait a moment and run the question again."
    if "no longer available" in lowered:
        return "The configured Gemini model is no longer available. Set GOOGLE_MODEL in .env to a current model."
    if message:
        return message
    first = text.strip().splitlines()[0]
    if first and len(first) < 300 and not first.startswith("(") and "ServerError" not in first:
        return first
    return "The model request failed. Check the API key and model name, then try again."


def is_temporary(exc: BaseException) -> bool:
    """True when Gemini asked the caller to retry."""
    text = str(exc).lower()
    return any(
        token in text for token in ("503", "504", "unavailable", "high demand", "429", "deadline")
    )


def _parse_json(text: str, response_format: type[ModelT]) -> ModelT:
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"{response_format.__name__} response did not contain a JSON object.")
    payload = json.loads(candidate[start : end + 1])
    return response_format.model_validate(payload)
