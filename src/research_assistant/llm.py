"""Chat completion through a Microsoft Agent Framework agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from agent_framework import Agent, ChatOptions
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


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
        response = await agent.run(
            prompt,
            options=ChatOptions(temperature=0, response_format=response_format),
        )
        try:
            value = response.value
        except (ValidationError, ValueError) as exc:
            logger.warning("Structured parse failed for %s (%s); reading the message text.", name, exc)
            value = _parse_json(response.text, response_format)
        if isinstance(value, response_format):
            return value
        return response_format.model_validate(value)


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
