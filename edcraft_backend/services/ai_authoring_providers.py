"""Adapt validator model providers and own their client lifecycle."""

from __future__ import annotations

from typing import Any

from edcraft_validator.llm.forced_decision import ClassDecision
from edcraft_validator.llm.llm_contracts import (
    ModelProvider,
    StructuredGenerationRequest,
)
from edcraft_validator.llm.llm_errors import DecisionCapabilityError
from openai import OpenAI


class ManagedProvider:
    """Adapt a validator provider and own its OpenAI client lifecycle."""

    def __init__(
        self,
        delegate: ModelProvider,
        client: OpenAI,
        *,
        provider: str,
        model: str,
    ) -> None:
        self._delegate = delegate
        self._client = client
        self.provider = provider
        self.model = model

    def generation_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = self._delegate.generation_settings()
        return settings

    def generate(self, request: StructuredGenerationRequest[Any]) -> Any:
        return self._delegate.generate(request)

    def decide(
        self, messages: list[dict[str, Any]], labels: tuple[str, ...]
    ) -> ClassDecision:
        """Expose the delegate's forced classification for advisory checks."""
        decide = getattr(self._delegate, "decide", None)
        if not callable(decide):
            raise DecisionCapabilityError("Provider does not support token-probability decisions")
        decision: ClassDecision = decide(messages, labels)
        return decision

    @property
    def last_usage(self) -> Any:
        """Forward provider usage so relevance evidence stays complete."""
        return getattr(self._delegate, "last_usage", None)

    def close(self) -> None:
        self._client.close()
