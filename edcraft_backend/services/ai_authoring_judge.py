"""Wire authoring and advisory models for one authoring job."""

from __future__ import annotations

from edcraft_validator.llm.decision import ConfiguredDecisionProvider
from edcraft_validator.llm.judge_configuration import JudgeSettings, judge_settings_for
from edcraft_validator.llm.llm_contracts import ModelProvider, supports_decisions
from edcraft_validator.llm.provider_registry import create_model_provider
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server

from edcraft_backend.services.ai_authoring_runtime import AuthoringValidationError
from edcraft_backend.services.provider_endpoint import ResolvedConnection


class SelectedModelDecisionProvider(ConfiguredDecisionProvider):
    """Relevance classifier that reuses the model selected for the check."""

    def __init__(self, settings: JudgeSettings, provider: ModelProvider) -> None:
        if not supports_decisions(provider):
            raise ValueError("Provider does not support token-probability decisions")
        self.settings = settings
        self.delegate = provider
        self.provider = provider.provider
        self.model = provider.model
        self.threshold = settings.relevance_threshold
        self.served_model = None


def build_plan_provider(role: ResolvedConnection) -> ModelProvider:
    """Build one role's provider from a BYOK connection or a server profile."""
    if role.is_byok:
        return role.open_provider()
    return create_model_provider(role.selection)


def _named_role(role: ResolvedConnection) -> tuple[str, str]:
    model = role.selection.model
    if not isinstance(model, str):
        raise AuthoringValidationError("An advisory model must name a model.")
    return role.selection.provider, model


def build_model_tool_client(
    *,
    difficulty: ResolvedConnection,
    relevance: ResolvedConnection,
) -> FastMcpToolClient:
    """Build validation tools with separately selected advisory models."""
    difficulty_provider, difficulty_model = _named_role(difficulty)
    relevance_provider, relevance_model = _named_role(relevance)
    difficulty_settings = judge_settings_for(difficulty_provider, difficulty_model)
    relevance_settings = judge_settings_for(relevance_provider, relevance_model)
    return FastMcpToolClient(
        create_validation_server(
            judge_settings=difficulty_settings,
            difficulty_provider_factory=lambda: build_plan_provider(difficulty),
            decision_provider_factory=lambda: SelectedModelDecisionProvider(
                relevance_settings, build_plan_provider(relevance)
            ),
        )
    )
