"""Wire authoring and advisory models for one authoring job."""

from __future__ import annotations

from edcraft_validator.llm.llm_contracts import ModelProvider
from edcraft_validator.llm.provider_registry import create_model_provider
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.job_judges import job_judge_client

from edcraft_backend.services.ai_authoring_runtime import AuthoringValidationError
from edcraft_backend.services.provider_endpoint import ResolvedConnection


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
    decision_fallback: ResolvedConnection | None = None,
) -> FastMcpToolClient:
    """Build validation tools with separately selected advisory models."""
    return job_judge_client(
        difficulty=difficulty,
        relevance=relevance,
        decision_fallback=decision_fallback,
        open_provider=build_plan_provider,
        model_name=_named_role,
    )
