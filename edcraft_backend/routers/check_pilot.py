"""Authenticated, owner-scoped endpoints for the reusable check pilot."""

from typing import Any

from edcraft_validator.application.check_pilot import (
    CheckRecord,
    PilotBatch,
    PilotTemplateRecord,
)
from fastapi import APIRouter

from edcraft_backend.dependencies import (
    CheckPilotServiceDep,
    CurrentUserDep,
    ProviderConnectionServiceDep,
)
from edcraft_backend.schemas.ai_authoring import ErrorResponse
from edcraft_backend.schemas.check_pilot import (
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Not signed in."},
    404: {"model": ErrorResponse, "description": "Unknown or unowned pilot record."},
    409: {"model": ErrorResponse, "description": "Stale digest or immutable conflict."},
    422: {"model": ErrorResponse, "description": "Unsupported pilot request."},
    429: {"model": ErrorResponse, "description": "Check authoring admission is full."},
    503: {
        "model": ErrorResponse,
        "description": "Pilot worker, provider or storage failed.",
    },
    504: {"model": ErrorResponse, "description": "Pilot worker or provider timed out."},
}

router = APIRouter(
    prefix="/ai-authoring/check-pilot",
    tags=["ai-authoring"],
    responses=_ERROR_RESPONSES,
)


@router.post("/templates", response_model=PilotTemplateRecord)
async def author_template(
    payload: PilotTemplateRequest,
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
    connections: ProviderConnectionServiceDep,
) -> PilotTemplateRecord:
    """Author one qualified private template with the resolved author model."""
    connection: ResolvedConnection | None = None
    if payload.connection_id is not None:
        connection = await connections.resolve(user.id, payload.connection_id, payload.model)
    return await service.author_template(user.id, payload, connection=connection)


@router.get("/templates", response_model=list[PilotTemplateRecord])
async def list_templates(
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> list[PilotTemplateRecord]:
    """List only the signed-in owner's private pilot templates."""
    return await service.list_templates(user.id)


@router.get("/templates/{template_id}", response_model=PilotTemplateRecord)
async def get_template(
    template_id: str,
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> PilotTemplateRecord:
    """Return one owned template with its check evidence and approval."""
    return await service.get_template(user.id, template_id)


@router.get("/checks", response_model=list[CheckRecord])
async def list_checks(
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> list[CheckRecord]:
    """List only the signed-in owner's qualified private checks."""
    return await service.list_checks(user.id)


@router.get("/checks/{check_id}", response_model=CheckRecord)
async def get_check(
    check_id: str,
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> CheckRecord:
    """Return one owned immutable check with its qualification evidence."""
    return await service.get_check(user.id, check_id)


@router.post("/templates/{template_id}/approval", response_model=PilotTemplateRecord)
async def decide_template(
    template_id: str,
    payload: PilotApprovalRequest,
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> PilotTemplateRecord:
    """Approve or reject the exact template and check digests shown."""
    return await service.decide(user.id, template_id, payload)


@router.post("/templates/{template_id}/generate", response_model=PilotBatch)
async def generate_questions(
    template_id: str,
    payload: PilotGenerateRequest,
    user: CurrentUserDep,
    service: CheckPilotServiceDep,
) -> PilotBatch:
    """Generate a deterministic seeded batch from an approved template."""
    return await service.generate(user.id, template_id, payload)
