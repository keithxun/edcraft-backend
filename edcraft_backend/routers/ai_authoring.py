"""Owner-scoped AI authoring endpoints for issues 48-50."""

from typing import Any, TypedDict
from uuid import UUID

from fastapi import APIRouter, status

from edcraft_backend.dependencies import (
    AiAuthoringServiceDep,
    CurrentUserDep,
    ProviderConnectionServiceDep,
)
from edcraft_backend.routers import ai_authoring_materials
from edcraft_backend.schemas.ai_authoring import (
    ApprovalRequest,
    ApprovalResponse,
    AuthoringCapabilitiesResponse,
    AuthoringJobStatusResponse,
    AuthoringJobSubmittedResponse,
    AuthoringJobSummary,
    AuthoringModelSelection,
    AuthoringReviewResponse,
    ErrorResponse,
    GeneratedQuestionsBatch,
    GenerateRequest,
    JudgeCompatibilityRequest,
    JudgeCompatibilityResponse,
    PreviewRequest,
    PreviewResponse,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.provider_connection_service import ProviderConnectionService
from edcraft_backend.services.provider_endpoint import ResolvedConnection

# FastAPI keeps its standard HTTPValidationError 422 body for request
# validation. AuthoringValidationError can also return a sanitized
# ErrorResponse with a string detail (for configured limits), so clients must
# accept either shape for 422 rather than treating every 422 as a string.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Not signed in."},
    404: {"model": ErrorResponse, "description": "Unknown or unowned job."},
    409: {"model": ErrorResponse, "description": "Stale or unusable artifact."},
    429: {"model": ErrorResponse, "description": "Authoring admission is full."},
    503: {
        "model": ErrorResponse,
        "description": "AI authoring storage is unavailable.",
    },
}

router = APIRouter(
    prefix="/ai-authoring",
    tags=["ai-authoring"],
    responses=_ERROR_RESPONSES,
)
router.include_router(ai_authoring_materials.router)


@router.get("/capabilities", response_model=AuthoringCapabilitiesResponse)
async def get_capabilities(
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
    connections: ProviderConnectionServiceDep,
) -> AuthoringCapabilitiesResponse:
    """Return server profiles and the caller's own BYOK connections."""
    return await service.capabilities(await connections.list_connections(user.id))


class _ResolvedModelConnections(TypedDict):
    author_connection: ResolvedConnection | None
    difficulty_connection: ResolvedConnection | None
    relevance_connection: ResolvedConnection | None


async def _resolve_model_connections(
    payload: JudgeCompatibilityRequest,
    owner_id: UUID,
    connections: ProviderConnectionService,
) -> _ResolvedModelConnections:
    async def resolve_connection(
        selection: AuthoringModelSelection | None,
    ) -> ResolvedConnection | None:
        if selection is None or selection.connection_id is None:
            return None
        return await connections.resolve(
            owner_id, selection.connection_id, selection.model
        )

    judges = payload.judges
    return _ResolvedModelConnections(
        author_connection=await resolve_connection(payload),
        difficulty_connection=await resolve_connection(
            judges.difficulty if judges else None
        ),
        relevance_connection=await resolve_connection(
            judges.relevance if judges else None
        ),
    )


@router.post("/judge-compatibility", response_model=JudgeCompatibilityResponse)
async def check_judge_compatibility(
    payload: JudgeCompatibilityRequest,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
    connections: ProviderConnectionServiceDep,
) -> JudgeCompatibilityResponse:
    """Check the chosen advisory models without sending question data."""
    resolved = await _resolve_model_connections(payload, user.id, connections)
    return await service.judge_compatibility(payload, **resolved)


@router.post(
    "/jobs",
    response_model=AuthoringJobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_job(
    payload: SubmitAuthoringJobRequest,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
    connections: ProviderConnectionServiceDep,
) -> AuthoringJobSubmittedResponse:
    """Admit one authoring job and return its polling URL."""
    resolved = await _resolve_model_connections(payload, user.id, connections)
    job = await service.submit(user.id, payload, **resolved)
    return AuthoringJobSubmittedResponse(
        job_id=job.job_id,
        status_url=f"/ai-authoring/jobs/{job.job_id}",
    )


@router.get("/jobs", response_model=list[AuthoringJobSummary])
async def list_jobs(
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> list[AuthoringJobSummary]:
    """Return recent jobs owned by the signed-in user."""
    jobs = await service.list_jobs(user.id)
    return [service.job_summary(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=AuthoringJobStatusResponse)
async def get_job(
    job_id: UUID,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> AuthoringJobStatusResponse:
    """Poll one owned job without exposing the internal report."""
    job = await service.get_job(user.id, job_id)
    return service.job_status(job)


@router.get("/jobs/{job_id}/review", response_model=AuthoringReviewResponse)
async def review_job(
    job_id: UUID,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> AuthoringReviewResponse:
    """Return safe prompt, plan, attempts, findings and exact artifact identity."""
    return await service.review(user.id, job_id)


@router.post("/jobs/{job_id}/preview", response_model=PreviewResponse)
async def preview_job(
    job_id: UUID,
    payload: PreviewRequest,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> PreviewResponse:
    """Generate one deterministic draft locally from a checked artifact."""
    return await service.preview(user.id, job_id, payload.seed)


@router.post("/jobs/{job_id}/approval", response_model=ApprovalResponse)
async def decide_job(
    job_id: UUID,
    payload: ApprovalRequest,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> ApprovalResponse:
    """Approve or reject the exact artifact digest shown during review."""
    return await service.decide(user.id, job_id, payload)


@router.post("/jobs/{job_id}/generate", response_model=GeneratedQuestionsBatch)
async def generate_job(
    job_id: UUID,
    payload: GenerateRequest,
    user: CurrentUserDep,
    service: AiAuthoringServiceDep,
) -> GeneratedQuestionsBatch:
    """Return deterministic consecutive questions from an approved artifact."""
    return await service.generate(user.id, job_id, payload)
