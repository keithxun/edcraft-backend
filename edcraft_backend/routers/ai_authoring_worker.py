"""Private attempt-scoped callbacks for the authoring worker."""

from typing import Annotated, Any
from uuid import UUID

from edcraft_validator.application.authoring_contracts import AuthoringProgress
from fastapi import APIRouter, Depends, Header, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from edcraft_backend.dependencies import AiAuthoringServiceDep
from edcraft_backend.services.ai_authoring_dispatch import AuthoringWorkerUnauthorized

router = APIRouter(prefix="/ai-authoring/worker", include_in_schema=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def _worker_token(
    auth: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> str:
    if auth is None:
        raise AuthoringWorkerUnauthorized()
    return auth.credentials


WorkerToken = Annotated[str, Depends(_worker_token)]
WorkerAllocation = Annotated[UUID, Header(alias="X-EdCraft-Worker-Allocation")]


class _WorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library_revision: str = Field(min_length=1, max_length=100)
    report: dict[str, Any]


class _WorkerError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)


@router.get("/{job_id}/{attempt_id}/context")
async def worker_context(
    job_id: UUID,
    attempt_id: UUID,
    token: WorkerToken,
    allocation: WorkerAllocation,
    service: AiAuthoringServiceDep,
    response: Response,
) -> dict[str, object]:
    context = await service.worker_lifecycle().worker_context(
        job_id, attempt_id, token, str(allocation)
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return context


@router.post("/{job_id}/{attempt_id}/progress", status_code=204)
async def worker_progress(
    job_id: UUID,
    attempt_id: UUID,
    token: WorkerToken,
    allocation: WorkerAllocation,
    service: AiAuthoringServiceDep,
    payload: AuthoringProgress,
) -> None:
    await service.worker_lifecycle().worker_progress(
        job_id, attempt_id, token, str(allocation), payload
    )


@router.post("/{job_id}/{attempt_id}/result", status_code=204)
async def worker_result(
    job_id: UUID,
    attempt_id: UUID,
    token: WorkerToken,
    allocation: WorkerAllocation,
    service: AiAuthoringServiceDep,
    payload: _WorkerResult,
) -> None:
    await service.worker_lifecycle().worker_result(
        job_id, attempt_id, token, str(allocation), payload.library_revision, payload.report
    )


@router.post("/{job_id}/{attempt_id}/error", status_code=204)
async def worker_error(
    job_id: UUID,
    attempt_id: UUID,
    token: WorkerToken,
    allocation: WorkerAllocation,
    service: AiAuthoringServiceDep,
    payload: _WorkerError,
) -> None:
    # Worker-provided messages are private, untrusted input.
    await service.worker_lifecycle().worker_error(job_id, attempt_id, token, str(allocation))
