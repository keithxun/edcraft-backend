"""Token-authenticated callbacks used by Nomad authoring workers.

There is deliberately no user authentication here: the unguessable token in
the path is the credential. The backend still owns every durable write.
"""

from __future__ import annotations

from typing import Any

from edcraft_validator.application.authoring_api import PublicAuthoringError
from edcraft_validator.application.authoring_contracts import AuthoringProgress
from fastapi import APIRouter, HTTPException, Response, status

from edcraft_backend.dependencies import AiAuthoringServiceDep
from edcraft_backend.schemas.ai_authoring import (
    WorkerErrorRequest,
    WorkerResultRequest,
)
from edcraft_backend.services.ai_authoring_service import LIBRARY_REVISION

router = APIRouter(
    prefix="/ai-authoring/worker",
    tags=["ai-authoring-worker"],
    include_in_schema=False,
)


@router.get("/{token}/context")
async def worker_context(
    token: str,
    service: AiAuthoringServiceDep,
    response: Response,
) -> dict[str, Any]:
    """Return the command and decrypted model plan for one worker job."""
    context = await service.worker_context(token)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return context


@router.post("/{token}/progress", status_code=status.HTTP_204_NO_CONTENT)
async def worker_progress(
    token: str,
    payload: AuthoringProgress,
    service: AiAuthoringServiceDep,
) -> Response:
    """Persist the latest progress event from a worker (best effort)."""
    await service.update_worker_progress(token, payload)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{token}/result", status_code=status.HTTP_204_NO_CONTENT)
async def worker_result(
    token: str,
    payload: WorkerResultRequest,
    service: AiAuthoringServiceDep,
) -> Response:
    """Accept the exact final AuthoringResult from a worker."""
    if payload.library_revision != LIBRARY_REVISION:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Worker library revision does not match the backend.",
        )
    await service.complete_worker_result(token, payload.report)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{token}/error", status_code=status.HTTP_204_NO_CONTENT)
async def worker_error(
    token: str,
    payload: WorkerErrorRequest,
    service: AiAuthoringServiceDep,
) -> Response:
    """Fail a job when the worker cannot produce an AuthoringResult."""
    await service.fail_worker_job(
        token,
        PublicAuthoringError(code=payload.code, message=payload.message),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
