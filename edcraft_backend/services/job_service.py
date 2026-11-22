"""Service for submitting and managing async Nomad jobs."""

import json
import logging
from typing import Any
from uuid import UUID

from edcraft_backend.config import settings
from edcraft_backend.exceptions import (
    InvalidTokenError,
    ResourceNotFoundError,
    UnauthorizedAccessError,
)
from edcraft_backend.executors.nomad import NomadExecutor
from edcraft_backend.models.job import Job, JobStatus, JobType
from edcraft_backend.repositories.job_repository import (
    JobRepository,
    JobTokenRepository,
)
from edcraft_backend.security import generate_token
from edcraft_backend.services.post_processing_service import PostProcessingService

logger = logging.getLogger(__name__)


class JobService:
    """Orchestrates job submission, callback handling, and status retrieval."""

    def __init__(
        self,
        job_repo: JobRepository,
        job_token_repo: JobTokenRepository,
        executor: NomadExecutor | None,
        post_processing_svc: PostProcessingService,
    ) -> None:
        self.job_repo = job_repo
        self.job_token_repo = job_token_repo
        self.executor = executor
        self.post_processing_svc = post_processing_svc

    async def submit(
        self,
        job_type: JobType,
        params: dict[str, object],
        user_id: UUID | None = None,
    ) -> Job:
        """Create a Job record, issue a callback token, and submit to Nomad."""
        job = Job(type=job_type.value, status=JobStatus.QUEUED.value, user_id=user_id)
        job = await self.job_repo.create(job)
        logger.info("Job created", extra={"job_id": job.id, "job_type": job_type.value})

        token = generate_token(32)
        await self.job_token_repo.create(token=token, job_id=job.id)

        callback_url = f"{settings.nomad.callback_base_url}/jobs/callback/{token}"
        nomad_job_id = f"edcraft-{job_type.value.replace('_', '-')}-{job.id}"

        if self.executor is None:
            raise RuntimeError("No executor configured")
        try:
            await self.executor.submit_job(
                nomad_job_id=nomad_job_id,
                job_type=job_type.value,
                params=params,
                callback_url=callback_url,
            )
            await self.job_repo.update_status(
                job.id, JobStatus.RUNNING.value, nomad_job_id=nomad_job_id
            )
        except Exception:
            logger.exception("Job submission failed", extra={"job_id": job.id})
            await self.job_repo.update_status(job.id, JobStatus.FAILED.value)
            raise

        logger.info("Job submitted", extra={"job_id": job.id, "nomad_job_id": nomad_job_id})
        return job

    async def on_callback(
        self,
        token: str,
        result_json: str | None,
        error: str | None,
    ) -> None:
        """Validate the one-time token and persist the processed job result.

        Called by the Nomad worker via POST /jobs/callback/{token}.
        Post-processes the raw engine result based on job type before storing.
        """
        job_token = await self.job_token_repo.get_valid_by_token(token)
        if job_token is None:
            raise InvalidTokenError("Invalid or already consumed callback token")

        job = await self.job_repo.get_by_id(job_token.job_id)
        if job is None:
            return
        if job.type == JobType.AI_AUTHORING:
            raise InvalidTokenError("Invalid or already consumed callback token")

        await self.job_token_repo.consume(token)

        if error or result_json is None:
            logger.error(
                "Job callback received with error",
                extra={"job_id": job_token.job_id, "error": error},
            )
            await self.job_repo.complete(
                job_id=job_token.job_id,
                result_json=None,
                error=error,
            )
            return

        processed_json: str | None = result_json
        process_error: str | None = None

        try:
            raw = json.loads(result_json)
            processed = await self._post_process(job.type, raw)
            processed_json = json.dumps(processed, default=str)
        except Exception as exc:
            logger.exception(
                "Post-processing failed for job %s (type=%s)", job.id, job.type
            )
            process_error = str(exc)
            processed_json = None

        logger.info("Job callback completed", extra={"job_id": job_token.job_id})
        await self.job_repo.complete(
            job_id=job_token.job_id,
            result_json=processed_json,
            error=process_error,
        )

    async def get_job(self, job_id: UUID, user_id: UUID | None = None) -> Job:
        """Fetch a job, enforcing ownership for jobs that have an owner.

        Anonymous jobs (user_id=None) are accessible to anyone with the job_id.
        Owned jobs are restricted to the owning user.
        """
        job = await self.job_repo.get_by_id(job_id)
        if job is None or job.type == JobType.AI_AUTHORING:
            raise ResourceNotFoundError("Job", str(job_id))
        if job.user_id is not None and job.user_id != user_id:
            raise UnauthorizedAccessError("Job", str(job_id))

        if job.status == JobStatus.RUNNING and job.nomad_job_id and self.executor:
            status = await self.executor.get_job_status(job.nomad_job_id)
            if status != JobStatus.RUNNING:
                await self.job_repo.update_status(job.id, status)
                job.status = status

        return job

    async def _post_process(
        self,
        job_type: str,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        if job_type == JobType.GENERATE_TEMPLATE:
            return self.post_processing_svc.post_process_generate_template(raw)
        if job_type == JobType.QUESTION_FROM_TEMPLATE:
            return self.post_processing_svc.post_process_question_from_template(raw)
        if job_type == JobType.ASSESSMENT_FROM_TEMPLATE:
            return await self.post_processing_svc.post_process_assessment_from_template(
                raw
            )
        return raw
