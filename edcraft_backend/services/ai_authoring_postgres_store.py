"""Owner-scoped transactional storage for AI authoring jobs and evidence."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

from edcraft_validator.application.authoring_api import PublicAuthoringError
from edcraft_validator.application.authoring_contracts import AuthoringProgress, AuthoringResult
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import defer

from edcraft_backend.database import AsyncSessionLocal
from edcraft_backend.models.ai_authoring import AuthoringJobData
from edcraft_backend.models.job import Job, JobType
from edcraft_backend.schemas.ai_authoring import GenerationMetadata
from edcraft_backend.services.ai_authoring_store import (
    ApprovalRecord,
    AuthoringConflict,
    AuthoringEvidence,
    AuthoringNotFound,
    AuthoringStorageUnavailable,
    EvidenceRequirement,
    StoredJob,
    completed_job,
    verify_evidence,
)

logger = logging.getLogger(__name__)

# The common jobs row owns these fields. Keep only authoring fields in the record.
_ROW_FIELDS = {"job_id", "owner_id", "state", "created_at", "finished_at", "artifact_sha256"}


class PostgresAuthoringStore:
    """Persist authoring data and its shared jobs row within one transaction."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal) -> None:
        self._sessions = sessions

    async def ensure_ready(self) -> None:
        """Check that the configured database has the authoring schema."""
        async with self._transaction() as db:
            await db.execute(select(AuthoringJobData).limit(0))

    async def create_job(self, job: StoredJob) -> StoredJob:
        """Persist a new job; a shared jobs UUID collision is a conflict."""
        async with self._transaction() as db:
            self._create(db, job)
            try:
                await db.flush()
            except IntegrityError as exc:
                if getattr(exc.orig, "sqlstate", None) == "23505":
                    raise AuthoringConflict("Authoring job already exists.") from exc
                raise
        return job

    async def get_job(self, job_id: UUID) -> StoredJob:
        """Load an authoring job for internal lifecycle work."""
        async with self._transaction() as db:
            return self._snapshot(*await self._load(db, job_id))

    async def get_owned_job(self, job_id: UUID, owner_id: UUID) -> StoredJob:
        async with self._transaction() as db:
            return self._snapshot(*await self._load(db, job_id, owner_id=owner_id))

    async def list_jobs(self, owner_id: UUID, *, limit: int = 50) -> list[StoredJob]:
        async with self._transaction() as db:
            result = await db.execute(
                select(Job, AuthoringJobData)
                .join(AuthoringJobData)
                .where(Job.type == JobType.AI_AUTHORING, Job.user_id == owner_id)
                .options(
                    defer(AuthoringJobData.artifact_bytes), defer(AuthoringJobData.report_bytes)
                )
                .order_by(Job.created_at.desc(), Job.id.desc())
                .limit(limit)
            )
            return [self._snapshot(row, data) for row, data in result.tuples()]

    async def mark_started(self, job_id: UUID) -> StoredJob:
        async with self._transaction() as db:
            return self._start(*await self._load(db, job_id, lock=True))

    async def update_progress(self, job_id: UUID, progress: AuthoringProgress) -> None:
        try:
            async with self._transaction() as db:
                row, data = await self._load(db, job_id, lock=True)
                self._progress(row, data, progress)
        except (AuthoringNotFound, AuthoringConflict, AuthoringStorageUnavailable):
            logger.warning("Could not persist authoring progress")

    async def complete_job(self, job_id: UUID, report: AuthoringResult) -> StoredJob:
        async with self._transaction() as db:
            row, data = await self._load(db, job_id, lock=True)
            return self._complete(row, data, report)

    async def fail_job(self, job_id: UUID, error: PublicAuthoringError) -> StoredJob:
        async with self._transaction() as db:
            row, data = await self._load(db, job_id, lock=True)
            return self._fail(row, data, error)

    async def save_approval(self, job_id: UUID, approval: ApprovalRecord) -> StoredJob:
        async with self._transaction() as db:
            row, data = await self._load(
                db, job_id, owner_id=approval.reviewer_id, lock=True, evidence=True
            )
            evidence = self._evidence(
                row, data, EvidenceRequirement.REUSABLE, approval.artifact_sha256
            )
            job = evidence.job.model_copy(update={"approval": approval, "updated_at": _now()})
            self._write(row, data, job)
            return job

    async def save_generation(
        self,
        job_id: UUID,
        metadata: GenerationMetadata,
        *,
        owner_id: UUID,
        expected_approval: ApprovalRecord,
    ) -> StoredJob:
        async with self._transaction() as db:
            row, data = await self._load(db, job_id, owner_id=owner_id, lock=True, evidence=True)
            evidence = self._evidence(
                row, data, EvidenceRequirement.APPROVED, metadata.artifact_sha256
            )
            if evidence.approval != expected_approval:
                raise AuthoringConflict(
                    "The exact artifact must remain approved while generating questions."
                )
            job = evidence.job.model_copy(
                update={"last_generation": metadata, "updated_at": _now()}
            )
            self._write(row, data, job)
            return job

    async def read_report(self, job_id: UUID, *, owner_id: UUID) -> AuthoringResult:
        async with self._transaction() as db:
            _, data = await self._load(db, job_id, owner_id=owner_id, evidence=True)
            return self._report(data)

    async def read_evidence(
        self,
        job_id: UUID,
        *,
        owner_id: UUID,
        requirement: EvidenceRequirement = EvidenceRequirement.REUSABLE,
        expected_artifact_sha256: str | None = None,
    ) -> AuthoringEvidence:
        async with self._transaction() as db:
            row, data = await self._load(db, job_id, owner_id=owner_id, evidence=True)
            return self._evidence(row, data, requirement, expected_artifact_sha256)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
        try:
            async with self._sessions.begin() as db:
                yield db
        except (SQLAlchemyError, OSError) as exc:
            raise AuthoringStorageUnavailable("AI authoring storage is unavailable.") from exc

    @staticmethod
    async def _load(
        db: AsyncSession,
        job_id: UUID,
        *,
        owner_id: UUID | None = None,
        lock: bool = False,
        evidence: bool = False,
    ) -> tuple[Job, AuthoringJobData]:
        owner = Job.user_id.is_not(None) if owner_id is None else Job.user_id == owner_id
        conditions = (Job.id == job_id, Job.type == JobType.AI_AUTHORING, owner)
        if lock:
            # A waiting join could see the new locked jobs row with old companion data.
            # The second statement reads a fresh snapshot after acquiring the parent lock.
            locked = await db.scalar(select(Job.id).where(*conditions).with_for_update())
            if locked is None:
                raise AuthoringNotFound("Authoring job not found.")
        statement = select(Job, AuthoringJobData).join(AuthoringJobData).where(*conditions)
        if lock:
            statement = statement.execution_options(populate_existing=True)
        if not evidence:
            statement = statement.options(
                defer(AuthoringJobData.artifact_bytes), defer(AuthoringJobData.report_bytes)
            )
        result = (await db.execute(statement)).tuples().one_or_none()
        if result is None:
            raise AuthoringNotFound("Authoring job not found.")
        return result

    def _create(self, db: AsyncSession, job: StoredJob) -> tuple[Job, AuthoringJobData]:
        row = Job(
            id=job.job_id,
            type=JobType.AI_AUTHORING,
            user_id=job.owner_id,
            created_at=job.created_at,
        )
        data = AuthoringJobData(job_id=job.job_id, job=row)
        self._write(row, data, job)
        db.add(data)
        return row, data

    @staticmethod
    def _snapshot(row: Job, data: AuthoringJobData) -> StoredJob:
        if row.user_id is None:
            raise AuthoringNotFound("Authoring job not found.")
        try:
            return StoredJob.model_validate(
                {
                    **data.record,
                    "job_id": row.id,
                    "owner_id": row.user_id,
                    "state": row.status,
                    "created_at": row.created_at,
                    "finished_at": row.completed_at,
                    "artifact_sha256": data.artifact_sha256,
                }
            )
        except (ValidationError, ValueError) as exc:
            raise AuthoringConflict("Authoring job record is invalid.") from exc

    @staticmethod
    def _write(row: Job, data: AuthoringJobData, job: StoredJob) -> None:
        row.status = job.state
        row.completed_at = job.finished_at
        data.record = job.model_dump(mode="json", exclude=_ROW_FIELDS)
        data.artifact_sha256 = job.artifact_sha256

    def _start(self, row: Job, data: AuthoringJobData) -> StoredJob:
        job = self._snapshot(row, data)
        if job.state != "queued":
            raise AuthoringConflict("Authoring job is not queued.")
        now = _now()
        job = job.model_copy(update={"state": "running", "started_at": now, "updated_at": now})
        self._write(row, data, job)
        return job

    def _progress(self, row: Job, data: AuthoringJobData, progress: AuthoringProgress) -> None:
        if row.status == "running":
            self._write(
                row,
                data,
                self._snapshot(row, data).model_copy(
                    update={"progress": progress, "updated_at": _now()}
                ),
            )

    def _complete(self, row: Job, data: AuthoringJobData, report: AuthoringResult) -> StoredJob:
        job, artifact_bytes, report_bytes = completed_job(self._snapshot(row, data), report)
        data.artifact_bytes = artifact_bytes
        data.report_bytes = report_bytes
        data.report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        self._write(row, data, job)
        return job

    def _fail(self, row: Job, data: AuthoringJobData, error: PublicAuthoringError) -> StoredJob:
        job = self._snapshot(row, data)
        if job.state not in {"queued", "running"}:
            return job
        now = _now()
        job = job.model_copy(
            update={
                "state": "failed",
                "progress": None,
                "error": error,
                "updated_at": now,
                "finished_at": now,
            }
        )
        self._write(row, data, job)
        return job

    def _evidence(
        self,
        row: Job,
        data: AuthoringJobData,
        requirement: EvidenceRequirement,
        expected_artifact_sha256: str | None,
    ) -> AuthoringEvidence:
        return verify_evidence(
            self._snapshot(row, data),
            lambda: self._artifact(data),
            lambda: self._report(data),
            requirement=requirement,
            expected_artifact_sha256=expected_artifact_sha256,
        )

    @staticmethod
    def _artifact(data: AuthoringJobData) -> bytes:
        if data.artifact_bytes is None:
            raise AuthoringConflict("The validated artifact is unavailable.")
        return data.artifact_bytes

    @staticmethod
    def _report(data: AuthoringJobData) -> AuthoringResult:
        raw = data.report_bytes
        if raw is None:
            raise AuthoringConflict("Authoring review evidence is unavailable.")
        if hashlib.sha256(raw).hexdigest() != data.report_sha256:
            raise AuthoringConflict("Authoring review evidence failed its integrity check.")
        try:
            return AuthoringResult.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            raise AuthoringConflict("Authoring review evidence is invalid.") from exc


def _now() -> datetime:
    return datetime.now(UTC)
