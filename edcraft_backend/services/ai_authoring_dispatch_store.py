"""PostgreSQL admission, execution claims and attempt-scoped transactions."""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from edcraft_validator.application.authoring_api import PublicAuthoringError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import defer
from sqlalchemy.sql.elements import ColumnElement

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.executors.nomad_authoring import nomad_authoring_job_id
from edcraft_backend.models.ai_authoring import AuthoringJobData
from edcraft_backend.models.job import Job, JobStatus, JobType
from edcraft_backend.services.ai_authoring_lifecycle import AuthoringAdmissionFull
from edcraft_backend.services.ai_authoring_postgres_store import PostgresAuthoringStore
from edcraft_backend.services.ai_authoring_store import AuthoringStorageUnavailable, StoredJob

_ADMISSION_LOCK = -3102632311952460229
_INTERRUPTED = PublicAuthoringError(
    code="SERVER_RESTARTED",
    message="The authoring runner disconnected before the job result was confirmed.",
)
_OWNER_REMOVED = PublicAuthoringError(
    code="AUTHORING_FAILED",
    message="The authoring job no longer has an owner.",
)


def _inline_lock(attempt_id: UUID) -> int:
    # Keep the per-attempt lock keys in a disjoint range from admission.
    return attempt_id.int >> 65


def _attempt_active() -> tuple[ColumnElement[bool], ColumnElement[bool]]:
    return AuthoringJobData.attempt_id.is_not(None), AuthoringJobData.attempt_ended_at.is_(None)


class DurableAuthoringStore(PostgresAuthoringStore):
    """Coordinate all replicas through one short admission transaction."""

    async def admit_job(
        self,
        job: StoredJob,
        *,
        configuration: AuthoringSettings,
        execution_payload: str,
    ) -> StoredJob:
        async with self._transaction() as session:
            await session.execute(select(func.pg_advisory_xact_lock(_ADMISSION_LOCK)))
            admission = (
                select(func.count(Job.id), func.count(Job.id).filter(Job.user_id == job.owner_id))
                .select_from(Job)
                .join(AuthoringJobData)
                .where(
                    or_(
                        and_(
                            Job.status == JobStatus.QUEUED,
                            Job.user_id.is_not(None),
                            AuthoringJobData.attempt_id.is_(None),
                            AuthoringJobData.execution_payload.is_not(None),
                        ),
                        and_(*_attempt_active()),
                    )
                )
            )
            total, owner = (await session.execute(admission)).one()
            if owner >= configuration.max_inflight_jobs_per_user:
                raise AuthoringAdmissionFull(
                    "You already have the maximum number of AI authoring jobs in progress."
                )
            if total >= configuration.max_active_jobs + configuration.max_queue:
                raise AuthoringAdmissionFull(
                    "AI authoring is at capacity; try again when a job finishes."
                )
            _, data = self._create(session, job)
            data.execution_mode = configuration.executor
            data.execution_payload = execution_payload
        return job

    async def _claim(
        self, session: AsyncSession, mode: str, settings: AuthoringSettings, attempt_id: UUID
    ) -> tuple[UUID, UUID] | None:
        await session.execute(select(func.pg_advisory_xact_lock(_ADMISSION_LOCK)))
        active = and_(*_attempt_active())
        active_count = (
            await session.scalar(select(func.count()).select_from(AuthoringJobData).where(active))
        ) or 0
        if active_count >= settings.max_active_jobs:
            return None
        busy_owners = (
            select(Job.user_id)
            .join(AuthoringJobData)
            .where(active, Job.user_id.is_not(None))
            .group_by(Job.user_id)
            .having(func.count() >= settings.max_active_jobs_per_user)
        )
        candidate = (
            select(Job.id)
            .join(AuthoringJobData)
            .where(
                Job.status == JobStatus.QUEUED,
                Job.user_id.is_not(None),
                Job.user_id.not_in(busy_owners),
                AuthoringJobData.execution_mode == mode,
                AuthoringJobData.execution_payload.is_not(None),
                AuthoringJobData.attempt_id.is_(None),
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
            .with_for_update(of=Job, skip_locked=True)
        )
        job_id = await session.scalar(candidate)
        if job_id is None:
            return None
        row, data = await self._load(session, job_id)
        job = self._start(row, data)
        now = datetime.now(UTC)
        data.attempt_id = attempt_id
        data.attempt_started_at = now
        data.attempt_deadline_at = now + timedelta(seconds=settings.job_timeout_seconds)
        if mode == "nomad":
            row.nomad_job_id = nomad_authoring_job_id(job.job_id, attempt_id)
        else:
            data.worker_claim_id = f"{socket.gethostname()[:200]}:{os.getpid()}"
        return job.job_id, attempt_id

    async def claim_nomad(self, settings: AuthoringSettings) -> tuple[UUID, UUID] | None:
        async with self._transaction() as session:
            return await self._claim(session, "nomad", settings, uuid4())

    @asynccontextmanager
    async def claim_inline(
        self, configuration: AuthoringSettings
    ) -> AsyncIterator[tuple[UUID, UUID] | None]:
        """Hold a session advisory lock through the non-interruptible inline run."""
        engine = self._sessions.kw.get("bind")
        if not isinstance(engine, AsyncEngine):
            raise AuthoringStorageUnavailable("Authoring execution database is unavailable.")
        attempt_id = uuid4()
        key = _inline_lock(attempt_id)
        async with engine.connect() as connection:
            await connection.execute(select(func.pg_advisory_lock(key)))
            try:
                await connection.commit()
                async with self._transaction() as session:
                    claim = await self._claim(session, "inline", configuration, attempt_id)
                yield claim
            finally:
                try:
                    await connection.execute(select(func.pg_advisory_unlock(key)))
                except BaseException:
                    # Do not give a pooled session back with an advisory lock.
                    await connection.invalidate()
                    raise

    async def active_attempts(self, mode: str) -> list[tuple[UUID, UUID]]:
        async with self._transaction() as session:
            return [
                (job_id, attempt_id)
                for job_id, attempt_id in (
                    await session.execute(
                        select(AuthoringJobData.job_id, AuthoringJobData.attempt_id).where(
                            *_attempt_active(), AuthoringJobData.execution_mode == mode
                        )
                    )
                ).all()
                if attempt_id is not None
            ]

    async def has_work(self, *, queued_mode: str | None = None) -> bool:
        queued = and_(
            Job.status == JobStatus.QUEUED,
            Job.user_id.is_not(None),
            AuthoringJobData.attempt_id.is_(None),
            AuthoringJobData.execution_payload.is_not(None),
        )
        condition = (
            and_(queued, AuthoringJobData.execution_mode == queued_mode)
            if queued_mode is not None
            else or_(queued, and_(*_attempt_active()))
        )
        async with self._transaction() as session:
            query = select(Job.id).join(AuthoringJobData).where(condition).limit(1)
            return await session.scalar(query) is not None

    @asynccontextmanager
    async def lock_attempt(
        self,
        job_id: UUID,
        attempt_id: UUID,
        *,
        skip_locked: bool = False,
        evidence: bool = False,
    ) -> AsyncIterator[tuple[Job, AuthoringJobData] | None]:
        """Serialize dispatch changes, context issuance and terminal callbacks."""
        async with self._transaction() as session:
            locked = await session.scalar(
                select(Job.id).where(Job.id == job_id).with_for_update(skip_locked=skip_locked)
            )
            if locked is None:
                yield None
                return
            query = (
                select(Job, AuthoringJobData)
                .join(AuthoringJobData)
                .where(Job.id == job_id, AuthoringJobData.attempt_id == attempt_id)
            )
            if not evidence:
                query = query.options(
                    defer(AuthoringJobData.artifact_bytes), defer(AuthoringJobData.report_bytes)
                )
            found = (await session.execute(query)).one_or_none()
            yield (found[0], found[1]) if found is not None else None

    async def reconcile_inline(self) -> None:
        """Fence a disconnected runner without assuming its thread has stopped."""
        for job_id, attempt_id in await self.active_attempts("inline"):
            async with self._transaction() as session:
                acquired = await session.scalar(
                    select(func.pg_try_advisory_xact_lock(_inline_lock(attempt_id)))
                )
                if acquired:
                    async with self.lock_attempt(job_id, attempt_id, skip_locked=True) as attempt:
                        if attempt is not None and attempt[1].attempt_ended_at is None:
                            self.fail_attempt(*attempt, _INTERRUPTED)

    async def clean_ownerless_queue(self) -> None:
        """Settle queued authoring work whose owner was deleted before dispatch."""
        async with self._transaction() as session:
            candidates = (
                select(Job, AuthoringJobData)
                .join(AuthoringJobData)
                .where(
                    Job.type == JobType.AI_AUTHORING,
                    Job.status == JobStatus.QUEUED,
                    Job.user_id.is_(None),
                    AuthoringJobData.attempt_id.is_(None),
                    AuthoringJobData.execution_payload.is_not(None),
                )
                .order_by(Job.created_at, Job.id)
                .limit(50)
                .with_for_update(of=Job, skip_locked=True)
            )
            for row, data in (await session.execute(candidates)).tuples():
                self.fail_attempt(row, data, _OWNER_REMOVED)

    def fail_attempt(self, row: Job, data: AuthoringJobData, error: PublicAuthoringError) -> None:
        if row.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            if row.user_id is None:
                now = datetime.now(UTC)
                row.status = JobStatus.FAILED
                row.completed_at = now
                data.record = {
                    **data.record,
                    "progress": None,
                    "error": error.model_dump(mode="json"),
                    "updated_at": now.isoformat(),
                }
            else:
                self._fail(row, data, error)
        data.execution_payload = None

    def release_attempt(
        self, row: Job, data: AuthoringJobData, error: PublicAuthoringError | None = None
    ) -> None:
        if error is not None:
            self.fail_attempt(row, data, error)
        data.execution_payload = None
        data.attempt_ended_at = datetime.now(UTC)
