"""Durable background dispatch and worker callbacks for AI authoring."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from edcraft_validator.application.authoring_api import AuthoringJobCommand, PublicAuthoringError
from edcraft_validator.application.authoring_contracts import AuthoringProgress, AuthoringResult
from edcraft_validator.domains.domain_modes import load_artifact
from edcraft_validator.domains.domain_registry import create_domain
from pydantic import ValidationError

from edcraft_backend.config import settings as app_settings
from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.executors.nomad_authoring import (
    NomadAuthoringExecutor,
    NomadAuthoringStatus,
    NomadUnavailable,
)
from edcraft_backend.models.ai_authoring import AuthoringJobData
from edcraft_backend.models.job import Job, JobStatus
from edcraft_backend.services.ai_authoring_dispatch_store import DurableAuthoringStore
from edcraft_backend.services.ai_authoring_lifecycle import AuthoringAdmissionFull
from edcraft_backend.services.ai_authoring_runtime import (
    AuthoringValidationError,
    ExecutionPayload,
    ExecutionSecrets,
    JobExecutor,
    JobRuntime,
    ProgressCallback,
)
from edcraft_backend.services.ai_authoring_store import (
    AuthoringConflict,
    AuthoringStorageUnavailable,
    AuthoringStoreError,
    StoredJob,
)
from edcraft_backend.services.ai_domains import LIBRARY_REVISION
from edcraft_backend.services.provider_host_guard import allow_private_provider_hosts
from edcraft_backend.services.provider_secrets import ProviderSecretError

logger = logging.getLogger(__name__)

_RESULT_RETRY_WINDOW = timedelta(minutes=5)
# A disconnected allocation can run until the worker's hard process timer fires.
_WORKER_EXIT_GRACE = timedelta(seconds=30)
_WORKER_FAILED = PublicAuthoringError(
    code="AUTHORING_FAILED",
    message="Authoring stopped before a usable result was saved.",
)
_JOB_TIMED_OUT = PublicAuthoringError(
    code="AUTHORING_TIMEOUT",
    message="Authoring timed out before the job produced a result.",
)


class AuthoringWorkerUnauthorized(EdCraftBaseException):
    def __init__(self) -> None:
        super().__init__("Authoring worker credential is invalid or expired.", 401)


class DurableAuthoringLifecycle:
    """Process the shared queue without owning the lifetime of a Nomad worker."""

    def __init__(
        self,
        store: DurableAuthoringStore,
        settings: AuthoringSettings,
        executor: JobExecutor,
        *,
        nomad: NomadAuthoringExecutor | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._executor = executor
        self._secrets = ExecutionSecrets(settings, app_settings.jwt.secret)
        self._image = (settings.nomad_container_image or "").strip()
        self._nomad = nomad or NomadAuthoringExecutor(
            app_settings.nomad,
            image=self._image,
            callback_base_url=settings.effective_nomad_callback_base_url(
                app_settings.nomad.callback_base_url
            ),
            force_pull=settings.nomad_force_pull,
            allow_private_provider_hosts=(
                not app_settings.is_production and allow_private_provider_hosts()
            ),
        )
        self._controller: asyncio.Task[None] | None = None
        self._inline_runs: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()
        self._shutting_down = False

    async def start(self) -> None:
        if self._controller is not None and not self._controller.done():
            return
        await self._store.ensure_ready()
        self._shutting_down = False
        self._wake = asyncio.Event()
        self._controller = asyncio.create_task(self._run_controller(), name="ai-authoring-dispatch")

    async def submit(
        self, job: StoredJob, command: AuthoringJobCommand, runtime: JobRuntime
    ) -> StoredJob:
        if self._shutting_down:
            raise AuthoringAdmissionFull("The authoring service is shutting down.")
        remote = self._settings.executor == "nomad"
        if remote and not self._image:
            raise AuthoringValidationError("The authoring worker image is not configured.")
        payload = ExecutionPayload.capture(command, runtime, remote=remote)
        admitted = await self._store.admit_job(
            job, configuration=self._settings, execution_payload=self._secrets.encrypt(payload)
        )
        self._wake.set()
        return admitted

    async def shutdown(self, timeout: float) -> None:
        self._shutting_down = True
        if self._controller is not None:
            self._controller.cancel()
            with suppress(asyncio.CancelledError):
                await self._controller
            self._controller = None
        if self._inline_runs:
            await asyncio.wait(self._inline_runs, timeout=timeout)

    async def wait_for_idle(self) -> None:
        """Wait for outstanding attempts and queued inputs in focused tests."""
        while await self._store.has_work():
            self._wake.set()
            await asyncio.sleep(0.02)

    async def _run_controller(self) -> None:
        while True:
            self._wake.clear()
            try:
                await self.dispatch_once()
            except (AuthoringStorageUnavailable, OSError):
                logger.warning("AI authoring dispatcher could not reach its database")
            except Exception as exc:
                logger.error("AI authoring dispatcher iteration failed (%s)", type(exc).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self._settings.dispatch_interval_seconds)

    async def dispatch_once(self) -> None:
        """Reconcile existing attempts before claiming additional capacity."""
        await self._store.clean_ownerless_queue()
        await self._store.reconcile_inline()
        active = await self._store.active_attempts("nomad")
        await asyncio.gather(*(self._reconcile_nomad(*attempt) for attempt in active))
        if self._image:
            claimed = []
            for _ in range(self._settings.max_active_jobs):
                attempt = await self._store.claim_nomad(self._settings)
                if attempt is None:
                    break
                claimed.append(attempt)
            await asyncio.gather(*(self._reconcile_nomad(*attempt) for attempt in claimed))
        if await self._store.has_work(queued_mode="inline"):
            for _ in range(self._settings.max_active_jobs - len(self._inline_runs)):
                task = asyncio.create_task(self._run_inline(), name="ai-authoring-inline")
                self._inline_runs.add(task)
                task.add_done_callback(self._inline_runs.discard)

    async def _reconcile_nomad(self, job_id: UUID, attempt_id: UUID) -> None:
        async with self._store.lock_attempt(job_id, attempt_id, skip_locked=True) as attempt:
            if attempt is None:
                return
            row, data = attempt
            deadline = data.attempt_deadline_at
            job_name = row.nomad_job_id
            if data.attempt_ended_at is not None or deadline is None or job_name is None:
                return
            stop = _now() >= deadline or row.user_id is None
            if stop:
                error = _JOB_TIMED_OUT if row.user_id is not None else _WORKER_FAILED
                self._store.fail_attempt(row, data, error)
            known_registration = data.nomad_registered_at

        # Routine status reads run outside the write transaction so a slow
        # scheduler cannot block context issuance or a completion callback.
        try:
            if stop:
                with suppress(NomadUnavailable):
                    await self._nomad.stop(job_name)
            status = await self._nomad.status(job_name)
            async with self._store.lock_attempt(job_id, attempt_id, skip_locked=True) as attempt:
                if attempt is not None and attempt[1].attempt_ended_at is None:
                    await self._apply_nomad_state(
                        *attempt, attempt_id, status, known_registration, deadline
                    )
        except NomadUnavailable:
            logger.warning("Nomad state could not be confirmed for authoring job %s", job_id)

    async def _apply_nomad_state(
        self,
        row: Job,
        data: AuthoringJobData,
        attempt_id: UUID,
        status: NomadAuthoringStatus,
        known_registration: datetime | None,
        deadline: datetime,
    ) -> None:
        now = _now()
        if status == "active":
            data.nomad_registered_at = data.nomad_registered_at or now
            return
        if status == "missing" and data.nomad_registered_at != known_registration:
            # Another dispatcher observed or submitted this attempt during the query.
            return
        might_be_running = status == "unreachable" or (
            status == "missing" and data.worker_context_at is not None
        )
        if might_be_running and now < deadline + _WORKER_EXIT_GRACE:
            data.nomad_registered_at = data.nomad_registered_at or now
            return
        if (
            status != "missing"
            or data.nomad_registered_at is not None
            or row.status != JobStatus.RUNNING
        ):
            self._store.release_attempt(row, data, _WORKER_FAILED)
        elif self._image and now < deadline:
            job = self._store._snapshot(row, data)
            if job.library_revision != LIBRARY_REVISION:
                self._store.release_attempt(row, data, _WORKER_FAILED)
                return
            # Serialize POST with cancellation. A lost reply is reconciled
            # against the same create-only external identity on the next pass.
            await self._nomad.register(
                row.id,
                attempt_id,
                job.library_revision,
                self._secrets.worker_token(row.id, attempt_id),
            )
            data.nomad_registered_at = now

    async def _run_inline(self) -> None:
        try:
            async with self._store.claim_inline(self._settings) as attempt:
                if attempt is not None:
                    await self._execute_inline(*attempt)
        except Exception as exc:
            logger.error("Inline authoring attempt could not be settled (%s)", type(exc).__name__)

    async def _execute_inline(self, job_id: UUID, attempt_id: UUID) -> None:
        async with self._store.lock_attempt(job_id, attempt_id) as attempt:
            if attempt is None:
                return
            row, data = attempt
            deadline = data.attempt_deadline_at
            try:
                payload = self._secrets.decrypt(data.execution_payload)
            except ProviderSecretError:
                self._store.release_attempt(row, data, _WORKER_FAILED)
                return
            if (
                deadline is None
                or self._store._snapshot(row, data).library_revision != LIBRARY_REVISION
            ):
                self._store.release_attempt(row, data, _WORKER_FAILED)
                return
        runtime = payload.runtime(self._executor)
        progress = self._progress_callback(job_id, attempt_id)
        execution = asyncio.create_task(asyncio.to_thread(runtime.run, payload.command, progress))
        try:
            completed = await asyncio.wait_for(
                asyncio.shield(execution), timeout=max(0.0, (deadline - _now()).total_seconds())
            )
        except TimeoutError:
            try:
                async with self._store.lock_attempt(job_id, attempt_id) as attempt:
                    if attempt is not None:
                        self._store.fail_attempt(*attempt, _JOB_TIMED_OUT)
            finally:
                # A running Python thread still occupies this attempt's slot.
                with suppress(Exception):
                    await asyncio.shield(execution)
            await self._finish_inline(job_id, attempt_id, error=_JOB_TIMED_OUT)
        except Exception:
            await self._finish_inline(job_id, attempt_id, error=_WORKER_FAILED)
        else:
            await self._finish_inline(job_id, attempt_id, report=completed.report)

    def _progress_callback(self, job_id: UUID, attempt_id: UUID) -> ProgressCallback:
        loop = asyncio.get_running_loop()

        def progress(update: AuthoringProgress) -> None:
            if loop.is_running():
                write = asyncio.run_coroutine_threadsafe(
                    self._inline_progress(job_id, attempt_id, update), loop
                )
                try:
                    write.result(timeout=5)
                except TimeoutError:
                    write.cancel()

        return progress

    async def _inline_progress(
        self, job_id: UUID, attempt_id: UUID, progress: AuthoringProgress
    ) -> None:
        try:
            async with self._store.lock_attempt(job_id, attempt_id) as attempt:
                if attempt is not None and attempt[0].user_id is not None:
                    self._store._progress(*attempt, progress)
        except AuthoringStoreError:
            logger.warning("Could not persist inline authoring progress")

    async def _finish_inline(
        self,
        job_id: UUID,
        attempt_id: UUID,
        *,
        report: AuthoringResult | None = None,
        error: PublicAuthoringError | None = None,
    ) -> None:
        async with self._store.lock_attempt(job_id, attempt_id) as attempt:
            if attempt is not None and attempt[1].attempt_ended_at is None:
                row, data = attempt
                if (
                    report is not None
                    and row.status == JobStatus.RUNNING
                    and row.user_id is not None
                ):
                    self._store._complete(row, data, report)
                self._store.release_attempt(row, data, error)
        self._wake.set()

    @asynccontextmanager
    async def _worker_attempt(
        self,
        job_id: UUID,
        attempt_id: UUID,
        token: str,
        allocation_id: str,
        *,
        context: bool = False,
        evidence: bool = False,
    ) -> AsyncIterator[tuple[Job, AuthoringJobData, datetime]]:
        if not self._secrets.valid_worker_token(job_id, attempt_id, token):
            raise AuthoringWorkerUnauthorized()
        async with self._store.lock_attempt(job_id, attempt_id, evidence=evidence) as attempt:
            if attempt is None:
                raise AuthoringWorkerUnauthorized()
            row, data = attempt
            deadline = data.attempt_deadline_at
            first_context = context and data.worker_claim_id is None
            if (
                row.user_id is None
                or data.execution_mode != "nomad"
                or deadline is None
                or _now() >= deadline + (_RESULT_RETRY_WINDOW if evidence else timedelta())
                or (data.worker_claim_id != allocation_id and not first_context)
            ):
                raise AuthoringWorkerUnauthorized()
            yield row, data, deadline

    async def worker_context(
        self, job_id: UUID, attempt_id: UUID, token: str, allocation_id: str
    ) -> dict[str, object]:
        async with self._worker_attempt(job_id, attempt_id, token, allocation_id, context=True) as (
            row,
            data,
            deadline,
        ):
            if row.status != JobStatus.RUNNING or data.attempt_ended_at is not None:
                raise AuthoringConflict("Authoring attempt is no longer running.")
            job = self._store._snapshot(row, data)
            if job.library_revision != LIBRARY_REVISION:
                raise AuthoringValidationError("Worker library revision does not match.")
            try:
                payload = self._secrets.decrypt(data.execution_payload)
            except ProviderSecretError:
                raise AuthoringStorageUnavailable(
                    "Authoring execution inputs are unavailable."
                ) from None
            now = _now()
            data.worker_claim_id = allocation_id
            data.worker_context_at = data.worker_context_at or now
            return payload.worker_context(job.library_revision, (deadline - now).total_seconds())

    async def worker_progress(
        self,
        job_id: UUID,
        attempt_id: UUID,
        token: str,
        allocation_id: str,
        progress: AuthoringProgress,
    ) -> None:
        async with self._worker_attempt(job_id, attempt_id, token, allocation_id) as (row, data, _):
            if row.status == JobStatus.RUNNING and not _older_progress(
                progress, self._store._snapshot(row, data).progress
            ):
                self._store._progress(row, data, progress)

    async def worker_result(
        self,
        job_id: UUID,
        attempt_id: UUID,
        token: str,
        allocation_id: str,
        library_revision: str,
        wire_report: dict[str, Any],
    ) -> None:
        async with self._worker_attempt(
            job_id, attempt_id, token, allocation_id, evidence=True
        ) as (row, data, deadline):
            job = self._store._snapshot(row, data)
            if library_revision != LIBRARY_REVISION or library_revision != job.library_revision:
                raise AuthoringValidationError("Worker library revision does not match.")
            if data.report_bytes is None and (
                _now() >= deadline or data.attempt_ended_at is not None
            ):
                raise AuthoringWorkerUnauthorized()
            report = _validated_worker_report(job, wire_report)
            if data.report_bytes is not None:
                stored = self._store._report(data)
                if stored.model_dump(mode="json") == report.model_dump(mode="json"):
                    return
                raise AuthoringConflict("Authoring job already has a different result.")
            try:
                command = self._secrets.decrypt(data.execution_payload).command
            except ProviderSecretError:
                raise AuthoringStorageUnavailable(
                    "Authoring execution inputs are unavailable."
                ) from None
            if report.request != command.request:
                raise AuthoringValidationError(
                    "Worker report did not match this authoring request."
                )
            self._store._complete(row, data, report)
            data.execution_payload = None
        self._wake.set()

    async def worker_error(
        self, job_id: UUID, attempt_id: UUID, token: str, allocation_id: str
    ) -> None:
        async with self._worker_attempt(job_id, attempt_id, token, allocation_id) as (row, data, _):
            self._store.fail_attempt(row, data, _WORKER_FAILED)
        self._wake.set()


def _validated_worker_report(job: StoredJob, wire_report: dict[str, Any]) -> AuthoringResult:
    try:
        report = AuthoringResult.model_validate_json(json.dumps(wire_report))
        if (report.domain, report.provider, report.model) != (
            job.request.domain,
            job.provider,
            job.model,
        ):
            raise ValueError("Report identity did not match")
        if report.artifact is not None:
            artifact = load_artifact(
                create_domain(report.domain), json.dumps(wire_report["artifact"])
            )
            report = report.model_copy(update={"artifact": artifact})
        return report
    except (ValidationError, ValueError, KeyError, TypeError):
        raise AuthoringValidationError("Worker report did not match this authoring job.") from None


def _older_progress(update: AuthoringProgress, current: AuthoringProgress | None) -> bool:
    """A timed-out HTTP progress write can arrive after its successor."""
    if current is None:
        return False
    if update.attempt != current.attempt:
        return int(update.attempt) < int(current.attempt)
    new_time, old_time = update.phase_started_at, current.phase_started_at
    if new_time is not None and old_time is not None:
        return (
            new_time.replace(tzinfo=UTC) if new_time.tzinfo is None else new_time,
            update.checks_done or 0,
        ) < (
            old_time.replace(tzinfo=UTC) if old_time.tzinfo is None else old_time,
            current.checks_done or 0,
        )
    return False


def _now() -> datetime:
    return datetime.now(UTC)
