"""Bounded in-process runner for the pinned AI authoring library."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    PublicAuthoringError,
    run_authoring_job_sync,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringProgress,
    AuthoringResult,
)
from edcraft_validator.application.template_workflow import TemplateApplication

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.services.ai_authoring_judge import build_model_tool_client
from edcraft_backend.services.ai_authoring_providers import (
    JobModelPlan,
    build_managed_provider,
)
from edcraft_backend.services.ai_authoring_store import AuthoringStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[AuthoringProgress], None]
ReportResolution = Literal["accepted", "duplicate", "conflict", "missing"]


class JobExecutor(Protocol):
    """Synchronous library boundary executed inside a worker thread."""

    def __call__(
        self,
        command: AuthoringJobCommand,
        plan: JobModelPlan,
        on_progress: ProgressCallback | None,
    ) -> AuthoringJobRecord: ...


class RemoteJobExecutor(Protocol):
    """Asynchronous Nomad boundary for one authoring worker allocation."""

    async def submit(
        self,
        job_id: UUID,
        command: dict[str, object],
        callback_url: str,
        progress_url: str,
        context_url: str,
        error_url: str,
        timeout_seconds: float,
    ) -> str: ...

    async def stop(self, nomad_job_id: str) -> None: ...

    async def status(self, nomad_job_id: str) -> str: ...


class AuthoringAdmissionFull(EdCraftBaseException):
    """The bounded active-plus-waiting capacity is exhausted."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 429)


class WorkerJobRejection(Exception):
    """A worker reported a fatal error instead of an AuthoringResult."""

    def __init__(self, error: PublicAuthoringError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True, slots=True)
class WorkerChannel:
    """Per-job token-authenticated callback URLs passed to a remote worker."""

    token: str = field(repr=False)
    context_url: str
    progress_url: str
    callback_url: str
    error_url: str


@dataclass(slots=True)
class PendingJob:
    """One in-flight remote job awaiting a worker callback."""

    future: asyncio.Future[AuthoringResult]
    nomad_job_id: str | None = None


def run_library_authoring_job(
    command: AuthoringJobCommand,
    plan: JobModelPlan,
    on_progress: ProgressCallback | None,
) -> AuthoringJobRecord:
    """Run the validator with separately selected author and advisory models."""
    tool_client = build_model_tool_client(
        difficulty=plan.difficulty,
        relevance=plan.relevance,
    )

    def application_factory() -> TemplateApplication:
        return TemplateApplication(tool_client=tool_client)

    if plan.author.credentials is None:
        return run_authoring_job_sync(
            command,
            application_factory=application_factory,
            on_progress=on_progress,
        )
    credentials = plan.author.credentials
    return run_authoring_job_sync(
        command,
        application_factory=application_factory,
        provider_factory=lambda _selection: build_managed_provider(credentials),
        on_progress=on_progress,
    )


class AuthoringRunner:
    """Bound global and per-owner authoring concurrency and admission."""

    def __init__(
        self,
        store: AuthoringStore,
        settings: AuthoringSettings,
        executor: JobExecutor | None = None,
        *,
        remote_executor: RemoteJobExecutor | None = None,
        job_timeout_seconds: float | None = None,
        on_remote_finished: Callable[[UUID], None] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._executor: JobExecutor = executor or run_library_authoring_job
        self._remote_executor = remote_executor
        self._job_timeout_seconds = (
            settings.job_timeout_seconds
            if job_timeout_seconds is None
            else job_timeout_seconds
        )
        self._on_remote_finished = on_remote_finished
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._owner_semaphores: dict[UUID, asyncio.Semaphore] = {}
        self._admission_lock: asyncio.Lock | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[UUID, PendingJob] = {}
        self._admitted = 0
        self._admitted_by_owner: dict[UUID, int] = {}
        self._shutting_down = False

    async def start(self) -> None:
        """Bind loop-owned primitives on the serving event loop."""
        self._shutting_down = False
        self._ensure_loop_state()

    async def reserve(self, owner_id: UUID) -> None:
        """Reserve one admitted slot before a job record is created."""
        self._ensure_loop_state()
        if self._admission_lock is None:
            raise RuntimeError("Authoring runner admission lock is unavailable.")
        async with self._admission_lock:
            if self._shutting_down:
                raise AuthoringAdmissionFull("The authoring service is shutting down.")
            owner_inflight = self._admitted_by_owner.get(owner_id, 0)
            if owner_inflight >= self._settings.max_inflight_jobs_per_user:
                raise AuthoringAdmissionFull(
                    "You already have the maximum number of AI authoring jobs in progress."
                )
            capacity = self._settings.max_active_jobs + self._settings.max_queue
            if self._admitted >= capacity:
                raise AuthoringAdmissionFull(
                    "AI authoring is at capacity; try again when a job finishes."
                )
            self._admitted += 1
            self._admitted_by_owner[owner_id] = owner_inflight + 1

    async def release(self, owner_id: UUID) -> None:
        """Release a reservation when a job finishes or scheduling fails."""
        self._ensure_loop_state()
        if self._admission_lock is None:
            raise RuntimeError("Authoring runner admission lock is unavailable.")
        async with self._admission_lock:
            self._admitted = max(0, self._admitted - 1)
            owner_remaining = self._admitted_by_owner.get(owner_id, 0) - 1
            if owner_remaining > 0:
                self._admitted_by_owner[owner_id] = owner_remaining
            else:
                self._admitted_by_owner.pop(owner_id, None)

    def start_job(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        plan: JobModelPlan,
        owner_id: UUID,
        *,
        worker_channel: WorkerChannel | None = None,
    ) -> None:
        """Schedule an admitted job without waiting for model work."""
        self._ensure_loop_state()
        task = asyncio.create_task(
            self._run(job_id, command, plan, owner_id, worker_channel),
            name=f"ai-authoring-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def resolve(self, job_id: UUID, report: AuthoringResult) -> ReportResolution:
        """Resolve a pending remote future from a token-authenticated callback."""
        pending = self._pending.get(job_id)
        if pending is None:
            return "missing"
        future = pending.future
        if future.done():
            if future.cancelled():
                return "missing"
            exception = future.exception()
            if exception is None:
                existing = future.result()
                if existing.model_dump(mode="json") == report.model_dump(mode="json"):
                    return "duplicate"
                return "conflict"
            return "conflict"
        future.set_result(report)
        return "accepted"

    def reject(self, job_id: UUID, error: PublicAuthoringError) -> ReportResolution:
        """Reject a pending remote future with a worker-reported fatal error."""
        pending = self._pending.get(job_id)
        if pending is None:
            return "missing"
        future = pending.future
        if future.done():
            return "conflict"
        future.set_exception(WorkerJobRejection(error))
        return "accepted"

    async def wait_for_idle(self) -> None:
        """Test/shutdown helper: wait for scheduled inline and remote jobs."""
        while True:
            tasks = set(self._tasks) | set(self._background_tasks)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self, timeout: float) -> None:
        """Stop admitting work, cancel remote waits and wait up to ``timeout``."""
        self._shutting_down = True
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.cancel()
        tasks = set(self._tasks) | set(self._background_tasks)
        if not tasks:
            return
        _, pending_tasks = await asyncio.wait(tasks, timeout=timeout)
        if pending_tasks:
            logger.warning(
                "AI authoring shutdown timed out with %s task(s) still running",
                len(pending_tasks),
            )

    async def _run(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        plan: JobModelPlan,
        owner_id: UUID,
        worker_channel: WorkerChannel | None,
    ) -> None:
        owner_semaphore = self._owner_semaphore(owner_id)
        if self._semaphore is None or owner_semaphore is None:
            await asyncio.to_thread(
                self._store.fail_job,
                job_id,
                PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message="Authoring stopped before it could start.",
                ),
            )
            await self.release(owner_id)
            return
        try:
            async with owner_semaphore:
                async with self._semaphore:
                    await asyncio.to_thread(self._store.mark_started, job_id)
                    if self._remote_executor is not None:
                        await self._run_remote(
                            job_id, command, worker_channel, owner_id
                        )
                    else:
                        await self._run_inline(job_id, command, plan)
        except Exception:
            logger.exception("AI authoring runner failed", extra={"job_id": str(job_id)})
            await asyncio.to_thread(
                self._store.fail_job,
                job_id,
                PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message="Authoring stopped before it could start.",
                ),
            )
        finally:
            await self.release(owner_id)

    async def _run_inline(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        plan: JobModelPlan,
    ) -> None:
        try:
            record = await asyncio.to_thread(
                self._executor,
                command,
                plan,
                self._progress_callback(job_id),
            )
            await asyncio.to_thread(self._store.complete_job, job_id, record.report)
        except Exception:
            logger.exception("AI authoring job failed", extra={"job_id": str(job_id)})
            await asyncio.to_thread(
                self._store.fail_job,
                job_id,
                PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message=("Authoring stopped before a usable artifact was saved."),
                ),
            )

    async def _run_remote(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        worker_channel: WorkerChannel | None,
        owner_id: UUID,
    ) -> None:
        """Submit one Nomad job and wait for its callback or timeout."""
        if worker_channel is None or self._remote_executor is None:
            logger.error(
                "Remote authoring job has no worker channel",
                extra={"job_id": str(job_id)},
            )
            await asyncio.to_thread(
                self._store.fail_job,
                job_id,
                PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message="Authoring could not be scheduled.",
                ),
            )
            return
        pending = PendingJob(future=asyncio.get_running_loop().create_future())
        self._pending[job_id] = pending
        try:
            try:
                pending.nomad_job_id = await self._remote_executor.submit(
                    job_id,
                    command.model_dump(mode="json"),
                    worker_channel.callback_url,
                    worker_channel.progress_url,
                    worker_channel.context_url,
                    worker_channel.error_url,
                    self._job_timeout_seconds,
                )
            except Exception:
                logger.exception(
                    "Could not submit Nomad authoring job",
                    extra={"job_id": str(job_id)},
                )
                await asyncio.to_thread(
                    self._store.fail_job,
                    job_id,
                    PublicAuthoringError(
                        code="AUTHORING_FAILED",
                        message="Authoring could not be scheduled.",
                    ),
                )
                return
            try:
                report = await asyncio.wait_for(
                    pending.future, timeout=self._job_timeout_seconds
                )
            except TimeoutError:
                logger.warning(
                    "Nomad authoring job timed out", extra={"job_id": str(job_id)}
                )
                await self._stop_remote(pending.nomad_job_id)
                await asyncio.to_thread(
                    self._store.fail_job,
                    job_id,
                    PublicAuthoringError(
                        code="AUTHORING_TIMEOUT",
                        message="Authoring timed out before the worker reported a result.",
                    ),
                )
                return
            except WorkerJobRejection as exc:
                await asyncio.to_thread(self._store.fail_job, job_id, exc.error)
                return
            except asyncio.CancelledError:
                await self._stop_remote(pending.nomad_job_id)
                raise
            await asyncio.to_thread(self._store.complete_job, job_id, report)
        finally:
            self._pending.pop(job_id, None)
            if self._on_remote_finished is not None:
                self._on_remote_finished(job_id)

    async def _stop_remote(self, nomad_job_id: str | None) -> None:
        if nomad_job_id is None or self._remote_executor is None:
            return
        try:
            await self._remote_executor.stop(nomad_job_id)
        except Exception:
            logger.warning(
                "Could not stop Nomad authoring job",
                extra={"nomad_job_id": nomad_job_id},
                exc_info=True,
            )

    def _owner_semaphore(self, owner_id: UUID) -> asyncio.Semaphore:
        """Return the loop-bound active-job semaphore for one owner."""
        semaphore = self._owner_semaphores.get(owner_id)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._settings.max_active_jobs_per_user)
            self._owner_semaphores[owner_id] = semaphore
        return semaphore

    def _progress_callback(self, job_id: UUID) -> ProgressCallback:
        loop = asyncio.get_running_loop()

        def publish(progress: AuthoringProgress) -> None:
            loop.call_soon_threadsafe(self._schedule_progress, job_id, progress)

        return publish

    def _schedule_progress(self, job_id: UUID, progress: AuthoringProgress) -> None:
        task = asyncio.create_task(
            asyncio.to_thread(self._store.update_progress, job_id, progress)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _ensure_loop_state(self) -> None:
        loop = asyncio.get_running_loop()
        if loop is self._loop:
            return
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._settings.max_active_jobs)
        self._owner_semaphores = {}
        self._admission_lock = asyncio.Lock()
        self._tasks = set()
        self._background_tasks = set()
        self._pending = {}
        self._admitted_by_owner = {}


__all__ = [
    "AuthoringAdmissionFull",
    "AuthoringRunner",
    "JobExecutor",
    "PendingJob",
    "RemoteJobExecutor",
    "WorkerChannel",
    "WorkerJobRejection",
    "run_library_authoring_job",
]
