"""One owner for the authoring job lifecycle.

The lifecycle reserves admission, persists the queued record, schedules the
worker and releases the reservation exactly once. The store stays plain
persistence and the runtime owns synchronous execution, so a cancellation or
scheduling failure is settled in one place.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from uuid import UUID

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    PublicAuthoringError,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.services.ai_authoring_runtime import (
    JobRuntime,
    ProgressCallback,
)
from edcraft_backend.services.ai_authoring_store import AuthoringStore, StoredJob

logger = logging.getLogger(__name__)

_START_FAILED = PublicAuthoringError(
    code="AUTHORING_FAILED",
    message="Authoring stopped before it could start.",
)
_EXECUTION_FAILED = PublicAuthoringError(
    code="AUTHORING_FAILED",
    message="Authoring stopped before a usable artifact was saved.",
)
_EXECUTION_TIMEOUT = PublicAuthoringError(
    code="AUTHORING_TIMEOUT",
    message="Authoring timed out before the job produced a result.",
)
_SCHEDULE_FAILED = PublicAuthoringError(
    code="AUTHORING_FAILED",
    message="Authoring could not be scheduled.",
)


def _consume_execution_result(task: asyncio.Task[AuthoringJobRecord]) -> None:
    """Retrieve a late executor outcome so a timed-out task cannot warn."""
    if not task.cancelled():
        task.exception()


class AuthoringAdmissionFull(EdCraftBaseException):
    """The bounded active-plus-waiting capacity is exhausted."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 429)


class AdmissionLease:
    """One admitted slot that can be released at most once."""

    def __init__(
        self,
        owner_id: UUID,
        release: Callable[[UUID], Awaitable[None]],
    ) -> None:
        self.owner_id = owner_id
        self._release = release
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._release(self.owner_id)


class AuthoringLifecycle:
    """Reserve, persist, schedule, finish and release one admitted job."""

    def __init__(self, store: AuthoringStore, settings: AuthoringSettings) -> None:
        self._store = store
        self._settings = settings
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._owner_semaphores: dict[UUID, asyncio.Semaphore] = {}
        self._admission_lock: asyncio.Lock | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._progress_chains: dict[UUID, asyncio.Task[None]] = {}
        self._admitted = 0
        self._admitted_by_owner: dict[UUID, int] = {}
        self._shutting_down = False

    async def start(self) -> None:
        """Prepare durable storage, reconcile interruptions and bind the loop."""
        await asyncio.to_thread(self._store.ensure_ready)
        await asyncio.to_thread(self._store.reconcile_startup)
        self._shutting_down = False
        self._ensure_loop_state()

    async def submit(
        self,
        job: StoredJob,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
    ) -> StoredJob:
        """Admit, persist and schedule one job without waiting for model work."""
        lease = await self._reserve(job.owner_id)
        write_task = asyncio.create_task(asyncio.to_thread(self._store.create_job, job))
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            # The write cannot be interrupted, so keep the job and give it a
            # worker before propagating cancellation. Settlement runs detached:
            # a second cancellation of this request must not abort scheduling
            # or leak the admission slot.
            self._schedule_background(
                self._settle_after_cancellation(write_task, job.job_id, command, runtime, lease),
                name=f"ai-authoring-settlement-{job.job_id}",
            )
            raise
        except Exception:
            await lease.release()
            raise
        try:
            self._start_job(job.job_id, command, runtime, lease)
        except Exception:
            await self._fail_scheduled_job(job.job_id, lease)
            raise
        return job

    async def wait_for_idle(self) -> None:
        """Test/shutdown helper: wait for currently scheduled job tasks."""
        while True:
            tasks = set(self._tasks) | set(self._background_tasks)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)
            # A gather of finished tasks can return before their callbacks run.
            self._tasks.difference_update(tasks)
            self._background_tasks.difference_update(tasks)

    async def shutdown(self, timeout: float) -> None:
        """Stop admitting work and wait up to ``timeout`` for existing tasks."""
        self._shutting_down = True
        tasks = set(self._tasks) | set(self._background_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning(
                "AI authoring shutdown timed out with %s task(s) still running",
                len(pending),
            )

    async def _settle_after_cancellation(
        self,
        write_task: asyncio.Task[StoredJob],
        job_id: UUID,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
        lease: AdmissionLease,
    ) -> None:
        """Let the write finish, then schedule the job so it is not orphaned."""
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            # Detached settlement can only be cancelled during loop shutdown;
            # release rather than leave capacity reserved.
            await lease.release()
            raise
        except Exception:
            await lease.release()
            return
        try:
            self._start_job(job_id, command, runtime, lease)
        except Exception:
            await self._fail_scheduled_job(job_id, lease)

    async def _fail_scheduled_job(self, job_id: UUID, lease: AdmissionLease) -> None:
        """Fail a persisted job that could not be scheduled and release its slot."""
        try:
            await asyncio.to_thread(self._store.fail_job, job_id, _SCHEDULE_FAILED)
        finally:
            await lease.release()

    async def _reserve(self, owner_id: UUID) -> AdmissionLease:
        """Reserve one admitted slot before a job record is created."""
        self._ensure_loop_state()
        if self._admission_lock is None:
            raise RuntimeError("Authoring admission lock is unavailable.")
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
        return AdmissionLease(owner_id, self._release_counters)

    async def _release_counters(self, owner_id: UUID) -> None:
        self._ensure_loop_state()
        if self._admission_lock is None:
            raise RuntimeError("Authoring admission lock is unavailable.")
        async with self._admission_lock:
            self._admitted = max(0, self._admitted - 1)
            owner_remaining = self._admitted_by_owner.get(owner_id, 0) - 1
            if owner_remaining > 0:
                self._admitted_by_owner[owner_id] = owner_remaining
            else:
                self._admitted_by_owner.pop(owner_id, None)

    def _start_job(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
        lease: AdmissionLease,
    ) -> None:
        """Schedule an admitted job without waiting for model work."""
        self._ensure_loop_state()
        task = asyncio.create_task(
            self._run(job_id, command, runtime, lease),
            name=f"ai-authoring-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _schedule_background(
        self,
        coro: Coroutine[object, object, None],
        *,
        name: str,
    ) -> None:
        """Track a detached coroutine so idle/shutdown waits for it."""
        self._ensure_loop_state()
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
        lease: AdmissionLease,
    ) -> None:
        owner_semaphore = self._owner_semaphore(lease.owner_id)
        try:
            if self._semaphore is None or owner_semaphore is None:
                await asyncio.to_thread(self._store.fail_job, job_id, _START_FAILED)
                return
            async with owner_semaphore:
                async with self._semaphore:
                    await asyncio.to_thread(self._store.mark_started, job_id)
                    try:
                        record = await self._run_executor(job_id, command, runtime)
                        if record is not None:
                            await asyncio.to_thread(self._store.complete_job, job_id, record.report)
                    except Exception:
                        logger.exception("AI authoring job failed", extra={"job_id": str(job_id)})
                        await asyncio.to_thread(self._store.fail_job, job_id, _EXECUTION_FAILED)
        except Exception:
            logger.exception("AI authoring runner failed", extra={"job_id": str(job_id)})
            await asyncio.to_thread(self._store.fail_job, job_id, _START_FAILED)
        finally:
            await lease.release()

    async def _run_executor(
        self,
        job_id: UUID,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
    ) -> AuthoringJobRecord | None:
        """Run the synchronous executor under the per-job deadline.

        A Python thread cannot be interrupted. When the deadline fires the job
        is failed immediately, but its admission slot is held until the thread
        actually returns, so a late worker cannot oversubscribe the process.
        """
        execution = asyncio.create_task(
            asyncio.to_thread(
                runtime.run,
                command,
                self._progress_callback(job_id),
            )
        )
        execution.add_done_callback(_consume_execution_result)
        try:
            return await asyncio.wait_for(
                asyncio.shield(execution),
                timeout=self._settings.job_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "AI authoring job timed out",
                extra={
                    "job_id": str(job_id),
                    "timeout_seconds": self._settings.job_timeout_seconds,
                },
            )
            await asyncio.to_thread(self._store.fail_job, job_id, _EXECUTION_TIMEOUT)
            try:
                await asyncio.shield(execution)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Timed-out authoring worker also failed",
                    exc_info=True,
                    extra={"job_id": str(job_id)},
                )
            return None

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
        """Queue one progress write behind the previous write for that job.

        Updates arrive from the executor thread in the order they were made, yet
        each used to be written by its own thread, so a slow earlier write could
        land after a later one. Chaining keeps the stored progress equal to the
        newest update received rather than the one that finished last.
        """
        previous = self._progress_chains.get(job_id)
        task = asyncio.create_task(
            self._write_progress(job_id, progress, previous),
            name=f"ai-authoring-progress-{job_id}",
        )
        self._progress_chains[job_id] = task
        self._background_tasks.add(task)
        task.add_done_callback(lambda finished: self._settle_progress_write(job_id, finished))

    async def _write_progress(
        self,
        job_id: UUID,
        progress: AuthoringProgress,
        previous: asyncio.Task[None] | None,
    ) -> None:
        """Persist one progress update once the one before it has finished."""
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        await asyncio.to_thread(self._store.update_progress, job_id, progress)

    def _settle_progress_write(self, job_id: UUID, task: asyncio.Task[None]) -> None:
        """Forget a finished write so idle waits and the chain stay accurate."""
        self._background_tasks.discard(task)
        if self._progress_chains.get(job_id) is task:
            del self._progress_chains[job_id]

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
        self._progress_chains = {}
        self._admitted = 0
        self._admitted_by_owner = {}


__all__ = [
    "AdmissionLease",
    "AuthoringAdmissionFull",
    "AuthoringLifecycle",
]
