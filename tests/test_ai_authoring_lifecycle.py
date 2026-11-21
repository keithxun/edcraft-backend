"""Ordering guarantees for progress writes made by the authoring lifecycle."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from edcraft_validator.application.authoring_contracts import AuthoringProgress

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.services.ai_authoring_lifecycle import AuthoringLifecycle
from edcraft_backend.services.ai_authoring_store import (
    AuthoringRequestRecord,
    AuthoringStore,
    StoredJob,
)


def _job(owner_id: UUID, job_id: UUID) -> StoredJob:
    now = datetime.now(UTC)
    return StoredJob(
        job_id=job_id,
        owner_id=owner_id,
        request=AuthoringRequestRecord(
            prompt="Create a tracing question",
            difficulty="easy",
            num_distractors=3,
            domain="code",
            model_profile="test-profile",
        ),
        provider="stub",
        model="stub-model",
        library_revision="test-revision",
        created_at=now,
        updated_at=now,
    )


class SlowFirstWriteStore(AuthoringStore):
    """Delay the earliest update so an unsequenced write would land last."""

    def update_progress(self, job_id: UUID, progress: AuthoringProgress) -> None:
        if progress.attempt == 0:
            time.sleep(0.05)
        super().update_progress(job_id, progress)


async def test_rapid_progress_updates_end_with_the_last_one_stored(tmp_path) -> None:
    settings = AuthoringSettings(storage_dir=tmp_path)
    store = SlowFirstWriteStore(settings)
    lifecycle = AuthoringLifecycle(store, settings)
    await lifecycle.start()
    job = _job(uuid4(), uuid4())
    store.create_job(job)
    store.mark_started(job.job_id)

    for attempt in range(5):
        lifecycle._schedule_progress(
            job.job_id, AuthoringProgress(phase="checking", attempt=attempt)
        )
    await lifecycle.wait_for_idle()

    stored = store.get_job(job.job_id)
    assert stored.progress is not None
    assert stored.progress.attempt == 4


async def test_idle_wait_handles_completed_task_before_its_callback_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = AuthoringSettings(storage_dir=tmp_path)
    lifecycle = AuthoringLifecycle(AuthoringStore(settings), settings)
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    lifecycle._tasks.add(task)
    task.add_done_callback(lifecycle._tasks.discard)

    original_gather = asyncio.gather
    calls = 0

    async def fail_if_wait_spins(
        *tasks: asyncio.Task[None], return_exceptions: bool
    ) -> list[None | BaseException]:
        nonlocal calls
        calls += 1
        assert calls == 1, "The first wait should settle the completed task."
        return await original_gather(*tasks, return_exceptions=return_exceptions)

    with monkeypatch.context() as context:
        context.setattr(asyncio, "gather", fail_if_wait_spins)
        await lifecycle.wait_for_idle()

    assert not lifecycle._tasks
