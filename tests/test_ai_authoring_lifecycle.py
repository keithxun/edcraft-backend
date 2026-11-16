"""Ordering guarantees for progress writes made by the authoring lifecycle."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

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
