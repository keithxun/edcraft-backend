"""Exercise authoring durability across independent PostgreSQL sessions."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from edcraft_backend.config.settings import AuthoringProfileSettings, AuthoringSettings
from edcraft_backend.dependencies import (
    get_ai_authoring_service,
    get_current_user,
    get_provider_connection_service,
)
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.main import edcraft_exception_handler
from edcraft_backend.models.ai_authoring import AuthoringJobData
from edcraft_backend.models.job import Job, JobType
from edcraft_backend.models.user import User
from edcraft_backend.routers.ai_authoring import router as authoring_router
from edcraft_backend.schemas.ai_authoring import GenerationMetadata
from edcraft_backend.services.ai_authoring_dispatch_store import DurableAuthoringStore
from edcraft_backend.services.ai_authoring_postgres_store import PostgresAuthoringStore
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.ai_authoring_store import (
    ApprovalRecord,
    AuthoringConflict,
    AuthoringNotFound,
    AuthoringRequestRecord,
    StoredJob,
)
from tests.ai_authoring_helpers import StaticExecutor, make_checked_report, make_command

pytestmark = pytest.mark.integration


@pytest.fixture
async def authoring_db(
    test_engine: AsyncEngine,
    test_db_setup: None,
) -> AsyncIterator[tuple[async_sessionmaker[AsyncSession], UUID, UUID]]:
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    owner_id, other_id = uuid4(), uuid4()
    async with sessions.begin() as db:
        db.add_all(
            User(id=user_id, email=f"{user_id}@example.com", name="Author")
            for user_id in (owner_id, other_id)
        )

    yield sessions, owner_id, other_id

    async with sessions.begin() as db:
        await db.execute(delete(Job).where(Job.user_id.in_([owner_id, other_id])))
        await db.execute(delete(User).where(User.id.in_([owner_id, other_id])))


def authoring_job(job_id: UUID, owner_id: UUID) -> StoredJob:
    now = datetime.now(UTC)
    return StoredJob(
        job_id=job_id,
        owner_id=owner_id,
        request=AuthoringRequestRecord(
            prompt="Create a tracing question", difficulty="easy", domain="code"
        ),
        provider="stub",
        model="stub-model",
        library_revision="test-revision",
        created_at=now,
        updated_at=now,
    )


async def test_postgres_round_trips_full_exact_evidence_and_enforces_ownership(
    authoring_db: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, owner_id, other_id = authoring_db
    writer, reader = PostgresAuthoringStore(sessions), PostgresAuthoringStore(sessions)
    command = make_command()
    job = authoring_job(command.job_id, owner_id)
    job.request.prompt = "Create a tracing question from this pasted string: \x00"
    await writer.create_job(job)
    with pytest.raises(AuthoringConflict, match="already exists"):
        await writer.create_job(job)
    await writer.mark_started(job.job_id)
    report = make_checked_report(command)
    assert report.artifact is not None
    artifact_bytes = report.artifact.model_dump_json().encode()
    report_bytes = report.model_dump_json().encode()
    completed = await writer.complete_job(job.job_id, report)

    evidence = await reader.read_evidence(job.job_id, owner_id=owner_id)
    assert evidence.job == completed
    assert evidence.artifact_bytes == artifact_bytes
    assert [item.job_id for item in await reader.list_jobs(owner_id)] == [job.job_id]
    assert await reader.list_jobs(other_id) == []
    with pytest.raises(AuthoringNotFound):
        await reader.read_evidence(job.job_id, owner_id=other_id)
    with pytest.raises(AuthoringNotFound):
        await reader.read_report(job.job_id, owner_id=other_id)

    async with sessions() as db:
        row, data = (
            await db.execute(
                select(Job, AuthoringJobData).join(AuthoringJobData).where(Job.id == job.job_id)
            )
        ).one()
        assert row.type == JobType.AI_AUTHORING
        assert row.status == "completed"
        assert data.artifact_sha256 == hashlib.sha256(artifact_bytes).hexdigest()
        assert data.report_sha256 == hashlib.sha256(report_bytes).hexdigest()
        assert data.report_bytes == report_bytes


async def test_postgres_serializes_competing_transitions_and_checks_report_hash(
    authoring_db: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, owner_id, other_id = authoring_db
    first, second = PostgresAuthoringStore(sessions), PostgresAuthoringStore(sessions)
    command = make_command()
    job = authoring_job(command.job_id, owner_id)
    await first.create_job(job)
    starts = await asyncio.gather(
        first.mark_started(job.job_id), second.mark_started(job.job_id), return_exceptions=True
    )
    assert sum(isinstance(result, StoredJob) for result in starts) == 1
    assert sum(isinstance(result, AuthoringConflict) for result in starts) == 1

    report = make_checked_report(command)
    completions = await asyncio.gather(
        first.complete_job(job.job_id, report),
        second.complete_job(job.job_id, report),
        return_exceptions=True,
    )
    assert sum(isinstance(result, StoredJob) for result in completions) == 1
    assert sum(isinstance(result, AuthoringConflict) for result in completions) == 1
    evidence = await second.read_evidence(job.job_id, owner_id=owner_id)
    digest = evidence.job.artifact_sha256
    assert digest is not None
    approval = ApprovalRecord(
        decision="approved",
        artifact_sha256=digest,
        reviewer_id=owner_id,
        reviewer=str(owner_id),
        decided_at=datetime.now(UTC),
    )
    with pytest.raises(AuthoringNotFound):
        await first.save_approval(job.job_id, approval.model_copy(update={"reviewer_id": other_id}))

    # The bytes still parse to the same report and artifact identity.
    async with sessions.begin() as db:
        await db.execute(
            update(AuthoringJobData)
            .where(AuthoringJobData.job_id == job.job_id)
            .values(report_bytes=report.model_dump_json(indent=2).encode())
        )
    with pytest.raises(AuthoringConflict, match="integrity"):
        await first.save_approval(job.job_id, approval)
    assert (await second.get_owned_job(job.job_id, owner_id)).approval is None


async def test_postgres_generation_requires_the_approval_that_authorized_it(
    authoring_db: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, owner_id, other_id = authoring_db
    first, second = PostgresAuthoringStore(sessions), PostgresAuthoringStore(sessions)
    command = make_command()
    job = authoring_job(command.job_id, owner_id)
    await first.create_job(job)
    completed = await first.complete_job(job.job_id, make_checked_report(command))
    digest = completed.artifact_sha256
    assert digest is not None
    approval = ApprovalRecord(
        decision="approved",
        artifact_sha256=digest,
        reviewer_id=owner_id,
        reviewer=str(owner_id),
        decided_at=datetime.now(UTC),
    )
    await first.save_approval(job.job_id, approval)
    await second.save_approval(job.job_id, approval.model_copy(update={"decision": "rejected"}))
    replacement = approval.model_copy(update={"note": "Reviewed again"})
    await second.save_approval(job.job_id, replacement)
    generated = GenerationMetadata(
        artifact_sha256=digest,
        base_seed=0,
        requested_quantity=1,
        actual_quantity=1,
        distinct_count=1,
        duplicates=0,
        created_at=datetime.now(UTC),
    )

    with pytest.raises(AuthoringConflict, match="must remain approved"):
        await first.save_generation(
            job.job_id, generated, owner_id=owner_id, expected_approval=approval
        )
    with pytest.raises(AuthoringNotFound):
        await first.save_generation(
            job.job_id, generated, owner_id=other_id, expected_approval=replacement
        )

    assert (await second.get_owned_job(job.job_id, owner_id)).last_generation is None
    await first.save_generation(
        job.job_id, generated, owner_id=owner_id, expected_approval=replacement
    )
    assert (await second.get_owned_job(job.job_id, owner_id)).last_generation == generated


async def test_inline_authoring_http_uses_shared_postgres_across_replicas(
    authoring_db: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
    tmp_path: Path,
) -> None:
    sessions, owner_id, other_id = authoring_db
    legacy_path = tmp_path / "unused-authoring-files"
    configuration = AuthoringSettings(
        executor="inline",
        storage_dir=legacy_path,
        dispatch_interval_seconds=0.02,
        model_profiles=[
            AuthoringProfileSettings(
                id="test-profile", label="Test", provider="stub", model="stub-model"
            )
        ],
    )

    def current_user(request: Request) -> User:
        user_id = other_id if request.headers.get("X-Test-Owner") == "other" else owner_id
        return User(id=user_id)

    def app_for(service: AiAuthoringService) -> FastAPI:
        app = FastAPI()
        app.exception_handler(EdCraftBaseException)(edcraft_exception_handler)
        app.include_router(authoring_router)
        app.dependency_overrides[get_ai_authoring_service] = lambda: service
        app.dependency_overrides[get_current_user] = current_user
        app.dependency_overrides[get_provider_connection_service] = lambda: None
        return app

    async with AsyncExitStack() as stack:
        clients: list[AsyncClient] = []
        executors: list[StaticExecutor] = []
        for _ in range(2):
            executor = StaticExecutor(make_checked_report)
            executors.append(executor)
            service = AiAuthoringService(
                configuration, store=DurableAuthoringStore(sessions), executor=executor
            )
            await service.startup()
            stack.push_async_callback(service.shutdown)
            clients.append(
                await stack.enter_async_context(
                    AsyncClient(
                        transport=ASGITransport(app=app_for(service)), base_url="http://test"
                    )
                )
            )

        first, second = clients
        submitted = await first.post(
            "/ai-authoring/jobs",
            json={
                "domain": "code",
                "difficulty": "easy",
                "prompt": "Write a question",
                "model_profile": "test-profile",
            },
        )
        assert submitted.status_code == 202, submitted.text
        job_url = submitted.json()["status_url"]
        async with asyncio.timeout(5):
            while True:
                polled = await second.get(job_url)
                assert polled.status_code == 200, polled.text
                if polled.json()["state"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.02)
        assert polled.json()["state"] == "completed", polled.text
        assert sum(executor.calls for executor in executors) == 1

        invisible = await second.get(f"{job_url}/review", headers={"X-Test-Owner": "other"})
        assert invisible.status_code == 404, invisible.text
        review = await second.get(f"{job_url}/review")
        assert review.status_code == 200, review.text
        assert "never-return-this" not in review.text
        digest = review.json()["artifact_sha256"]
        preview = await second.post(f"{job_url}/preview", json={"seed": 7})
        assert preview.status_code == 200, preview.text
        approved = await first.post(
            f"{job_url}/approval", json={"artifact_sha256": digest, "decision": "approved"}
        )
        assert approved.status_code == 200, approved.text
        generated = await second.post(
            f"{job_url}/generate", json={"artifact_sha256": digest, "seed": 7, "quantity": 2}
        )
        assert generated.status_code == 200, generated.text
        assert generated.json()["questions"][0] == preview.json()["question"]
        refreshed = await first.get(job_url)
        assert refreshed.json()["last_generation"]["requested_quantity"] == 2

    assert not legacy_path.exists()
