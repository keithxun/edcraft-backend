"""Exercise the authoring scheduler across independent PostgreSQL connections."""

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from threading import Event
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
import pytest
from edcraft_validator.application.authoring_api import AuthoringJobCommand
from edcraft_validator.application.authoring_contracts import AuthoringResult
from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    NomadSettings,
)
from edcraft_backend.dependencies import get_ai_authoring_service
from edcraft_backend.executors.nomad_authoring import NomadAuthoringExecutor, nomad_authoring_job_id
from edcraft_backend.main import app
from edcraft_backend.models.ai_authoring import AuthoringJobData
from edcraft_backend.models.job import Job, JobType
from edcraft_backend.models.user import User
from edcraft_backend.schemas.ai_authoring import (
    ApprovalRequest,
    AuthoringJudgeSelections,
    AuthoringModelSelection,
    GenerateRequest,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.ai_authoring_dispatch import AuthoringWorkerUnauthorized
from edcraft_backend.services.ai_authoring_dispatch_store import DurableAuthoringStore
from edcraft_backend.services.ai_authoring_lifecycle import AuthoringAdmissionFull
from edcraft_backend.services.ai_authoring_runtime import JobExecutor
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.ai_authoring_store import StoredJob
from edcraft_backend.services.ai_domains import LIBRARY_REVISION
from tests.ai_authoring_helpers import StaticExecutor, make_checked_report

pytestmark = pytest.mark.integration

_SUBMISSION = SubmitAuthoringJobRequest(
    prompt="Create a deterministic tracing question",
    difficulty="easy",
    model_profile="worker",
    judges=AuthoringJudgeSelections(relevance=AuthoringModelSelection(model_profile="worker")),
)


class NomadSimulation:
    """Simulate registration, lost responses and delayed allocation exit."""

    def __init__(self) -> None:
        self.registrations: dict[str, dict[str, Any]] = {}
        self.stop_requests: set[str] = set()
        self.stopped: set[str] = set()
        self.disconnected: set[str] = set()
        self.registration_requests: list[str] = []
        self.lose_response = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["EnforceIndex"] is True
            name = str(body["Job"]["ID"])
            self.registration_requests.append(name)
            if name in self.registrations:
                return httpx.Response(409)
            self.registrations[name] = body["Job"]
            if self.lose_response:
                self.lose_response = False
                raise httpx.ReadTimeout("Nomad accepted the job", request=request)
            return httpx.Response(200, json={})

        endpoint = request.url.path.removeprefix("/v1/job/")
        name = endpoint.removesuffix("/allocations")
        if name not in self.registrations:
            return httpx.Response(404)
        if request.method == "DELETE":
            assert request.url.params["purge"] == "false"
            self.stop_requests.add(name)
            return httpx.Response(200, json={})
        if endpoint.endswith("/allocations"):
            status = "running"
            if name in self.stopped:
                status = "complete"
            elif name in self.disconnected:
                status = "lost"
            return httpx.Response(200, json=[{"ClientStatus": status}])
        return httpx.Response(
            200,
            json={
                "Status": "dead"
                if name in (self.stopped | self.disconnected | self.stop_requests)
                else "running"
            },
        )

    def env(self, name: str) -> dict[str, str]:
        env: dict[str, str] = self.registrations[name]["TaskGroups"][0]["Tasks"][0]["Env"]
        return env


@pytest.fixture
async def dispatch_db(
    test_engine: AsyncEngine, test_db_setup: None
) -> AsyncIterator[tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]]]:
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    owners = tuple(uuid4() for _ in range(3))
    async with sessions.begin() as db:
        db.add_all(User(id=owner, email=f"{owner}@example.com", name="Teacher") for owner in owners)
    yield sessions, owners
    async with sessions.begin() as db:
        await db.execute(delete(Job).where(Job.user_id.in_(owners)))
        await db.execute(delete(User).where(User.id.in_(owners)))


@pytest.fixture
def simulation(monkeypatch: pytest.MonkeyPatch) -> NomadSimulation:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-worker-provider-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return NomadSimulation()


def replica(
    sessions: async_sessionmaker[AsyncSession],
    simulation: NomadSimulation,
    mode: Literal["inline", "nomad"] = "nomad",
    runtime_executor: JobExecutor | None = None,
    **overrides: Any,
) -> AiAuthoringService:
    settings = AuthoringSettings(
        executor=mode,
        nomad_container_image="worker:revision",
        model_profiles=[
            AuthoringProfileSettings(id="worker", label="Worker", provider="openai", model="demo")
        ],
        **overrides,
    )
    return AiAuthoringService(
        settings,
        store=DurableAuthoringStore(sessions),
        executor=runtime_executor,
        nomad=NomadAuthoringExecutor(
            NomadSettings(),
            image="worker:revision",
            callback_base_url="http://api.test",
            transport=httpx.MockTransport(simulation),
        ),
    )


async def test_concurrent_admission_and_claim_enforce_both_limits(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, owners = dispatch_db
    first = replica(sessions, simulation, max_queue=2, max_inflight_jobs_per_user=3)
    second = replica(sessions, simulation, max_queue=2, max_inflight_jobs_per_user=3)
    owner_outcomes = await asyncio.gather(
        *(service.submit(owners[0], _SUBMISSION) for service in [first, second] * 3),
        return_exceptions=True,
    )
    assert sum(isinstance(value, StoredJob) for value in owner_outcomes) == 3
    assert sum(isinstance(value, AuthoringAdmissionFull) for value in owner_outcomes) == 3
    total_outcomes = await asyncio.gather(
        *(
            service.submit(owner, _SUBMISSION)
            for service in [first, second] * 3
            for owner in owners[1:]
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, StoredJob) for value in total_outcomes) == 3
    await asyncio.gather(
        first.worker_lifecycle().dispatch_once(), second.worker_lifecycle().dispatch_once()
    )
    async with sessions() as db:
        running = (
            await db.scalars(
                select(Job.user_id).where(Job.user_id.in_(owners), Job.status == "running")
            )
        ).all()
    assert len(running) == 4
    assert max(Counter(running).values()) == 2
    assert len(simulation.registrations) == 4


async def test_replicas_resume_one_ambiguous_nomad_registration_and_accept_one_result(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (owner, *_rest) = dispatch_db
    first, second, restarted = (replica(sessions, simulation) for _ in range(3))
    job = await first.submit(owner, _SUBMISSION)
    store = DurableAuthoringStore(sessions)
    # An API process dies after committing a claim, before making its Nomad call.
    claim = await store.claim_nomad(first.settings)
    assert claim is not None and not simulation.registrations
    name = nomad_authoring_job_id(*claim)
    simulation.lose_response = True
    await second.worker_lifecycle().dispatch_once()
    assert simulation.registration_requests == [name]
    await restarted.worker_lifecycle().dispatch_once()
    assert simulation.registration_requests == [name]
    env = simulation.env(name)
    assert "synthetic-worker-provider-key" not in str(simulation.registrations)
    async with sessions() as db:
        data = await db.get(AuthoringJobData, job.job_id)
        assert data is not None and data.execution_payload
        assert "synthetic-worker-provider-key" not in data.execution_payload
        assert env["EDCRAFT_WORKER_TOKEN"] not in data.execution_payload

    app.dependency_overrides[get_ai_authoring_service] = lambda: restarted
    headers = {
        "Authorization": f"Bearer {env['EDCRAFT_WORKER_TOKEN']}",
        "X-EdCraft-Worker-Allocation": str(uuid4()),
    }
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            wrong = {**headers, "Authorization": "Bearer expired"}
            assert (await client.get(env["EDCRAFT_CONTEXT_URL"], headers=wrong)).status_code == 401
            context = await client.get(env["EDCRAFT_CONTEXT_URL"], headers=headers)
            assert context.status_code == 200
            assert context.headers["cache-control"] == "no-store"
            plan = context.json()["plan"]
            assert plan["author"]["api_key"] == "synthetic-worker-provider-key"
            command = AuthoringJobCommand.model_validate(context.json()["command"])
            assert command.job_id == job.job_id
            checking_started = datetime.now(UTC)
            progress = await client.post(
                env["EDCRAFT_PROGRESS_URL"],
                headers=headers,
                json={
                    "phase": "checking",
                    "attempt": 1,
                    "phase_started_at": checking_started.isoformat(),
                },
            )
            assert progress.status_code == 204
            older = await client.post(
                env["EDCRAFT_PROGRESS_URL"],
                headers=headers,
                json={
                    "phase": "generating",
                    "attempt": 1,
                    "phase_started_at": (checking_started - timedelta(seconds=1)).isoformat(),
                },
            )
            assert older.status_code == 204
            latest = (await first.get_job(owner, job.job_id)).progress
            assert latest is not None and latest.phase == "checking"

            report = make_checked_report(command).model_copy(
                update={"provider": "openai", "model": "demo"}
            )
            payload = {
                "library_revision": LIBRARY_REVISION,
                "report": report.model_dump(mode="json"),
            }
            replaced = {**headers, "X-EdCraft-Worker-Allocation": str(uuid4())}
            stale_url = env["EDCRAFT_CALLBACK_URL"].replace(str(claim[1]), str(uuid4()))
            assert (await client.post(stale_url, headers=headers, json=payload)).status_code == 401
            assert (
                await client.post(env["EDCRAFT_CALLBACK_URL"], headers=replaced, json=payload)
            ).status_code == 401
            assert (
                await client.post(
                    env["EDCRAFT_CALLBACK_URL"],
                    headers=headers,
                    json={**payload, "library_revision": "wrong"},
                )
            ).status_code == 422
            unrelated = {
                **payload,
                "report": {**payload["report"], "request": {"prompt": "Other job"}},
            }
            assert (
                await client.post(env["EDCRAFT_CALLBACK_URL"], headers=headers, json=unrelated)
            ).status_code == 422
            app.dependency_overrides[get_ai_authoring_service] = lambda: second
            duplicate = await asyncio.gather(
                *(
                    client.post(env["EDCRAFT_CALLBACK_URL"], headers=headers, json=payload)
                    for _ in range(2)
                )
            )
            assert [response.status_code for response in duplicate] == [204, 204]
            altered = {**payload, "report": {**payload["report"], "reason": "a different result"}}
            assert (
                await client.post(env["EDCRAFT_CALLBACK_URL"], headers=headers, json=altered)
            ).status_code == 409
    finally:
        app.dependency_overrides.pop(get_ai_authoring_service, None)

    finished = await restarted.get_job(owner, job.job_id)
    assert finished.state == "completed"
    assert finished.artifact_sha256
    assert await store.active_attempts("nomad") == [claim]
    assert report.artifact is not None
    evidence = await store.read_evidence(job.job_id, owner_id=owner)
    assert evidence.artifact_bytes == report.artifact.model_dump_json().encode()
    async with sessions() as db:
        data = await db.get(AuthoringJobData, job.job_id)
        assert data is not None and data.execution_payload is None
        assert data.report_bytes == report.model_dump_json().encode()
    await first.decide(
        owner,
        job.job_id,
        ApprovalRequest(decision="approved", artifact_sha256=finished.artifact_sha256),
    )
    generation = GenerateRequest(seed=2, quantity=2, artifact_sha256=finished.artifact_sha256)
    assert (await first.generate(owner, job.job_id, generation)).questions == (
        await restarted.generate(owner, job.job_id, generation)
    ).questions
    simulation.stopped.add(name)
    await second.worker_lifecycle().dispatch_once()
    assert await store.active_attempts("nomad") == []


async def test_timeout_releases_capacity_only_after_nomad_confirms_exit(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (owner, other, _) = dispatch_db
    service = replica(sessions, simulation, max_active_jobs=1, max_queue=1)
    job = await service.submit(owner, _SUBMISSION)
    next_job = await service.submit(other, _SUBMISSION)
    await service.worker_lifecycle().dispatch_once()
    (name,) = simulation.registrations
    [attempt] = await DurableAuthoringStore(sessions).active_attempts("nomad")
    env = simulation.env(name)
    async with sessions.begin() as db:
        await db.execute(
            update(AuthoringJobData)
            .where(AuthoringJobData.job_id == job.job_id)
            .values(attempt_deadline_at=datetime.now(UTC) - timedelta(seconds=1))
        )

    with pytest.raises(AuthoringWorkerUnauthorized):
        await service.worker_lifecycle().worker_context(
            *attempt, env["EDCRAFT_WORKER_TOKEN"], str(uuid4())
        )
    await service.worker_lifecycle().dispatch_once()
    assert name in simulation.stop_requests
    assert (await service.get_job(owner, job.job_id)).state == "failed"
    assert (await service.get_job(other, next_job.job_id)).state == "queued"
    simulation.stopped.add(name)
    await service.worker_lifecycle().dispatch_once()
    assert (await service.get_job(other, next_job.job_id)).state == "running"
    assert len(simulation.registrations) == 2


@pytest.mark.parametrize("sends_error", [True, False], ids=["worker-error", "lost-callback"])
async def test_worker_exit_without_result_never_replays_the_model(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
    sends_error: bool,
) -> None:
    sessions, (owner, *_rest) = dispatch_db
    service = replica(sessions, simulation)
    job = await service.submit(owner, _SUBMISSION)
    lifecycle = service.worker_lifecycle()
    await lifecycle.dispatch_once()
    store = DurableAuthoringStore(sessions)
    [attempt] = await store.active_attempts("nomad")
    name = nomad_authoring_job_id(*attempt)
    worker = (*attempt, simulation.env(name)["EDCRAFT_WORKER_TOKEN"], str(uuid4()))
    await lifecycle.worker_context(*worker)
    if sends_error:
        await lifecycle.worker_error(*worker)
        await lifecycle.worker_error(*worker)
        assert (await service.get_job(owner, job.job_id)).state == "failed"
        assert await store.active_attempts("nomad") == [attempt]

    simulation.stopped.add(name)
    await lifecycle.dispatch_once()
    await lifecycle.dispatch_once()
    failed = await service.get_job(owner, job.job_id)
    assert failed.state == "failed" and failed.result is None
    assert failed.error is not None and failed.error.code == "AUTHORING_FAILED"
    assert await store.active_attempts("nomad") == []
    assert simulation.registration_requests == [name]


async def test_worker_context_preserves_an_unconfigured_advisory_role(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, (owner, *_rest) = dispatch_db
    monkeypatch.setenv("EDCRAFT_JUDGE_PROVIDER", "soclaas")
    monkeypatch.delenv("SOCLAAS_API_KEY", raising=False)
    service = replica(sessions, simulation)
    job = await service.submit(owner, _SUBMISSION.model_copy(update={"judges": None}))
    lifecycle = service.worker_lifecycle()
    await lifecycle.dispatch_once()
    [(job_id, attempt_id)] = await DurableAuthoringStore(sessions).active_attempts("nomad")
    assert job.job_id == job_id
    env = simulation.env(nomad_authoring_job_id(job_id, attempt_id))
    context = await lifecycle.worker_context(
        job_id, attempt_id, env["EDCRAFT_WORKER_TOKEN"], str(uuid4())
    )
    assert isinstance(context["plan"], dict)
    assert context["plan"]["relevance"]["provider"] == "soclaas"
    assert context["plan"]["relevance"]["api_key"] is None


async def test_disconnected_worker_cannot_release_capacity_before_its_deadline(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (owner, other, _) = dispatch_db
    service = replica(sessions, simulation, max_active_jobs=1, max_queue=1)
    job = await service.submit(owner, _SUBMISSION)
    next_job = await service.submit(other, _SUBMISSION)
    await service.worker_lifecycle().dispatch_once()
    (name,) = simulation.registrations
    simulation.disconnected.add(name)
    await service.worker_lifecycle().dispatch_once()
    assert (await service.get_job(owner, job.job_id)).state == "running"
    assert (await service.get_job(other, next_job.job_id)).state == "queued"
    assert simulation.registration_requests == [name]

    # Even if the node stays offline, the worker has a hard process timer.
    async with sessions.begin() as db:
        await db.execute(
            update(AuthoringJobData)
            .where(AuthoringJobData.job_id == job.job_id)
            .values(attempt_deadline_at=datetime.now(UTC) - timedelta(seconds=31))
        )
    await service.worker_lifecycle().dispatch_once()
    assert (await service.get_job(owner, job.job_id)).state == "failed"
    assert (await service.get_job(other, next_job.job_id)).state == "running"


async def test_inline_attempt_survives_another_replica_and_fails_if_abandoned(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (owner, *_rest) = dispatch_db
    first, second = (replica(sessions, simulation, "inline") for _ in range(2))
    job = await first.submit(owner, _SUBMISSION)
    store = DurableAuthoringStore(sessions)
    async with store.claim_inline(first.settings) as attempt:
        assert attempt is not None and attempt[0] == job.job_id
        await second.worker_lifecycle().dispatch_once()
        assert (await second.get_job(owner, job.job_id)).state == "running"
    # Closing the advisory-lock connection simulates a lost inline owner.
    await second.worker_lifecycle().dispatch_once()
    abandoned = await first.get_job(owner, job.job_id)
    assert abandoned.state == "failed"
    assert abandoned.error is not None and abandoned.error.code == "SERVER_RESTARTED"
    assert await store.active_attempts("inline") == [attempt]
    assert not simulation.registrations


async def test_lost_inline_database_session_holds_slot_until_live_runner_returns(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (owner, next_owner, _) = dispatch_db
    entered, finish = Event(), Event()

    def blocked_report(command: AuthoringJobCommand) -> AuthoringResult:
        entered.set()
        assert finish.wait(timeout=10)
        return make_checked_report(command)

    normal_executor = StaticExecutor(make_checked_report)
    first = replica(
        sessions, simulation, "inline", StaticExecutor(blocked_report), max_active_jobs=1
    )
    second = replica(sessions, simulation, "inline", normal_executor, max_active_jobs=1)
    job = await first.submit(owner, _SUBMISSION)
    next_job = await second.submit(next_owner, _SUBMISSION)
    store = DurableAuthoringStore(sessions)
    await first.worker_lifecycle().dispatch_once()
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        [attempt] = await store.active_attempts("inline")
        advisory_key = attempt[1].int >> 65
        # Terminate only the PostgreSQL session holding this test attempt's lock.
        async with sessions.begin() as db:
            terminated = await db.scalar(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_locks "
                    "WHERE locktype='advisory' AND granted AND objsubid=1 "
                    "AND classid=CAST(:high AS oid) AND objid=CAST(:low AS oid)"
                ),
                {"high": advisory_key >> 32, "low": advisory_key & 0xFFFFFFFF},
            )
            assert terminated
        for _ in range(20):
            await second.worker_lifecycle().dispatch_once()
            if (await second.get_job(owner, job.job_id)).state == "failed":
                break
            await asyncio.sleep(0.02)
        disconnected = await second.get_job(owner, job.job_id)
        assert disconnected.error is not None and disconnected.error.code == "SERVER_RESTARTED"
        assert await store.active_attempts("inline") == [attempt]
        assert (await second.get_job(next_owner, next_job.job_id)).state == "queued"
        assert normal_executor.calls == 0
        async with sessions() as db:
            data = await db.get(AuthoringJobData, job.job_id)
            assert data is not None and data.execution_payload is None
            assert data.worker_claim_id and ":" in data.worker_claim_id
    finally:
        finish.set()
        await first.shutdown()
    assert (await second.get_job(owner, job.job_id)).state == "failed"
    assert await store.active_attempts("inline") == []
    await second.worker_lifecycle().dispatch_once()
    await second.shutdown()
    assert (await second.get_job(next_owner, next_job.job_id)).state == "completed"
    assert normal_executor.calls == 1
    assert await store.active_attempts("inline") == []


async def test_owner_deleted_while_queued_clears_only_its_authoring_inputs(
    dispatch_db: tuple[async_sessionmaker[AsyncSession], tuple[UUID, ...]],
    simulation: NomadSimulation,
) -> None:
    sessions, (deleted_owner, owner, _) = dispatch_db
    service = replica(sessions, simulation)
    abandoned = await service.submit(deleted_owner, _SUBMISSION)
    kept = await service.submit(owner, _SUBMISSION)
    unrelated_id = uuid4()
    try:
        async with sessions.begin() as db:
            db.add(Job(id=unrelated_id, type=JobType.GENERATE_INPUTS, user_id=None))
            await db.execute(delete(User).where(User.id == deleted_owner))
        await service.worker_lifecycle().dispatch_once()
        async with sessions() as db:
            row = await db.get(Job, abandoned.job_id)
            data = await db.get(AuthoringJobData, abandoned.job_id)
            unrelated = await db.get(Job, unrelated_id)
            assert row is not None and row.user_id is None and row.status == "failed"
            assert data is not None and data.execution_payload is None
            assert data.attempt_id is None
            assert unrelated is not None and unrelated.status == "queued"
        assert (await service.get_job(owner, kept.job_id)).state == "running"
        assert len(simulation.registrations) == 1
    finally:
        async with sessions.begin() as db:
            await db.execute(delete(Job).where(Job.id.in_([abandoned.job_id, unrelated_id])))
