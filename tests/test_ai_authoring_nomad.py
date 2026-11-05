"""Flag-gated Nomad authoring mode: executor, tokens, callbacks and timeout."""

from __future__ import annotations

import asyncio
import base64
import json
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
import pytest
from edcraft_validator.application.authoring_api import AuthoringJobCommand
from edcraft_validator.application.authoring_contracts import AuthoringError
from httpx import ASGITransport, AsyncClient

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
)
from edcraft_backend.dependencies import (
    get_ai_authoring_service,
    get_current_user,
    get_provider_connection_service,
)
from edcraft_backend.executors.nomad_authoring import NomadAuthoringExecutor
from edcraft_backend.main import app
from edcraft_backend.services.ai_authoring_service import (
    LIBRARY_REVISION,
    AiAuthoringService,
    hash_worker_token,
)
from edcraft_backend.services.ai_authoring_store import StoredJob
from tests.ai_authoring_helpers import (
    StaticExecutor,
    make_checked_report,
    make_command,
    make_needs_review_report,
)

PROFILE = AuthoringProfileSettings(
    id="test-profile",
    label="Test author",
    provider="soclaas",
    model="x-test-2",
)
SERVER_API_KEY = "server-key-never-in-nomad"
SERVER_BASE_URL = "https://soclaas.test/v1"
CALLBACK_BASE = "http://backend.test:8000"


@dataclass
class FakeUser:
    id: UUID


@dataclass
class NomadHarness:
    client: AsyncClient
    service: AiAuthoringService
    user: FakeUser
    executor: Any


class NoConnections:
    """Connection service stub for authoring tests without a database."""

    async def list_connections(self, owner_id: UUID) -> list[Any]:
        return []


@dataclass
class Submission:
    job_id: UUID
    command: dict[str, Any]
    callback_url: str
    progress_url: str
    context_url: str
    error_url: str
    timeout_seconds: float


class FakeRemoteExecutor:
    """Record submissions and stops without contacting a live Nomad server."""

    def __init__(self) -> None:
        self.submissions: list[Submission] = []
        self.stops: list[str] = []
        self.statuses: list[str] = []

    async def submit(
        self,
        job_id: UUID,
        command: dict[str, Any],
        callback_url: str,
        progress_url: str,
        context_url: str,
        error_url: str,
        timeout_seconds: float,
    ) -> str:
        self.submissions.append(
            Submission(
                job_id=job_id,
                command=command,
                callback_url=callback_url,
                progress_url=progress_url,
                context_url=context_url,
                error_url=error_url,
                timeout_seconds=timeout_seconds,
            )
        )
        return f"edcraft-authoring-{job_id}"

    async def stop(self, nomad_job_id: str) -> None:
        self.stops.append(nomad_job_id)

    async def status(self, nomad_job_id: str) -> str:
        self.statuses.append(nomad_job_id)
        return "running"


@pytest.fixture(autouse=True)
def server_profile_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply the server-profile key the worker context endpoint resolves."""
    monkeypatch.setenv("SOCLAAS_API_KEY", SERVER_API_KEY)
    monkeypatch.setenv("SOCLAAS_BASE_URL", SERVER_BASE_URL)
    monkeypatch.setenv("SOCLAAS_TIMEOUT_SECONDS", "600")


@asynccontextmanager
async def nomad_env(
    tmp_path: Path,
    *,
    executor: str = "nomad",
    remote: FakeRemoteExecutor | None = None,
    inline_executor: Any = None,
    job_timeout_seconds: float = 5.0,
    max_active_jobs: int = 2,
    max_active_jobs_per_user: int = 2,
    max_inflight_jobs_per_user: int = 4,
    max_queue: int = 4,
) -> AsyncGenerator[NomadHarness, None]:
    settings = AuthoringSettings(
        storage_dir=tmp_path / "authoring",
        model_profiles=[PROFILE],
        max_active_jobs=max_active_jobs,
        max_active_jobs_per_user=max_active_jobs_per_user,
        max_inflight_jobs_per_user=max_inflight_jobs_per_user,
        max_queue=max_queue,
        shutdown_timeout_seconds=1.0,
        executor=executor,
        worker_token_ttl_seconds=7200,
        job_timeout_seconds=job_timeout_seconds,
        nomad_callback_base_url=CALLBACK_BASE,
    )
    service = AiAuthoringService(
        settings,
        executor=inline_executor,
        remote_executor=remote if executor == "nomad" else None,
    )
    await service.startup()
    user = FakeUser(uuid4())

    def current_user() -> FakeUser:
        return user

    def authoring_service() -> AiAuthoringService:
        return service

    app.dependency_overrides[get_ai_authoring_service] = authoring_service
    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_provider_connection_service] = NoConnections
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield NomadHarness(
                client=client,
                service=service,
                user=user,
                executor=remote or inline_executor,
            )
    finally:
        app.dependency_overrides.clear()
        await service.wait_for_idle()
        await service.shutdown(0.5)


def payload(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "domain": "code",
        "prompt": "Create an arithmetic tracing question",
        "difficulty": "easy",
        "num_distractors": 3,
        "model_profile": "test-profile",
    }
    value.update(overrides)
    return value


async def submit(env: NomadHarness, **overrides: Any) -> UUID:
    response = await env.client.post("/ai-authoring/jobs", json=payload(**overrides))
    assert response.status_code == 202, response.text
    return UUID(response.json()["job_id"])


def raw_token(submission: Submission) -> str:
    return urlparse(submission.context_url).path.split("/")[-2]


def result_body(report: Any, revision: str = LIBRARY_REVISION) -> dict[str, Any]:
    return {
        "library_revision": revision,
        "report": report.model_dump(mode="json"),
    }


async def wait_for_submission(
    remote: FakeRemoteExecutor, index: int = 0
) -> Submission:
    for _ in range(200):
        if len(remote.submissions) > index:
            return remote.submissions[index]
        await asyncio.sleep(0.005)
    raise AssertionError("worker submission never reached the fake executor")


async def wait_for_running(service: AiAuthoringService, job_id: UUID) -> StoredJob:
    for _ in range(100):
        job = service.store.get_job(job_id)
        if job.state == "running":
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("job never started")


@pytest.fixture
def remote() -> FakeRemoteExecutor:
    return FakeRemoteExecutor()


@pytest.mark.asyncio
async def test_nomad_job_spec_has_no_credential_material() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"EvalID": "eval-1"})

    executor = NomadAuthoringExecutor(
        host="nomad.test",
        port=4646,
        library_revision=LIBRARY_REVISION,
        container_image="registry.test/authoring-worker:v1",
        cpu_mhz=500,
        memory_mb=512,
        force_pull=False,
        transport=httpx.MockTransport(handler),
    )
    command = make_command()
    nomad_job_id = await executor.submit(
        command.job_id,
        command.model_dump(mode="json"),
        f"{CALLBACK_BASE}/ai-authoring/worker/token/result",
        f"{CALLBACK_BASE}/ai-authoring/worker/token/progress",
        f"{CALLBACK_BASE}/ai-authoring/worker/token/context",
        f"{CALLBACK_BASE}/ai-authoring/worker/token/error",
        1800,
    )

    assert nomad_job_id == f"edcraft-authoring-{command.job_id}"
    spec = captured["body"]["Job"]
    group = spec["TaskGroups"][0]
    task = group["Tasks"][0]
    assert spec["Type"] == "batch"
    assert group["Count"] == 1
    assert group["RestartPolicy"] == {"Attempts": 0, "Mode": "fail"}
    assert task["Config"]["image"] == "registry.test/authoring-worker:v1"
    assert task["Config"]["command"] == "python"
    assert task["Config"]["args"] == ["-m", "edcraft_validator.worker"]
    assert "Templates" not in task
    assert task["Resources"] == {"CPU": 500, "MemoryMB": 512}
    assert set(task["Env"]) == {
        "EDCRAFT_PARAMS_B64",
        "EDCRAFT_CONTEXT_URL",
        "EDCRAFT_PROGRESS_URL",
        "EDCRAFT_CALLBACK_URL",
        "EDCRAFT_ERROR_URL",
        "EDCRAFT_LIBRARY_REVISION",
        "EDCRAFT_JOB_ID",
    }
    decoded = json.loads(base64.b64decode(task["Env"]["EDCRAFT_PARAMS_B64"]))
    assert decoded == command.model_dump(mode="json")
    assert task["Env"]["EDCRAFT_LIBRARY_REVISION"] == LIBRARY_REVISION
    serialized = json.dumps(spec)
    assert SERVER_API_KEY not in serialized
    assert "api_key" not in serialized
    assert "never-return-this" not in serialized


@pytest.mark.asyncio
async def test_token_is_hashed_at_rest_and_context_is_no_store(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        job_id = await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        job = env.service.store.get_job(job_id)

        assert job.worker_token_hash == hash_worker_token(token)
        assert job.worker_token_hash != token
        assert job.worker_token_expires_at is not None
        assert job.nomad_job_id == f"edcraft-authoring-{job_id}"
        raw_record = (env.service.store.jobs_dir / f"{job_id}.json").read_text()
        assert token not in raw_record
        assert SERVER_API_KEY not in raw_record

        first = await env.client.get(f"/ai-authoring/worker/{token}/context")
        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store"
        body = first.json()
        assert body["job_id"] == str(job_id)
        assert body["library_revision"] == LIBRARY_REVISION
        assert body["command"]["request"]["prompt"] == payload()["prompt"]
        assert body["plan"]["author"]["provider"] == "soclaas"
        assert body["plan"]["author"]["api_key"] == SERVER_API_KEY
        assert body["plan"]["author"]["base_url"] == SERVER_BASE_URL
        assert body["plan"]["difficulty"]["model"] == "x-test-2"
        assert body["plan"]["relevance"]["timeout_seconds"] == 600

        second = await env.client.get(f"/ai-authoring/worker/{token}/context")
        assert second.status_code == 200
        assert env.service.store.get_job(job_id).worker_context_issued_at is not None


@pytest.mark.asyncio
async def test_context_progress_result_happy_path_and_idempotent_duplicate(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        job_id = await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        command = AuthoringJobCommand.model_validate(submission.command)
        report = make_checked_report(command)

        context = await env.client.get(f"/ai-authoring/worker/{token}/context")
        assert context.status_code == 200, context.text

        await wait_for_running(env.service, job_id)
        progress = await env.client.post(
            f"/ai-authoring/worker/{token}/progress",
            json={"phase": "generating", "attempt": 1},
        )
        assert progress.status_code == 204
        assert env.service.store.get_job(job_id).progress.phase == "generating"

        accepted = await env.client.post(
            f"/ai-authoring/worker/{token}/result", json=result_body(report)
        )
        assert accepted.status_code == 204, accepted.text
        await env.service.wait_for_idle()

        completed = env.service.store.get_job(job_id)
        assert completed.state == "completed"
        assert completed.result is not None
        assert completed.result.status == "checked"
        assert completed.error is None
        assert env.service.store.read_report(job_id).artifact is not None

        duplicate = await env.client.post(
            f"/ai-authoring/worker/{token}/result", json=result_body(report)
        )
        assert duplicate.status_code == 204, duplicate.text
        assert env.service.store.get_job(job_id).state == "completed"

        after_result = await env.client.get(
            f"/ai-authoring/worker/{token}/context"
        )
        assert after_result.status_code == 409


@pytest.mark.asyncio
async def test_conflicting_result_and_worker_error(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        command = AuthoringJobCommand.model_validate(submission.command)

        accepted = await env.client.post(
            f"/ai-authoring/worker/{token}/result",
            json=result_body(make_checked_report(command)),
        )
        assert accepted.status_code == 204
        await env.service.wait_for_idle()

        conflicting = await env.client.post(
            f"/ai-authoring/worker/{token}/result",
            json=result_body(make_needs_review_report(command)),
        )
        assert conflicting.status_code == 409, conflicting.text

        # A worker fatal error fails the job and releases admission.
        second = await submit(env)
        second_token = raw_token(await wait_for_submission(remote, 1))
        failed = await env.client.post(
            f"/ai-authoring/worker/{second_token}/error",
            json={"code": "GENERATION_ERROR", "message": "Provider failed."},
        )
        assert failed.status_code == 204, failed.text
        await env.service.wait_for_idle()
        failed_job = env.service.store.get_job(second)
        assert failed_job.state == "failed"
        assert failed_job.error is not None
        assert failed_job.error.code == "GENERATION_ERROR"

        late_result = await env.client.post(
            f"/ai-authoring/worker/{second_token}/result",
            json=result_body(make_checked_report(command)),
        )
        assert late_result.status_code == 409


@pytest.mark.asyncio
async def test_expired_and_unknown_tokens_are_unauthorized(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        job_id = await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        command = AuthoringJobCommand.model_validate(submission.command)

        assert (
            await env.client.get("/ai-authoring/worker/not-a-token/context")
        ).status_code == 401

        job = env.service.store.get_job(job_id)
        assert job.worker_token_hash is not None
        env.service.store.set_worker_channel(
            job_id,
            job.worker_token_hash,
            datetime.now(UTC) - timedelta(seconds=1),
            job.nomad_job_id or "",
        )

        assert (
            await env.client.get(f"/ai-authoring/worker/{token}/context")
        ).status_code == 401
        assert (
            await env.client.post(
                f"/ai-authoring/worker/{token}/progress",
                json={"phase": "checking", "attempt": 1},
            )
        ).status_code == 401
        assert (
            await env.client.post(
                f"/ai-authoring/worker/{token}/result",
                json=result_body(make_checked_report(command)),
            )
        ).status_code == 401
        assert (
            await env.client.post(
                f"/ai-authoring/worker/{token}/error",
                json={"code": "AUTHORING_FAILED", "message": "expired"},
            )
        ).status_code == 401


@pytest.mark.asyncio
async def test_wrong_revision_is_rejected_without_resolving(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        job_id = await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        command = AuthoringJobCommand.model_validate(submission.command)
        report = make_checked_report(command)

        wrong = await env.client.post(
            f"/ai-authoring/worker/{token}/result",
            json=result_body(report, revision="0" * 40),
        )
        assert wrong.status_code == 422, wrong.text
        assert env.service.store.get_job(job_id).state in {"queued", "running"}

        accepted = await env.client.post(
            f"/ai-authoring/worker/{token}/result", json=result_body(report)
        )
        assert accepted.status_code == 204
        await env.service.wait_for_idle()
        assert env.service.store.get_job(job_id).state == "completed"


@pytest.mark.asyncio
async def test_needs_review_stays_completed_with_template_check_code(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(tmp_path, remote=remote) as env:
        job_id = await submit(env)
        submission = await wait_for_submission(remote)
        token = raw_token(submission)
        command = AuthoringJobCommand.model_validate(submission.command)
        report = make_needs_review_report(
            command,
            failure=AuthoringError(stage="checking", code="TEMPLATE_CHECKS_FAILED"),
        )

        accepted = await env.client.post(
            f"/ai-authoring/worker/{token}/result", json=result_body(report)
        )
        assert accepted.status_code == 204
        await env.service.wait_for_idle()

        job = env.service.store.get_job(job_id)
        assert job.state == "completed"
        assert job.result is not None
        assert job.result.status == "needs_review"
        assert job.result.error is not None
        assert job.result.error.code == "TEMPLATE_CHECKS_FAILED"
        assert job.error is None


@pytest.mark.asyncio
async def test_timeout_stops_nomad_job_and_releases_admission(
    tmp_path: Path, remote: FakeRemoteExecutor
) -> None:
    async with nomad_env(
        tmp_path,
        remote=remote,
        job_timeout_seconds=0.05,
        max_active_jobs=1,
        max_active_jobs_per_user=1,
        max_inflight_jobs_per_user=1,
        max_queue=0,
    ) as env:
        job_id = await submit(env)
        await env.service.wait_for_idle()

        job = env.service.store.get_job(job_id)
        assert job.state == "failed"
        assert job.error is not None
        assert job.error.code == "AUTHORING_TIMEOUT"
        assert remote.stops == [f"edcraft-authoring-{job_id}"]

        # The single admission slot must be free again.
        second = await submit(env)
        assert env.service.store.get_job(second).state in {"queued", "running"}


@pytest.mark.asyncio
async def test_inline_mode_is_unchanged(tmp_path: Path) -> None:
    inline = StaticExecutor(lambda command: make_checked_report(command))
    async with nomad_env(
        tmp_path, executor="inline", inline_executor=inline
    ) as env:
        job_id = await submit(env)
        await env.service.wait_for_idle()

        job = env.service.store.get_job(job_id)
        assert job.state == "completed"
        assert job.result is not None and job.result.status == "checked"
        assert job.worker_token_hash is None
        assert job.nomad_job_id is None
        assert inline.calls == 1


def test_library_revision_matches_pyproject_pin() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pinned = data["tool"]["uv"]["sources"]["edcraft-validator"]["rev"]
    assert LIBRARY_REVISION == pinned
    assert LIBRARY_REVISION == "0392a726dac501e4f5ec4f2f3ea741d9785a3375"
