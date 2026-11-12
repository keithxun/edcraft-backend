"""HTTP integration tests for the authenticated check-pilot routes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    CheckPilotSettings,
)
from edcraft_backend.dependencies import (
    get_check_pilot_service,
    get_current_user,
    get_provider_connection_service,
)
from edcraft_backend.main import app
from edcraft_backend.services.check_pilot_service import CheckPilotService
from edcraft_backend.services.provider_endpoint import ResolvedConnection

DIGEST = "c" * 64
CHECK_REF = {"check_id": "check-polynomial-diff", "version": "1.0.0", "sha256": DIGEST}
CHECK = {
    "check_ref": CHECK_REF,
    "title": "Polynomial differentiation",
    "description": "Checks d/dx of bounded integer polynomials.",
    "scope": {"subject": "polynomial", "max_degree": 5},
    "definition": {"recipe": "differentiate"},
    "qualification": {
        "status": "passed",
        "checking_mode": "property_based",
        "accepted_cases": 20,
        "execution_calls": 40,
        "seed": 7,
        "max_examples": 20,
        "max_execution_calls": 40,
        "timeout_seconds": 2.0,
        "counterexample": None,
        "findings": [],
        "versions": {"validator": "pilot-1"},
    },
    "provenance": {"author": "check-author"},
    "created_at": "2026-11-12T10:00:00Z",
}
TEMPLATE = {
    "template_id": "template-polynomial-diff",
    "template_sha256": DIGEST,
    "prompt": "Differentiate a polynomial",
    "status": "ready",
    "approval": "unreviewed",
    "reused_check": False,
    "check": CHECK,
    "template": {"coefficients": {"min": -4, "max": 4}},
    "qualification": CHECK["qualification"],
    "findings": [],
    "created_at": "2026-11-12T10:01:00Z",
}
BATCH = {
    "template_id": "template-polynomial-diff",
    "seed": 11,
    "questions": [
        {
            "question": "What is d/dx of 3x^2?",
            "expression": "3x^2",
            "choices": ["6x", "3x", "6x^2", "9x"],
            "correct_index": 0,
            "answer": "6x",
            "parameters": {"a": 3},
            "check_ref": CHECK_REF,
        }
    ],
    "requested_quantity": 1,
    "shortfall": 0,
    "findings": [],
}


@dataclass
class FakeUser:
    id: UUID


class FakeProvider:
    provider = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.closed = False

    def generation_settings(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self.closed = True


class FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.error: BaseException | None = None
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()

    def _result(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.calls.append((name, args, kwargs))
        self.entered.set()
        self.release.wait(5)
        if self.error is not None:
            raise self.error
        if name == "author_template":
            return TEMPLATE
        if name in {"list_templates", "list_checks"}:
            return [TEMPLATE] if name == "list_templates" else [CHECK]
        if name in {"get_template", "get_check"}:
            return TEMPLATE if name == "get_template" else CHECK
        if name == "decide":
            return {**TEMPLATE, "approval": kwargs["decision"]}
        return BATCH

    def author_template(
        self,
        owner_id: str,
        *,
        prompt: str,
        provider: Any,
        num_distractors: int = 3,
    ) -> Any:
        return self._result(
            "author_template",
            (owner_id,),
            {"prompt": prompt, "provider": provider, "num_distractors": num_distractors},
        )

    def list_templates(self, owner_id: str) -> Any:
        return self._result("list_templates", (owner_id,), {})

    def get_template(self, owner_id: str, template_id: str) -> Any:
        return self._result("get_template", (owner_id, template_id), {})

    def list_checks(self, owner_id: str) -> Any:
        return self._result("list_checks", (owner_id,), {})

    def get_check(self, owner_id: str, check_id: str) -> Any:
        return self._result("get_check", (owner_id, check_id), {})

    def decide(
        self,
        owner_id: str,
        template_id: str,
        *,
        template_sha256: str,
        check_sha256: str,
        decision: str,
    ) -> Any:
        return self._result(
            "decide",
            (owner_id, template_id),
            {
                "template_sha256": template_sha256,
                "check_sha256": check_sha256,
                "decision": decision,
            },
        )

    def generate(self, owner_id: str, template_id: str, *, seed: int, quantity: int) -> Any:
        return self._result(
            "generate",
            (owner_id, template_id),
            {"seed": seed, "quantity": quantity},
        )


class FakeConnections:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID, str | None]] = []
        self.connection: ResolvedConnection | None = None
        self.error: BaseException | None = None

    async def resolve(
        self, owner_id: UUID, connection_id: UUID, model: str | None
    ) -> ResolvedConnection:
        self.calls.append((owner_id, connection_id, model))
        if self.error is not None:
            raise self.error
        assert self.connection is not None
        return self.connection


@dataclass
class Harness:
    client: AsyncClient
    service: CheckPilotService
    app_model: FakeApplication
    connections: FakeConnections
    user: FakeUser
    providers: list[FakeProvider]


@asynccontextmanager
async def env_for(
    tmp_path: Path,
    *,
    max_quantity: int = 50,
    max_concurrent_requests: int = 2,
    admission_timeout_seconds: float = 1.0,
    request_timeout_seconds: float = 5.0,
) -> AsyncGenerator[Harness, None]:
    application = FakeApplication()
    providers: list[FakeProvider] = []

    def build_provider(role: ResolvedConnection) -> FakeProvider:
        provider = FakeProvider()
        providers.append(provider)
        return provider

    service = CheckPilotService(
        authoring=AuthoringSettings(
            storage_dir=tmp_path / "authoring",
            max_quantity=max_quantity,
            model_profiles=[
                AuthoringProfileSettings(
                    id="server-profile",
                    label="Server profile",
                    provider="stub",
                    model="stub-model",
                )
            ],
        ),
        settings=CheckPilotSettings(
            max_concurrent_requests=max_concurrent_requests,
            admission_timeout_seconds=admission_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        ),
        application_factory=lambda: application,
        provider_builder=build_provider,
    )
    user = FakeUser(uuid4())
    connections = FakeConnections()

    def current_user() -> FakeUser:
        return user

    def pilot_service() -> CheckPilotService:
        return service

    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_check_pilot_service] = pilot_service
    app.dependency_overrides[get_provider_connection_service] = lambda: connections
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(
                client=client,
                service=service,
                app_model=application,
                connections=connections,
                user=user,
                providers=providers,
            )
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncGenerator[Harness, None]:
    async with env_for(tmp_path) as harness:
        yield harness


def template_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": "Differentiate a polynomial",
        "model_profile": "server-profile",
        "num_distractors": 3,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_all_pilot_routes_require_sign_in(env: Harness) -> None:
    app.dependency_overrides.pop(get_current_user, None)

    for method, path, body in (
        ("post", "/ai-authoring/check-pilot/templates", template_payload()),
        ("get", "/ai-authoring/check-pilot/templates", None),
        ("get", "/ai-authoring/check-pilot/templates/t1", None),
        ("get", "/ai-authoring/check-pilot/checks", None),
        ("get", "/ai-authoring/check-pilot/checks/c1", None),
        (
            "post",
            "/ai-authoring/check-pilot/templates/t1/approval",
            {"template_sha256": DIGEST, "check_sha256": DIGEST, "decision": "approved"},
        ),
        (
            "post",
            "/ai-authoring/check-pilot/templates/t1/generate",
            {"seed": 0},
        ),
    ):
        if body is None:
            response = await getattr(env.client, method)(path)
        else:
            response = await getattr(env.client, method)(path, json=body)
        assert response.status_code == 401, (path, response.text)


@pytest.mark.asyncio
async def test_request_validation_rejects_incomplete_or_coerced_input(env: Harness) -> None:
    for payload in (
        {"prompt": "Differentiate"},
        {**template_payload(), "model_profile": None},
        {
            **template_payload(),
            "connection_id": str(uuid4()),
            "model": "m",
        },
        template_payload(prompt="   "),
        template_payload(num_distractors=0),
        template_payload(num_distractors=11),
        template_payload(num_distractors=True),
        template_payload(unknown=True),
    ):
        response = await env.client.post("/ai-authoring/check-pilot/templates", json=payload)
        assert response.status_code == 422, (payload, response.text)

    for payload in (
        {"template_sha256": DIGEST, "check_sha256": "bad", "decision": "approved"},
        {"template_sha256": DIGEST, "check_sha256": DIGEST, "decision": "maybe"},
        {"template_sha256": DIGEST},
    ):
        response = await env.client.post(
            "/ai-authoring/check-pilot/templates/t1/approval", json=payload
        )
        assert response.status_code == 422, (payload, response.text)

    for payload in (
        {"seed": -1},
        {"seed": True},
        {"seed": 0, "quantity": 0},
        {"seed": 0, "quantity": 51},
        {"seed": 2_147_483_647, "quantity": 2},
        {"seed": 0, "extra": 1},
    ):
        response = await env.client.post(
            "/ai-authoring/check-pilot/templates/t1/generate", json=payload
        )
        assert response.status_code == 422, (payload, response.text)

    assert env.app_model.calls == []


@pytest.mark.asyncio
async def test_profile_authoring_returns_the_ai_record_and_closes_provider(
    env: Harness,
) -> None:
    response = await env.client.post(
        "/ai-authoring/check-pilot/templates", json=template_payload(num_distractors=2)
    )
    assert response.status_code == 200, response.text
    assert response.json() == TEMPLATE

    name, args, kwargs = env.app_model.calls[0]
    assert name == "author_template"
    assert args == (str(env.user.id),)
    assert kwargs["prompt"] == "Differentiate a polynomial"
    assert kwargs["num_distractors"] == 2
    assert env.providers and env.providers[0].closed is True
    assert env.connections.calls == []


@pytest.mark.asyncio
async def test_byok_authoring_resolves_the_existing_connection_once(env: Harness) -> None:
    connection_id = uuid4()
    env.connections.connection = ResolvedConnection.for_connection(
        connection_id=connection_id,
        provider="openai",
        model="byok-model",
        api_key="not-a-real-key",
        base_url="https://api.example.com/v1",
    )

    response = await env.client.post(
        "/ai-authoring/check-pilot/templates",
        json={
            "prompt": "Differentiate a polynomial",
            "connection_id": str(connection_id),
            "model": "byok-model",
        },
    )

    assert response.status_code == 200, response.text
    assert env.connections.calls == [(env.user.id, connection_id, "byok-model")]
    assert env.providers and env.providers[0].closed is True


@pytest.mark.asyncio
async def test_list_get_approval_and_generation_use_the_same_owner(env: Harness) -> None:
    templates = await env.client.get("/ai-authoring/check-pilot/templates")
    assert templates.status_code == 200
    assert templates.json() == [TEMPLATE]

    template = await env.client.get("/ai-authoring/check-pilot/templates/template-polynomial-diff")
    assert template.status_code == 200
    assert template.json() == TEMPLATE

    checks = await env.client.get("/ai-authoring/check-pilot/checks")
    assert checks.status_code == 200
    assert checks.json() == [CHECK]

    check = await env.client.get("/ai-authoring/check-pilot/checks/check-polynomial-diff")
    assert check.status_code == 200
    assert check.json() == CHECK

    approval = await env.client.post(
        "/ai-authoring/check-pilot/templates/template-polynomial-diff/approval",
        json={"template_sha256": DIGEST, "check_sha256": DIGEST, "decision": "approved"},
    )
    assert approval.status_code == 200, approval.text
    assert approval.json()["approval"] == "approved"

    generated = await env.client.post(
        "/ai-authoring/check-pilot/templates/template-polynomial-diff/generate",
        json={"seed": 11, "quantity": 1},
    )
    assert generated.status_code == 200, generated.text
    assert generated.json() == BATCH

    owner = str(env.user.id)
    assert all(call[1][0] == owner for call in env.app_model.calls)
    assert env.app_model.calls[-1][1] == (owner, "template-polynomial-diff")
    assert env.app_model.calls[-1][2] == {"seed": 11, "quantity": 1}


@pytest.mark.asyncio
async def test_generation_quantity_is_bounded_by_the_authoring_setting(
    tmp_path: Path,
) -> None:
    async with env_for(tmp_path, max_quantity=2) as env:
        rejected = await env.client.post(
            "/ai-authoring/check-pilot/templates/template-polynomial-diff/generate",
            json={"seed": 0, "quantity": 3},
        )
        assert rejected.status_code == 422
        assert env.app_model.calls == []

        allowed = await env.client.post(
            "/ai-authoring/check-pilot/templates/template-polynomial-diff/generate",
            json={"seed": 0, "quantity": 2},
        )
        assert allowed.status_code == 200


@pytest.mark.asyncio
async def test_failures_are_safe_and_never_return_a_passing_record(env: Harness) -> None:
    env.app_model.error = RuntimeError("provider exploded with sk-secret")
    response = await env.client.post("/ai-authoring/check-pilot/templates", json=template_payload())
    assert response.status_code == 503
    assert "sk-secret" not in response.text
    assert "provider exploded" not in response.text
    assert env.providers[0].closed is True


@pytest.mark.asyncio
async def test_timeout_and_admission_are_bounded_and_visible(tmp_path: Path) -> None:
    async with env_for(
        tmp_path,
        max_concurrent_requests=1,
        admission_timeout_seconds=0.05,
        request_timeout_seconds=0.5,
    ) as env:
        env.app_model.release = threading.Event()
        first = asyncio.create_task(
            env.client.post("/ai-authoring/check-pilot/templates", json=template_payload())
        )
        assert await asyncio.to_thread(env.app_model.entered.wait, 2)

        busy = await env.client.post("/ai-authoring/check-pilot/templates", json=template_payload())
        assert busy.status_code == 429

        timed_out = await first
        assert timed_out.status_code == 504
        assert "timed out" in timed_out.json()["detail"].lower()

        env.app_model.release.set()
        for _ in range(100):
            listed = await env.client.get("/ai-authoring/check-pilot/templates")
            if listed.status_code == 200:
                break
            assert listed.status_code == 429
            await asyncio.sleep(0.01)
        assert listed.json() == [TEMPLATE]
