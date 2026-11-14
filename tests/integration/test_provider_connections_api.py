"""BYOK provider connections and connection-backed authoring tests."""

from __future__ import annotations

import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from edcraft_backend.config import settings
from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
)
from edcraft_backend.dependencies import get_ai_authoring_service
from edcraft_backend.main import app
from edcraft_backend.models.provider_connection import ProviderConnection
from edcraft_backend.models.user import User
from edcraft_backend.services import provider_endpoint
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.provider_connection_service import (
    ProviderConnectionInvalid,
)
from edcraft_backend.services.provider_endpoint import (
    HttpxModelTransport,
    ProviderEndpoint,
    ProviderHttpResponse,
    ResolvedConnection,
)
from edcraft_backend.services.provider_secrets import ProviderSecretCipher
from tests.ai_authoring_helpers import StaticExecutor, make_checked_report
from tests.conftest import _create_test_client
from tests.factories import create_and_login_user

BASE_URL = "https://api.example.com/v1"
API_KEY = "sk-test-secret-1234"
PROFILE = AuthoringProfileSettings(
    id="test-profile",
    label="Test author",
    provider="stub",
    model="stub-model",
)


async def create_connection(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Work OpenAI",
        "kind": "openai",
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "default_model": "gpt-test",
        "models": ["gpt-test", "gpt-other"],
    }
    payload.update(overrides)
    response = await client.post("/ai-authoring/connections", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


@asynccontextmanager
async def authoring_env(
    tmp_path: Path, executor: StaticExecutor
) -> AsyncGenerator[AiAuthoringService, None]:
    service = AiAuthoringService(
        AuthoringSettings(
            storage_dir=tmp_path / "authoring",
            model_profiles=[PROFILE],
            max_active_jobs=1,
            max_queue=10,
            shutdown_timeout_seconds=1.0,
        ),
        executor=executor,
    )
    await service.startup()
    app.dependency_overrides[get_ai_authoring_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_ai_authoring_service, None)
        await service.wait_for_idle()
        await service.shutdown(0.5)


def connection_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "domain": "code",
        "prompt": "Create an arithmetic tracing question",
        "difficulty": "easy",
        "num_distractors": 3,
        "connection_id": overrides.pop("connection_id", None),
        "model": overrides.pop("model", "gpt-test"),
    }
    payload.update(overrides)
    return payload


async def test_create_lists_safe_metadata_and_encrypts_key(
    test_client: AsyncClient, db_session: AsyncSession, user: User
) -> None:
    created = await create_connection(test_client)

    assert "api_key" not in created
    assert created["has_api_key"] is True
    assert created["key_last4"] == "1234"
    assert created["default_model"] == "gpt-test"
    assert created["models"] == ["gpt-test", "gpt-other"]

    stored = (
        await db_session.execute(
            select(ProviderConnection).where(ProviderConnection.id == UUID(created["id"]))
        )
    ).scalar_one()
    assert stored.api_key_ciphertext != API_KEY
    assert API_KEY not in stored.api_key_ciphertext

    cipher = ProviderSecretCipher(_encryption_secret())
    assert cipher.decrypt(stored.api_key_ciphertext) == API_KEY

    listing = await test_client.get("/ai-authoring/connections")
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert len(body) == 1
    assert body[0]["id"] == created["id"]
    assert "api_key" not in body[0]


async def test_update_rotates_key_and_delete_forgets_connection(
    test_client: AsyncClient, db_session: AsyncSession, user: User
) -> None:
    created = await create_connection(test_client)

    response = await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"name": "Renamed", "api_key": "rotated-key-9999", "enabled": False},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["name"] == "Renamed"
    assert updated["key_last4"] == "9999"
    assert updated["enabled"] is False

    deleted = await test_client.delete(f"/ai-authoring/connections/{created['id']}")
    assert deleted.status_code == 204, deleted.text
    listing = await test_client.get("/ai-authoring/connections")
    assert listing.json() == []
    remaining = (
        await db_session.execute(
            select(ProviderConnection).where(ProviderConnection.id == UUID(created["id"]))
        )
    ).scalar_one_or_none()
    assert remaining is None


async def test_update_keeps_default_model_in_models(test_client: AsyncClient, user: User) -> None:
    created = await create_connection(test_client)

    response = await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"default_model": "gpt-extra", "models": ["gpt-test"]},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["default_model"] == "gpt-extra"
    assert updated["models"] == ["gpt-test", "gpt-extra"]


async def test_changing_base_url_clears_verification(
    test_client: AsyncClient, db_session: AsyncSession, user: User
) -> None:
    created = await create_connection(test_client)
    stored = (
        await db_session.execute(
            select(ProviderConnection).where(ProviderConnection.id == UUID(created["id"]))
        )
    ).scalar_one()
    stored.verified_at = datetime.now(UTC)
    await db_session.commit()

    response = await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"base_url": "https://api.other.example.com/v1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["verified_at"] is None


async def test_connections_are_owner_scoped(
    test_client: AsyncClient,
    db_session: AsyncSession,
    user: User,
) -> None:
    created = await create_connection(test_client)

    async with _create_test_client(db_session) as other_client:
        other = await create_and_login_user(
            other_client, db_session, email="other-owner@example.com"
        )
        assert other.id != user.id

        listing = await other_client.get("/ai-authoring/connections")
        assert listing.status_code == 200
        assert listing.json() == []

        for method, path in (
            ("patch", f"/ai-authoring/connections/{created['id']}"),
            ("delete", f"/ai-authoring/connections/{created['id']}"),
            ("post", f"/ai-authoring/connections/{created['id']}/test"),
        ):
            kwargs = {"json": {"name": "Hijack"}} if method == "patch" else {}
            response = await getattr(other_client, method)(path, **kwargs)
            assert response.status_code == 404, response.text

        still_there = await test_client.get("/ai-authoring/connections")
        assert [item["id"] for item in still_there.json()] == [created["id"]]


async def test_probe_returns_sorted_models(
    test_client: AsyncClient, user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        content = (
            b'{"data": ['
            b'{"id": "zeta-model"},'
            b'{"id": "Alpha-Model"},'
            b'{"id": "zeta-model"},'
            b'{"id": ""},'
            b'{"not_a_model": true}'
            b"]}"
        )

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(
        "edcraft_backend.services.provider_endpoint.httpx.AsyncClient",
        FakeAsyncClient,
    )

    response = await test_client.post(
        "/ai-authoring/connections/probe",
        json={"kind": "openai", "base_url": BASE_URL, "api_key": API_KEY},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "models": ["Alpha-Model", "zeta-model"],
        "default_model": "Alpha-Model",
    }
    assert captured["url"] == f"{BASE_URL}/models"
    assert captured["headers"] == {"Authorization": f"Bearer {API_KEY}"}
    assert captured["client_kwargs"]["follow_redirects"] is False


async def test_probe_maps_provider_rejection_to_safe_error(
    test_client: AsyncClient, user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        status_code = 401

        content = b"unauthorized"

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "edcraft_backend.services.provider_endpoint.httpx.AsyncClient",
        FakeAsyncClient,
    )

    response = await test_client.post(
        "/ai-authoring/connections/probe",
        json={"kind": "openai", "base_url": BASE_URL, "api_key": API_KEY},
    )
    assert response.status_code == 502, response.text
    assert response.json()["detail"] == "The provider rejected this API key."
    assert API_KEY not in response.text


async def test_probe_rejects_plain_http_remote_host(test_client: AsyncClient, user: User) -> None:
    response = await test_client.post(
        "/ai-authoring/connections/probe",
        json={
            "kind": "openai",
            "base_url": "http://api.example.com/v1",
            "api_key": API_KEY,
        },
    )
    assert response.status_code == 422, response.text
    assert "Use HTTPS" in response.json()["detail"]


UNSAFE_BASE_URLS = [
    "https://127.0.0.1/v1",
    "https://localhost/v1",
    "https://[::1]/v1",
    "https://10.0.0.5/v1",
    "https://192.168.1.5/v1",
    "https://169.254.169.254/v1",
    "https://host:abc/v1",
    "https://host:99999/v1",
    "https://[::1/v1",
    "https://exa mple.com/v1",
    "https://host/v1?x=1",
    "https://metadata.google.internal/v1",
]


@pytest.mark.parametrize("base_url", UNSAFE_BASE_URLS)
async def test_create_rejects_unsafe_provider_urls(
    test_client: AsyncClient,
    db_session: AsyncSession,
    user: User,
    base_url: str,
) -> None:
    response = await test_client.post(
        "/ai-authoring/connections",
        json={
            "name": "Unsafe provider",
            "kind": "openai",
            "base_url": base_url,
            "api_key": API_KEY,
            "default_model": "gpt-test",
            "models": ["gpt-test"],
        },
    )
    assert response.status_code == 422, response.text
    assert (await db_session.execute(select(ProviderConnection))).scalars().all() == []


async def test_dev_flag_allows_private_hosts_but_keeps_metadata_blocked(
    test_client: AsyncClient,
    user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVIDER_ALLOW_PRIVATE_HOSTS", "true")

    created = await create_connection(test_client, base_url="https://10.0.0.5/v1")
    assert created["base_url"] == "https://10.0.0.5/v1"

    response = await test_client.post(
        "/ai-authoring/connections",
        json={
            "name": "Metadata provider",
            "kind": "openai",
            "base_url": "https://169.254.169.254/v1",
            "api_key": API_KEY,
            "default_model": "gpt-test",
            "models": ["gpt-test"],
        },
    )
    assert response.status_code == 422, response.text


async def test_probe_rejects_dns_name_resolving_to_private_address(
    test_client: AsyncClient,
    user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def private_getaddrinfo(host: str, port: int, **kwargs: Any) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("10.0.0.5", port),
            )
        ]

    monkeypatch.setattr(
        "edcraft_backend.services.provider_host_guard.socket.getaddrinfo",
        private_getaddrinfo,
    )

    response = await test_client.post(
        "/ai-authoring/connections/probe",
        json={
            "kind": "openai",
            "base_url": "https://internal.example.com/v1",
            "api_key": API_KEY,
        },
    )
    assert response.status_code == 422, response.text
    assert "not allowed" in response.json()["detail"]


async def test_list_models_joins_the_models_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeTransport:
        async def get(
            self, url: str, headers: dict[str, str], timeout: float
        ) -> ProviderHttpResponse:
            captured["url"] = url
            captured["headers"] = headers
            return ProviderHttpResponse(status_code=200, body=b'{"data": []}')

    monkeypatch.setattr(
        provider_endpoint.provider_host_guard,
        "assert_public_provider_host",
        lambda _base_url: None,
    )

    endpoint = ProviderEndpoint(kind="openai", base_url="https://host/v1")

    assert await endpoint.list_models(API_KEY, transport=FakeTransport()) == []
    assert captured["url"] == "https://host/v1/models"
    assert captured["headers"] == {"Authorization": f"Bearer {API_KEY}"}


async def test_httpx_transport_maps_invalid_url_to_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidURLClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> InvalidURLClient:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str, headers: dict[str, str]) -> Any:
            raise httpx.InvalidURL("invalid URL")

    monkeypatch.setattr(provider_endpoint.httpx, "AsyncClient", InvalidURLClient)

    with pytest.raises(ProviderConnectionInvalid) as caught:
        await HttpxModelTransport().get("https://host/v1/models", {}, 20.0)
    assert caught.value.status_code == 422


async def test_capabilities_include_the_callers_connections(
    test_client: AsyncClient, user: User, tmp_path: Path
) -> None:
    created = await create_connection(test_client)
    executor = StaticExecutor(lambda command: make_checked_report(command))
    async with authoring_env(tmp_path, executor):
        response = await test_client.get("/ai-authoring/capabilities")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["connections"]] == [created["id"]]
        assert body["connections"][0]["key_last4"] == "1234"
        assert API_KEY not in response.text
        assert [item["id"] for item in body["model_profiles"]] == ["test-profile"]


async def test_capabilities_only_offer_enabled_connections(
    test_client: AsyncClient, user: User, tmp_path: Path
) -> None:
    created = await create_connection(test_client)
    await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"enabled": False},
    )
    executor = StaticExecutor(lambda command: make_checked_report(command))
    async with authoring_env(tmp_path, executor):
        response = await test_client.get("/ai-authoring/capabilities")
        assert response.status_code == 200, response.text
        assert response.json()["connections"] == []


async def test_submit_uses_connection_credentials(
    test_client: AsyncClient,
    tmp_path: Path,
    user: User,
) -> None:
    created = await create_connection(test_client)
    executor = StaticExecutor(lambda command: make_checked_report(command))

    async with authoring_env(tmp_path, executor) as service:
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(connection_id=created["id"]),
        )
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        await service.wait_for_idle()

        status = await test_client.get(f"/ai-authoring/jobs/{job_id}")
        assert status.status_code == 200, status.text
        assert status.json()["state"] == "completed"

        assert executor.author_connections == [
            ResolvedConnection.for_connection(
                connection_id=UUID(created["id"]),
                provider="openai",
                model="gpt-test",
                api_key=API_KEY,
                base_url=BASE_URL,
            )
        ]
        stored = await service.get_job(user.id, job_id)
        assert stored.connection_id == UUID(created["id"])
        assert stored.provider == "openai"
        assert stored.model == "gpt-test"
        assert stored.request.connection_id == UUID(created["id"])
        assert stored.request.model_profile is None


async def test_submit_keeps_legacy_profile_path_working(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    executor = StaticExecutor(lambda command: make_checked_report(command))

    async with authoring_env(tmp_path, executor) as service:
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(
                connection_id=None,
                model=None,
                model_profile="test-profile",
            ),
        )
        assert response.status_code == 202, response.text
        await service.wait_for_idle()

        assert executor.author_connections == [None]
        command = executor.commands[0]
        assert command.author.provider == "stub"
        assert command.author.model == "stub-model"


async def test_judges_use_the_role_defaults_when_not_selected(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    executor = StaticExecutor(lambda command: make_checked_report(command))

    async with authoring_env(tmp_path, executor) as service:
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(
                connection_id=None,
                model=None,
                model_profile="test-profile",
            ),
        )
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        await service.wait_for_idle()

        runtime = executor.runtimes[0]
        assert runtime.difficulty == runtime.author
        assert runtime.relevance == ResolvedConnection.for_judge(
            provider="soclaas", model="x-test-3"
        )
        stored = await service.get_job(user.id, job_id)
        assert stored.request.judges is None


async def test_judges_use_their_own_connection_and_model(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    created = await create_connection(test_client)
    executor = StaticExecutor(lambda command: make_checked_report(command))

    async with authoring_env(tmp_path, executor) as service:
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(
                connection_id=None,
                model=None,
                model_profile="test-profile",
                judges={
                    "difficulty": {
                        "connection_id": created["id"],
                        "model": "gpt-test",
                    },
                    "relevance": {
                        "connection_id": created["id"],
                        "model": "gpt-other",
                    },
                },
            ),
        )
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        await service.wait_for_idle()

        runtime = executor.runtimes[0]
        assert runtime.author.selection.provider == "stub"
        assert runtime.author.selection.model == "stub-model"
        assert runtime.difficulty.selection.provider == "openai"
        assert runtime.difficulty.selection.model == "gpt-test"
        assert runtime.relevance.selection.provider == "openai"
        assert runtime.relevance.selection.model == "gpt-other"
        assert runtime.difficulty == ResolvedConnection.for_connection(
            connection_id=UUID(created["id"]),
            provider="openai",
            model="gpt-test",
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        assert runtime.relevance == ResolvedConnection.for_connection(
            connection_id=UUID(created["id"]),
            provider="openai",
            model="gpt-other",
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        assert runtime.author.is_byok is False

        stored = await service.get_job(user.id, job_id)
        assert stored.request.judges is not None
        assert stored.request.judges.difficulty is not None
        assert stored.request.judges.difficulty.connection_id == UUID(created["id"])
        assert stored.request.judges.difficulty.model == "gpt-test"
        assert stored.request.judges.relevance is not None
        assert stored.request.judges.relevance.model == "gpt-other"
        assert "api_key" not in stored.request.model_dump_json()


async def test_judge_can_use_a_server_profile_while_author_uses_a_connection(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    created = await create_connection(test_client)
    executor = StaticExecutor(lambda command: make_checked_report(command))

    async with authoring_env(tmp_path, executor) as service:
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(
                connection_id=created["id"],
                judges={
                    "difficulty": {"model_profile": "test-profile"},
                },
            ),
        )
        assert response.status_code == 202, response.text
        await service.wait_for_idle()

        runtime = executor.runtimes[0]
        assert runtime.author.selection.provider == "openai"
        assert runtime.author.selection.model == "gpt-test"
        assert runtime.author.is_byok is True
        assert runtime.difficulty.selection.provider == "stub"
        assert runtime.difficulty.selection.model == "stub-model"
        assert runtime.difficulty.is_byok is False
        assert runtime.relevance == ResolvedConnection.for_judge(
            provider="soclaas", model="x-test-3"
        )


async def test_submit_rejects_unknown_disabled_or_mixed_judge_selections(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    created = await create_connection(test_client)
    executor = StaticExecutor(lambda command: make_checked_report(command))

    missing = await test_client.post(
        "/ai-authoring/jobs",
        json=connection_payload(
            connection_id=None,
            model=None,
            model_profile="test-profile",
            judges={
                "difficulty": {
                    "connection_id": "00000000-0000-0000-0000-000000000001",
                    "model": "gpt-test",
                }
            },
        ),
    )
    assert missing.status_code == 404, missing.text

    mixed = await test_client.post(
        "/ai-authoring/jobs",
        json=connection_payload(
            connection_id=None,
            model=None,
            model_profile="test-profile",
            judges={
                "relevance": {
                    "connection_id": created["id"],
                    "model": "gpt-test",
                    "model_profile": "test-profile",
                }
            },
        ),
    )
    assert mixed.status_code == 422, mixed.text

    await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"enabled": False},
    )
    async with authoring_env(tmp_path, executor):
        disabled = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(
                connection_id=None,
                model=None,
                model_profile="test-profile",
                judges={
                    "difficulty": {
                        "connection_id": created["id"],
                        "model": "gpt-test",
                    }
                },
            ),
        )
        assert disabled.status_code == 422, disabled.text
        assert disabled.json()["detail"] == "This model connection is disabled."
        assert executor.calls == 0


async def test_submit_rejects_unknown_or_disabled_connections(
    test_client: AsyncClient,
    user: User,
    tmp_path: Path,
) -> None:
    executor = StaticExecutor(lambda command: make_checked_report(command))
    missing = await test_client.post(
        "/ai-authoring/jobs",
        json=connection_payload(connection_id="00000000-0000-0000-0000-000000000001"),
    )
    assert missing.status_code == 404, missing.text

    created = await create_connection(test_client)
    disabled = await test_client.patch(
        f"/ai-authoring/connections/{created['id']}",
        json={"enabled": False},
    )
    assert disabled.status_code == 200

    async with authoring_env(tmp_path, executor):
        response = await test_client.post(
            "/ai-authoring/jobs",
            json=connection_payload(connection_id=created["id"]),
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == "This model connection is disabled."
        assert executor.calls == 0


async def test_model_selection_is_exclusive(
    test_client: AsyncClient,
    user: User,
) -> None:
    both = await test_client.post(
        "/ai-authoring/jobs",
        json=connection_payload(
            connection_id="00000000-0000-0000-0000-000000000001",
            model_profile="test-profile",
        ),
    )
    assert both.status_code == 422, both.text

    neither = await test_client.post(
        "/ai-authoring/jobs",
        json=connection_payload(connection_id=None, model=None),
    )
    assert neither.status_code == 422, neither.text


def _encryption_secret() -> str:
    configured = settings.authoring.encryption_key
    return configured.get_secret_value() if configured is not None else settings.jwt.secret
