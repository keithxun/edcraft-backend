"""Focused service tests for bounded check-pilot dispatch."""

from __future__ import annotations

import asyncio
import builtins
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    CheckPilotSettings,
)
from edcraft_backend.schemas.check_pilot import (
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)
from edcraft_backend.services.check_pilot_service import (
    CheckPilotAdmissionFull,
    CheckPilotInputError,
    CheckPilotService,
    CheckPilotTimeout,
    CheckPilotUnavailable,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection

PROFILE = AuthoringProfileSettings(
    id="pilot-profile",
    label="Pilot author",
    provider="stub",
    model="stub-model",
)
DIGEST = "b" * 64


class FakeProvider:
    provider = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.closed = False

    def generation_settings(self) -> dict[str, Any]:
        return {}

    def generate(self, request: Any) -> Any:  # pragma: no cover - never called here
        raise AssertionError("the fake application does not generate")

    def close(self) -> None:
        self.closed = True


class FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.templates: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.result: Any = {"template_id": "pilot.template"}
        self.error: BaseException | None = None
        self.entered = threading.Event()
        self.release: threading.Event | None = None

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.calls.append((name, args, kwargs))
        self.entered.set()
        if self.release is not None:
            self.release.wait(5)
        if self.error is not None:
            raise self.error
        return self.result

    def author_template(
        self,
        owner_id: str,
        *,
        prompt: str,
        provider: Any,
        num_distractors: int = 3,
    ) -> Any:
        return self._record(
            "author_template",
            (owner_id,),
            {
                "prompt": prompt,
                "provider": provider,
                "num_distractors": num_distractors,
            },
        )

    def list_checks(self, owner_id: str) -> list[Any]:
        self._record("list_checks", (owner_id,), {})
        return self.checks

    def get_check(self, owner_id: str, check_id: str) -> Any:
        return self._record("get_check", (owner_id, check_id), {})

    def list_templates(self, owner_id: str) -> list[Any]:
        self._record("list_templates", (owner_id,), {})
        return self.templates

    def get_template(self, owner_id: str, template_id: str) -> Any:
        return self._record("get_template", (owner_id, template_id), {})

    def decide(
        self,
        owner_id: str,
        template_id: str,
        *,
        template_sha256: str,
        check_sha256: str,
        decision: str,
    ) -> Any:
        return self._record(
            "decide",
            (owner_id, template_id),
            {
                "template_sha256": template_sha256,
                "check_sha256": check_sha256,
                "decision": decision,
            },
        )

    def generate(self, owner_id: str, template_id: str, *, seed: int, quantity: int) -> Any:
        return self._record(
            "generate",
            (owner_id, template_id),
            {"seed": seed, "quantity": quantity},
        )


def make_service(
    tmp_path: Path,
    app: FakeApplication,
    *,
    provider: FakeProvider | None = None,
    max_concurrent_requests: int = 2,
    admission_timeout_seconds: float = 1.0,
    request_timeout_seconds: float = 5.0,
) -> tuple[CheckPilotService, list[ResolvedConnection]]:
    roles: list[ResolvedConnection] = []

    def build(role: ResolvedConnection) -> FakeProvider:
        roles.append(role)
        return provider or FakeProvider()

    service = CheckPilotService(
        authoring=AuthoringSettings(storage_dir=tmp_path / "authoring", model_profiles=[PROFILE]),
        settings=CheckPilotSettings(
            max_concurrent_requests=max_concurrent_requests,
            admission_timeout_seconds=admission_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        ),
        application_factory=lambda: app,
        provider_builder=build,
    )
    return service, roles


@pytest.mark.asyncio
async def test_author_template_forwards_owner_prompt_and_closes_provider(
    tmp_path: Path,
) -> None:
    app = FakeApplication()
    provider = FakeProvider()
    service, roles = make_service(tmp_path, app, provider=provider)
    owner = uuid4()

    result = await service.author_template(
        owner,
        PilotTemplateRequest(
            prompt="Differentiate a polynomial",
            model_profile="pilot-profile",
            num_distractors=2,
        ),
        connection=None,
    )

    assert result == app.result
    name, args, kwargs = app.calls[0]
    assert name == "author_template"
    assert args == (str(owner),)
    assert kwargs["prompt"] == "Differentiate a polynomial"
    assert kwargs["num_distractors"] == 2
    assert kwargs["provider"] is provider
    assert provider.closed is True
    assert roles == [ResolvedConnection.for_profile(PROFILE)]


@pytest.mark.asyncio
async def test_byok_connection_is_the_single_author_role(tmp_path: Path) -> None:
    app = FakeApplication()
    service, roles = make_service(tmp_path, app)
    connection = ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model="byok-model",
        api_key="not-a-real-key",
        base_url="https://api.example.com/v1",
    )

    await service.author_template(
        uuid4(),
        PilotTemplateRequest(
            prompt="Differentiate a polynomial",
            connection_id=uuid4(),
            model="byok-model",
        ),
        connection=connection,
    )

    assert roles == [connection]


@pytest.mark.asyncio
async def test_provider_is_closed_when_the_application_fails(tmp_path: Path) -> None:
    app = FakeApplication()
    app.error = RuntimeError("raw provider failure with secret sk-test")
    provider = FakeProvider()
    service, _ = make_service(tmp_path, app, provider=provider)

    with pytest.raises(CheckPilotUnavailable) as raised:
        await service.author_template(
            uuid4(),
            PilotTemplateRequest(prompt="Differentiate", model_profile="pilot-profile"),
            connection=None,
        )

    assert provider.closed is True
    assert "sk-test" not in str(raised.value)
    assert "raw provider failure" not in str(raised.value)


@pytest.mark.asyncio
async def test_unknown_profile_is_rejected_before_any_worker(tmp_path: Path) -> None:
    app = FakeApplication()
    service, _ = make_service(tmp_path, app)

    with pytest.raises(CheckPilotInputError):
        await service.author_template(
            uuid4(),
            PilotTemplateRequest(prompt="Differentiate", model_profile="missing"),
            connection=None,
        )

    assert app.calls == []


@pytest.mark.asyncio
async def test_missing_resolved_connection_is_rejected(tmp_path: Path) -> None:
    app = FakeApplication()
    service, _ = make_service(tmp_path, app)

    with pytest.raises(CheckPilotInputError):
        await service.author_template(
            uuid4(),
            PilotTemplateRequest(prompt="Differentiate", connection_id=uuid4(), model="m"),
            connection=None,
        )

    assert app.calls == []


@pytest.mark.asyncio
async def test_timeout_holds_admission_until_the_worker_returns(tmp_path: Path) -> None:
    app = FakeApplication()
    app.release = threading.Event()
    service, _ = make_service(
        tmp_path,
        app,
        max_concurrent_requests=1,
        admission_timeout_seconds=0.05,
        request_timeout_seconds=0.05,
    )
    request = PilotTemplateRequest(prompt="Differentiate", model_profile="pilot-profile")

    with pytest.raises(CheckPilotTimeout):
        await service.author_template(uuid4(), request, connection=None)

    # The Python worker cannot be interrupted, so its slot stays reserved.
    with pytest.raises(CheckPilotAdmissionFull):
        await service.list_templates(uuid4())

    app.release.set()
    assert await asyncio.to_thread(app.entered.wait, 2)
    for _ in range(100):
        try:
            listed = await service.list_templates(uuid4())
            break
        except CheckPilotAdmissionFull:
            await asyncio.sleep(0.01)
    else:
        raise AssertionError("the timed-out worker never released its slot")
    assert listed == []


@pytest.mark.asyncio
async def test_generate_enforces_authoring_quantity_limit(tmp_path: Path) -> None:
    app = FakeApplication()
    service = CheckPilotService(
        authoring=AuthoringSettings(storage_dir=tmp_path / "authoring", max_quantity=2),
        settings=CheckPilotSettings(),
        application_factory=lambda: app,
        provider_builder=lambda role: FakeProvider(),
    )

    with pytest.raises(CheckPilotInputError):
        await service.generate(uuid4(), "template-1", PilotGenerateRequest(seed=0, quantity=3))
    assert app.calls == []

    await service.generate(uuid4(), "template-1", PilotGenerateRequest(seed=0, quantity=2))
    assert app.calls[0][2] == {"seed": 0, "quantity": 2}


@pytest.mark.asyncio
async def test_reads_and_decisions_forward_owner_and_exact_digests(tmp_path: Path) -> None:
    app = FakeApplication()
    service, _ = make_service(tmp_path, app)
    owner = uuid4()

    await service.list_checks(owner)
    await service.get_check(owner, "check-1")
    await service.list_templates(owner)
    await service.get_template(owner, "template-1")
    await service.decide(
        owner,
        "template-1",
        PilotApprovalRequest(
            template_sha256=DIGEST,
            check_sha256=DIGEST,
            decision="approved",
        ),
    )

    assert [call[0] for call in app.calls] == [
        "list_checks",
        "get_check",
        "list_templates",
        "get_template",
        "decide",
    ]
    assert all(call[1][0] == str(owner) for call in app.calls)
    assert app.calls[-1][2] == {
        "template_sha256": DIGEST,
        "check_sha256": DIGEST,
        "decision": "approved",
    }


def test_default_application_import_is_lazy_and_does_not_touch_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "edcraft_validator.application.check_pilot":
            imports.append(name)
            raise AssertionError("the pilot application must not be imported eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    service = CheckPilotService(
        authoring=AuthoringSettings(storage_dir=tmp_path / "authoring"),
        settings=CheckPilotSettings(),
        provider_builder=lambda role: FakeProvider(),
    )

    assert imports == []
    assert not service.root.exists()
