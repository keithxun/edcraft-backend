"""End-to-end check pilot against the real AI application, when published.

These tests are skipped while the pinned validator wheel predates
``CheckPilotApplication``. They exercise the backend's default factory,
provider lifecycle and owner scoping with the AI-owned storage and records,
using a scripted structured provider so no model is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

check_pilot = pytest.importorskip("edcraft_validator.application.check_pilot")
if not hasattr(check_pilot, "CheckPilotApplication"):  # pragma: no cover - pre-pin
    pytest.skip(
        "the installed validator revision does not publish CheckPilotApplication",
        allow_module_level=True,
    )

from edcraft_validator.application.check_pilot import (  # noqa: E402
    CheckRecipeDefinition,
)
from edcraft_validator.application.check_pilot.models import (  # noqa: E402
    CheckAuthorProposal,
    CoefficientDomain,
    PolynomialFamily,
    RecipeArgument,
    RecipeStep,
    TemplateAuthorProposal,
)

from edcraft_backend.config.settings import (  # noqa: E402
    AuthoringProfileSettings,
    AuthoringSettings,
    CheckPilotSettings,
)
from edcraft_backend.dependencies import (  # noqa: E402
    get_check_pilot_service,
    get_current_user,
    get_provider_connection_service,
)
from edcraft_backend.main import app  # noqa: E402
from edcraft_backend.schemas.check_pilot import (  # noqa: E402
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)
from edcraft_backend.services.check_pilot_service import (  # noqa: E402
    CheckPilotConflict,
    CheckPilotNotFound,
    CheckPilotService,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection  # noqa: E402

PROFILE = AuthoringProfileSettings(
    id="pilot-profile",
    label="Pilot author",
    provider="stub",
    model="stub-structured",
)


def _input(name: str) -> RecipeArgument:
    return RecipeArgument(source="input", name=name)


def _step(name: str) -> RecipeArgument:
    return RecipeArgument(source="step", name=name)


def _check_proposal() -> CheckAuthorProposal:
    return CheckAuthorProposal(
        title="Differentiate a bounded polynomial",
        description=(
            "Checks a candidate derivative by comparing it with an independently "
            "derived polynomial over all coefficients."
        ),
        recipe=CheckRecipeDefinition(
            steps=[
                RecipeStep(
                    id="d1",
                    operation="poly.differentiate",
                    version="1",
                    arguments={"polynomial": _input("problem")},
                ),
                RecipeStep(
                    id="out",
                    operation="poly.equivalent",
                    version="1",
                    arguments={
                        "expected": _step("d1"),
                        "actual": _input("candidate"),
                    },
                ),
            ],
            output="out",
        ),
    )


def _template_proposal() -> TemplateAuthorProposal:
    return TemplateAuthorProposal(
        family=PolynomialFamily(
            degree=2,
            coefficients=[
                CoefficientDomain(exponent=0, values=[-3, -1, 0, 2]),
                CoefficientDomain(exponent=1, values=[-2, 0, 1, 2]),
                CoefficientDomain(exponent=2, values=[1, 2, 3]),
            ],
            num_distractors=3,
        )
    )


class ScriptedProvider:
    """Answer the pilot's two structured roles without any network call."""

    provider = "stub"
    model = "stub-structured"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted", "reasoning_effort": "none"}

    def generate(self, request: Any) -> Any:
        self.calls.append(request.prompt_version)
        if request.prompt_version.startswith("check-author"):
            return _check_proposal()
        return _template_proposal()

    def close(self) -> None:
        self.closed = True


def _service(
    tmp_path: Path,
    *,
    providers: list[ScriptedProvider] | None = None,
) -> CheckPilotService:
    def build(_: ResolvedConnection) -> ScriptedProvider:
        provider = ScriptedProvider()
        if providers is not None:
            providers.append(provider)
        return provider

    return CheckPilotService(
        authoring=AuthoringSettings(
            storage_dir=tmp_path / "authoring", model_profiles=[PROFILE]
        ),
        settings=CheckPilotSettings(
            max_concurrent_requests=1,
            admission_timeout_seconds=1.0,
            request_timeout_seconds=120.0,
        ),
        provider_builder=build,
    )


def _author(service: CheckPilotService, owner: UUID) -> Any:
    import asyncio

    return asyncio.run(
        service.author_template(
            owner,
            PilotTemplateRequest(
                prompt="Author a polynomial derivative check",
                model_profile=PROFILE.id,
                num_distractors=3,
            ),
            connection=None,
        )
    )


def test_default_factory_authors_approves_generates_and_survives_restart(
    tmp_path: Path,
) -> None:
    import asyncio

    owner = uuid4()
    providers: list[ScriptedProvider] = []
    service = _service(tmp_path, providers=providers)

    # The default factory imports the AI application lazily and creates no
    # durable storage until the first pilot call.
    assert not service.root.exists()

    record = _author(service, owner)

    assert providers and all(provider.closed for provider in providers)
    assert providers[0].calls[:2] == ["check-author/1", "template-author/1"]
    assert record.status == "ready"
    assert record.check is not None
    assert record.check_qualification is not None
    assert record.check_qualification.status == "passed"
    assert service.root.is_dir()

    # Records stay JSON-serializable end to end, including AI-added fields.
    payload = record.model_dump(mode="json")
    assert payload["template_id"] == record.template_id
    assert payload["check"]["check_ref"]["check_id"]

    checks = asyncio.run(service.list_checks(owner))
    templates = asyncio.run(service.list_templates(owner))
    assert [check.check_ref for check in checks] == [record.check.check_ref]
    assert [template.template_id for template in templates] == [record.template_id]

    approved = asyncio.run(
        service.decide(
            owner,
            record.template_id,
            PilotApprovalRequest(
                template_sha256=record.template_sha256,
                check_sha256=record.check.check_ref.sha256,
                decision="approved",
            ),
        )
    )
    assert approved.approval == "approved"

    batch = asyncio.run(
        service.generate(
            owner,
            record.template_id,
            PilotGenerateRequest(seed=42, quantity=3),
        )
    )
    assert batch.requested_quantity == 3
    assert len(batch.questions) == 3
    assert len({question.expression for question in batch.questions}) == 3
    assert len(batch.model_dump(mode="json")["questions"]) == 3

    # A fresh service over the same directory reads the same private records.
    restarted = _service(tmp_path)
    assert asyncio.run(restarted.get_check(owner, record.check.check_ref.check_id)) == (
        record.check
    )
    assert asyncio.run(restarted.get_template(owner, record.template_id)).approval == (
        "approved"
    )

    other_owner = uuid4()
    with pytest.raises(CheckPilotNotFound):
        asyncio.run(restarted.get_template(other_owner, record.template_id))
    with pytest.raises(CheckPilotNotFound):
        asyncio.run(
            restarted.get_check(other_owner, record.check.check_ref.check_id)
        )
    assert asyncio.run(restarted.list_checks(other_owner)) == []


def test_real_ai_conflicts_map_to_backend_conflict(tmp_path: Path) -> None:
    import asyncio

    owner = uuid4()
    service = _service(tmp_path)
    record = _author(service, owner)
    assert record.check is not None

    with pytest.raises(CheckPilotConflict):
        asyncio.run(
            service.decide(
                owner,
                record.template_id,
                PilotApprovalRequest(
                    template_sha256=record.template_sha256,
                    check_sha256="0" * 64,
                    decision="approved",
                ),
            )
        )


@dataclass
class _User:
    id: UUID


def _install(service: CheckPilotService, user: _User) -> None:
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_check_pilot_service] = lambda: service
    app.dependency_overrides[get_provider_connection_service] = lambda: object()


async def test_http_flow_authors_approves_generates_reuses_and_restarts(
    tmp_path: Path,
) -> None:
    owner = _User(uuid4())
    providers: list[ScriptedProvider] = []
    service = _service(tmp_path, providers=providers)
    _install(service, owner)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    base = "/ai-authoring/check-pilot"

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get(f"{base}/checks")).json() == []
            assert (await client.get(f"{base}/templates")).json() == []

            created = await client.post(
                f"{base}/templates",
                json={
                    "prompt": "Author a polynomial derivative check",
                    "model_profile": PROFILE.id,
                    "num_distractors": 3,
                },
            )
            assert created.status_code == 200
            record = created.json()
            assert record["status"] == "ready"
            assert record["approval"] == "unreviewed"
            assert record["reused_check"] is False
            assert record["check"]["qualification"]["status"] == "passed"
            assert record["check_qualification"]["status"] == "passed"
            assert [len(provider.calls) for provider in providers] == [2]
            assert all(provider.closed for provider in providers)

            template_id = record["template_id"]
            check_id = record["check"]["check_ref"]["check_id"]

            listed_templates = await client.get(f"{base}/templates")
            assert [item["template_id"] for item in listed_templates.json()] == [
                template_id
            ]
            fetched_template = await client.get(f"{base}/templates/{template_id}")
            assert fetched_template.json()["check"]["check_ref"]["check_id"] == check_id
            listed_checks = await client.get(f"{base}/checks")
            assert [item["check_ref"]["check_id"] for item in listed_checks.json()] == [
                check_id
            ]
            fetched_check = await client.get(f"{base}/checks/{check_id}")
            assert fetched_check.json()["check_ref"]["check_id"] == check_id
            assert (await client.get(f"{base}/checks/chk_{'a' * 24}")).status_code == 404

            stale = await client.post(
                f"{base}/templates/{template_id}/approval",
                json={
                    "template_sha256": "0" * 64,
                    "check_sha256": record["check"]["check_ref"]["sha256"],
                    "decision": "approved",
                },
            )
            assert stale.status_code == 409

            approved = await client.post(
                f"{base}/templates/{template_id}/approval",
                json={
                    "template_sha256": record["template_sha256"],
                    "check_sha256": record["check"]["check_ref"]["sha256"],
                    "decision": "approved",
                },
            )
            assert approved.status_code == 200
            assert approved.json()["approval"] == "approved"

            generated = await client.post(
                f"{base}/templates/{template_id}/generate",
                json={"seed": 42, "quantity": 3},
            )
            assert generated.status_code == 200
            batch = generated.json()
            assert len(batch["questions"]) == 3

            reused = await client.post(
                f"{base}/templates",
                json={
                    "prompt": "Author a second polynomial derivative check",
                    "model_profile": PROFILE.id,
                },
            )
            assert reused.status_code == 200
            second = reused.json()
            assert second["reused_check"] is True
            assert second["check"]["check_ref"] == record["check"]["check_ref"]
            # Reuse makes no check-author model call: one template call only.
            assert len(providers[-1].calls) == 1

            restarted = _service(tmp_path)
            _install(restarted, owner)
            persisted = await client.get(f"{base}/templates/{template_id}")
            assert persisted.json()["approval"] == "approved"
            regenerated = await client.post(
                f"{base}/templates/{template_id}/generate",
                json={"seed": 42, "quantity": 3},
            )
            assert regenerated.json() == batch

            _install(restarted, _User(uuid4()))
            assert (await client.get(f"{base}/templates")).json() == []
            assert (await client.get(f"{base}/templates/{template_id}")).status_code == 404
            assert (await client.get(f"{base}/checks/{check_id}")).status_code == 404
    finally:
        app.dependency_overrides.clear()


__all__ = ["ScriptedProvider"]
