"""End-to-end check pilot against the real pinned AI application.

These tests exercise the backend's default factory, provider lifecycle and
owner scoping with the AI-owned storage and records, using scripted structured
providers so no model is called. The pinned validator revision must publish
``CheckPilotApplication``; a missing pilot is a collection failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
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
from edcraft_backend.schemas.check_pilot import (
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)
from edcraft_backend.services.check_pilot_service import (
    CheckPilotConflict,
    CheckPilotNotFound,
    CheckPilotService,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection
from tests.check_catalogue_fixtures import (
    CHECK_PLAN,
    CHECK_RECIPE,
    DIFFERENTIATE,
    POLYNOMIAL_FAMILY,
    ProviderScripts,
    differentiation_script,
    plan,
    solve_script,
    template,
)

PROFILE = AuthoringProfileSettings(
    id="pilot-profile",
    label="Pilot author",
    provider="stub",
    model="stub-structured",
)


def _service(tmp_path: Path, *, scripts: ProviderScripts | None = None) -> CheckPilotService:
    scripts = scripts or ProviderScripts([])

    def build(_: ResolvedConnection) -> Any:
        return scripts.build(_)

    return CheckPilotService(
        authoring=AuthoringSettings(storage_dir=tmp_path / "authoring", model_profiles=[PROFILE]),
        settings=CheckPilotSettings(
            max_concurrent_requests=1,
            admission_timeout_seconds=1.0,
            request_timeout_seconds=120.0,
        ),
        provider_builder=build,
    )


def _author(service: CheckPilotService, owner: UUID) -> Any:
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
    owner = uuid4()
    scripts = ProviderScripts([differentiation_script()])
    service = _service(tmp_path, scripts=scripts)

    # The default factory imports the AI application lazily and creates no
    # durable storage until the first pilot call.
    assert not service.root.exists()

    record = _author(service, owner)

    assert scripts.providers and all(provider.closed for provider in scripts.providers)
    provider = scripts.providers[0]
    # The plan, check author and template author are distinct structured roles.
    assert provider.calls == [CHECK_PLAN, CHECK_RECIPE, POLYNOMIAL_FAMILY]
    assert record.status == "ready"
    assert record.check is not None
    assert record.check_qualification is not None
    assert record.check_qualification.status == "passed"
    assert record.selection is not None
    assert record.selection.contract_id == DIFFERENTIATE
    assert record.selection.source == "authored"
    assert record.selection.check_ref == record.check.check_ref
    assert service.root.is_dir()

    # Records stay JSON-serializable end to end, including AI-added fields.
    payload = record.model_dump(mode="json")
    assert payload["template_id"] == record.template_id
    assert payload["check"]["check_ref"]["check_id"]
    assert payload["selection"]["contract_id"] == DIFFERENTIATE

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
    assert asyncio.run(restarted.get_template(owner, record.template_id)).approval == ("approved")

    other_owner = uuid4()
    with pytest.raises(CheckPilotNotFound):
        asyncio.run(restarted.get_template(other_owner, record.template_id))
    with pytest.raises(CheckPilotNotFound):
        asyncio.run(restarted.get_check(other_owner, record.check.check_ref.check_id))
    assert asyncio.run(restarted.list_checks(other_owner)) == []


def test_real_ai_conflicts_map_to_backend_conflict(tmp_path: Path) -> None:
    owner = uuid4()
    service = _service(tmp_path, scripts=ProviderScripts([differentiation_script()]))
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
    scripts = ProviderScripts(
        [
            differentiation_script(),
            {
                CHECK_PLAN: [plan(DIFFERENTIATE, degree=2)],
                POLYNOMIAL_FAMILY: [template(DIFFERENTIATE, degree=2)],
            },
        ]
    )
    service = _service(tmp_path, scripts=scripts)
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
            assert record["selection"]["contract_id"] == DIFFERENTIATE
            assert record["selection"]["source"] == "authored"
            assert (
                record["selection"]["check_ref"]["check_id"]
                == record["check"]["check_ref"]["check_id"]
            )
            first_provider = scripts.providers[0]
            assert first_provider.calls == [
                CHECK_PLAN,
                CHECK_RECIPE,
                POLYNOMIAL_FAMILY,
            ]
            assert first_provider.closed is True

            template_id = record["template_id"]
            check_id = record["check"]["check_ref"]["check_id"]
            check_ref = record["check"]["check_ref"]

            listed_templates = await client.get(f"{base}/templates")
            assert [item["template_id"] for item in listed_templates.json()] == [template_id]
            fetched_template = await client.get(f"{base}/templates/{template_id}")
            assert fetched_template.json()["check"]["check_ref"]["check_id"] == check_id
            assert fetched_template.json()["selection"] == {
                "contract_id": DIFFERENTIATE,
                "reason": "The selected contract checks the complete answer to the "
                "declared mathematical task.",
                "source": "authored",
                "check_ref": check_ref,
            }
            listed_checks = await client.get(f"{base}/checks")
            assert [item["check_ref"]["check_id"] for item in listed_checks.json()] == [check_id]
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

            # A second template can select the exact saved check; the plan is
            # still required, but no check-author call runs.
            reuse_script = {
                CHECK_PLAN: [plan(DIFFERENTIATE, degree=2, saved_check_ref=check_ref)],
                POLYNOMIAL_FAMILY: [template(DIFFERENTIATE, degree=2)],
            }
            scripts.scripts.append(reuse_script)
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
            assert second["selection"]["source"] == "saved"
            assert second["selection"]["check_ref"] == check_ref
            reuse_provider = scripts.providers[-1]
            assert reuse_provider.calls == [CHECK_PLAN, POLYNOMIAL_FAMILY]

            restarted = _service(tmp_path)
            _install(restarted, owner)
            persisted = await client.get(f"{base}/templates/{template_id}")
            assert persisted.json()["approval"] == "approved"
            assert persisted.json()["selection"]["source"] == "authored"
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


__all__ = ["ProviderScripts", "solve_script"]
