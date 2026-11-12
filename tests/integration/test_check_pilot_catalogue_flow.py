"""Real-application HTTP coverage for catalogue selection and check authoring.

The backend only forwards prompts and returns AI-owned records. These tests
drive the genuine ``CheckPilotApplication`` flow with scripted structured
proposals: a planner selects a trusted contract, the application authors or
reuses an owner-private check, and approval/generation stay owner-scoped.

No external model is called and no backend mathematics is implemented.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

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
from edcraft_backend.services.check_pilot_service import CheckPilotService
from tests.check_catalogue_fixtures import (
    CHECK_PLAN,
    CHECK_RECIPE,
    DIFFERENTIATE,
    EQUATION,
    INEQUALITY,
    POLYNOMIAL_FAMILY,
    ProviderScripts,
    check,
    invalid_solve_script,
    plan,
    solve_script,
    template,
    unsupported_plan,
)

BASE = "/ai-authoring/check-pilot"
PROFILE = AuthoringProfileSettings(
    id="catalogue-profile",
    label="Catalogue author",
    provider="stub",
    model="stub-catalogue",
)
OWNER = "11111111-1111-1111-1111-111111111111"


@dataclass
class _User:
    id: UUID


@dataclass
class Harness:
    client: AsyncClient
    service: CheckPilotService
    scripts: ProviderScripts


def _service(tmp_path: Path, scripts: ProviderScripts) -> CheckPilotService:
    return CheckPilotService(
        authoring=AuthoringSettings(storage_dir=tmp_path / "authoring", model_profiles=[PROFILE]),
        settings=CheckPilotSettings(
            max_concurrent_requests=1,
            admission_timeout_seconds=1.0,
            request_timeout_seconds=120.0,
        ),
        provider_builder=scripts.build,
    )


@asynccontextmanager
async def _harness(
    tmp_path: Path,
    scripts: ProviderScripts,
    *,
    owner: str = OWNER,
) -> AsyncGenerator[Harness, None]:
    service = _service(tmp_path, scripts)
    user = _User(UUID(owner))
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_check_pilot_service] = lambda: service
    app.dependency_overrides[get_provider_connection_service] = lambda: object()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(client=client, service=service, scripts=scripts)
    finally:
        app.dependency_overrides.clear()


async def _author(client: AsyncClient, prompt: str) -> Any:
    return await client.post(
        f"{BASE}/templates",
        json={"prompt": prompt, "model_profile": PROFILE.id, "num_distractors": 3},
    )


async def _approve(client: AsyncClient, record: dict[str, Any]) -> Any:
    return await client.post(
        f"{BASE}/templates/{record['template_id']}/approval",
        json={
            "template_sha256": record["template_sha256"],
            "check_sha256": record["check"]["check_ref"]["sha256"],
            "decision": "approved",
        },
    )


async def _generate(client: AsyncClient, template_id: str, *, seed: int, quantity: int) -> Any:
    return await client.post(
        f"{BASE}/templates/{template_id}/generate",
        json={"seed": seed, "quantity": quantity},
    )


def _assert_fail_closed(record: dict[str, Any]) -> None:
    assert record["status"] in {"needs_review", "error"}, record
    assert record["status"] != "ready"
    assert record["approval"] == "unreviewed"
    assert record["findings"], record
    assert record["check"] is None or record["status"] != "ready"


@pytest.mark.asyncio
async def test_equation_and_inequality_compile_approve_generate_and_reuse(
    tmp_path: Path,
) -> None:
    equation = solve_script(EQUATION, relation="eq", degree=2)
    inequality = solve_script(INEQUALITY, relation="ge", degree=2)
    harness_scripts = [
        equation,
        inequality,
        solve_script(EQUATION, relation="eq", degree=2),
        solve_script(INEQUALITY, relation="ge", degree=2),
    ]
    scripts = ProviderScripts(harness_scripts[:2])
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Solve a bounded quadratic equation")
        assert created.status_code == 200, created.text
        equation_record = created.json()
        assert equation_record["status"] == "ready", equation_record["findings"]
        assert equation_record["reused_check"] is False
        assert equation_record["selection"]["contract_id"] == EQUATION
        assert equation_record["selection"]["source"] == "authored"
        assert equation_record["selection"]["check_ref"] is not None
        assert equation_record["check"]["scope"]["contract_id"] == EQUATION
        assert equation_record["check"]["scope"]["answer_type"] == "solution_set"
        assert equation_record["check"]["definition"]["operation_set"] == "polynomial-solve-real/1"
        assert equation_record["check_qualification"]["status"] == "passed"
        assert scripts.providers[0].calls == [
            CHECK_PLAN,
            CHECK_RECIPE,
            POLYNOMIAL_FAMILY,
        ]

        created = await _author(harness.client, "Solve a bounded quadratic inequality")
        assert created.status_code == 200, created.text
        inequality_record = created.json()
        assert inequality_record["status"] == "ready", inequality_record["findings"]
        assert inequality_record["selection"]["contract_id"] == INEQUALITY
        assert inequality_record["check"]["scope"]["contract_id"] == INEQUALITY
        assert inequality_record["check"]["check_ref"] != equation_record["check"]["check_ref"]
        assert scripts.providers[1].calls == [
            CHECK_PLAN,
            CHECK_RECIPE,
            POLYNOMIAL_FAMILY,
        ]

        listed = await harness.client.get(f"{BASE}/checks")
        assert len(listed.json()) == 2

        approved = await _approve(harness.client, equation_record)
        assert approved.status_code == 200, approved.text
        batch = await _generate(harness.client, equation_record["template_id"], seed=13, quantity=4)
        assert batch.status_code == 200, batch.text
        questions = batch.json()["questions"]
        assert len(questions) == 4
        assert len({question["expression"] for question in questions}) == 4
        for question in questions:
            assert question["check_ref"] == equation_record["check"]["check_ref"]
            assert question["answer"] in question["choices"]
            assert question["choices"][question["correct_index"]] == question["answer"]
        assert len({question["answer"] for question in questions}) > 1

        approved = await _approve(harness.client, inequality_record)
        assert approved.status_code == 200, approved.text
        batch = await _generate(
            harness.client, inequality_record["template_id"], seed=5, quantity=3
        )
        assert batch.status_code == 200, batch.text
        for question in batch.json()["questions"]:
            assert question["check_ref"] == inequality_record["check"]["check_ref"]
            assert question["answer"] in question["choices"]

        for reference in (
            equation_record["check"]["check_ref"],
            inequality_record["check"]["check_ref"],
        ):
            contract = (
                EQUATION if reference == equation_record["check"]["check_ref"] else INEQUALITY
            )
            scripts.scripts.append(
                solve_script(contract, relation="eq" if contract == EQUATION else "ge")
                | {
                    CHECK_PLAN: [
                        plan(
                            contract,
                            relation="eq" if contract == EQUATION else "ge",
                            degree=2,
                            saved_check_ref=reference,
                        )
                    ]
                }
            )
            reused = await _author(harness.client, "Reuse the qualified check")
            assert reused.status_code == 200, reused.text
            record = reused.json()
            assert record["status"] == "ready", record["findings"]
            assert record["reused_check"] is True
            assert record["approval"] == "unreviewed"
            assert record["selection"]["source"] == "saved"
            assert record["selection"]["check_ref"] == reference
            assert record["check"]["check_ref"] == reference
            assert scripts.providers[-1].calls == [
                CHECK_PLAN,
                POLYNOMIAL_FAMILY,
            ]

        # Restart reads the same private catalogue and approved templates.
        restarted = _service(tmp_path, ProviderScripts([]))
        app.dependency_overrides[get_check_pilot_service] = lambda: restarted
        assert len((await harness.client.get(f"{BASE}/checks")).json()) == 2
        assert (
            await harness.client.get(f"{BASE}/templates/{equation_record['template_id']}")
        ).json()["approval"] == "approved"
        assert (
            await _generate(
                harness.client,
                equation_record["template_id"],
                seed=13,
                quantity=4,
            )
        ).status_code == 200


@pytest.mark.asyncio
async def test_unknown_contract_plan_fails_closed_without_authoring(
    tmp_path: Path,
) -> None:
    scripts = ProviderScripts(
        [
            invalid_solve_script(
                plan("polynomial.arbitrary/9", relation="eq"),
                include_check=False,
            )
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Integrate a general function")
        assert created.status_code == 200, created.text
        record = created.json()
        _assert_fail_closed(record)
        assert any("contract" in finding.lower() for finding in record["findings"])
        assert (await harness.client.get(f"{BASE}/checks")).json() == []
        assert scripts.providers[0].calls == [CHECK_PLAN]

        rejected = await harness.client.post(
            f"{BASE}/templates/{record['template_id']}/approval",
            json={
                "template_sha256": record["template_sha256"],
                "check_sha256": "0" * 64,
                "decision": "approved",
            },
        )
        assert rejected.status_code in {409, 422}


@pytest.mark.asyncio
async def test_missing_capability_reports_development_needed(tmp_path: Path) -> None:
    scripts = ProviderScripts(
        [
            {
                CHECK_PLAN: [
                    unsupported_plan("LU decomposition requires a new developer-owned contract.")
                ]
            }
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Solve a linear system with LU")
        assert created.status_code == 200, created.text
        record = created.json()
        _assert_fail_closed(record)
        assert any(
            "LU decomposition" in finding or "development" in finding.lower()
            for finding in record["findings"]
        )
        assert (await harness.client.get(f"{BASE}/checks")).json() == []
        assert scripts.providers[0].calls == [CHECK_PLAN]


@pytest.mark.asyncio
async def test_incompatible_contract_goal_fails_closed(tmp_path: Path) -> None:
    # The equation contract only covers equality, so a model cannot borrow it
    # for an inequality goal.
    scripts = ProviderScripts(
        [
            invalid_solve_script(
                plan(EQUATION, relation="ge"),
                check_contract=EQUATION,
                template_contract=EQUATION,
                relation="ge",
                include_check=False,
            )
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Solve an inequality")
        assert created.status_code == 200, created.text
        record = created.json()
        _assert_fail_closed(record)
        assert any(
            "relation" in finding.lower() or "contract" in finding.lower()
            for finding in record["findings"]
        )
        assert (await harness.client.get(f"{BASE}/checks")).json() == []
        assert scripts.providers[0].calls == [CHECK_PLAN]


@pytest.mark.asyncio
async def test_plan_cannot_disable_complete_solution_policy(tmp_path: Path) -> None:
    payload = plan(EQUATION, relation="eq")
    payload["goal"]["requires_complete_solution_set"] = False
    scripts = ProviderScripts(
        [
            invalid_solve_script(
                payload,
                check_contract=EQUATION,
                template_contract=EQUATION,
                relation="eq",
                include_check=False,
            )
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Solve an equation")
        assert created.status_code == 200, created.text
        record = created.json()
        _assert_fail_closed(record)
        assert any(
            "complete" in finding.lower() or "solution" in finding.lower()
            for finding in record["findings"]
        )
        assert (await harness.client.get(f"{BASE}/checks")).json() == []
        assert scripts.providers[0].calls == [CHECK_PLAN]


@pytest.mark.asyncio
async def test_template_scope_mismatch_is_needs_review_not_substitution(
    tmp_path: Path,
) -> None:
    # The plan and check ask for an equation, but the concrete template asks
    # for an inequality. The application must not silently replace the task.
    scripts = ProviderScripts(
        [
            invalid_solve_script(
                plan(EQUATION, relation="eq"),
                check_contract=EQUATION,
                template_contract=EQUATION,
                relation="gt",
            )
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Solve an equation")
        assert created.status_code == 200, created.text
        record = created.json()
        assert record["status"] == "needs_review", record
        assert record["approval"] == "unreviewed"
        assert record["findings"]
        assert any(
            "relation" in finding.lower() or "template" in finding.lower()
            for finding in record["findings"]
        )
        assert record["qualification"] is None or record["qualification"]["status"] != "passed"
        rejected = await harness.client.post(
            f"{BASE}/templates/{record['template_id']}/approval",
            json={
                "template_sha256": record["template_sha256"],
                "check_sha256": record["check"]["check_ref"]["sha256"]
                if record["check"]
                else "0" * 64,
                "decision": "approved",
            },
        )
        assert rejected.status_code in {409, 422}


@pytest.mark.asyncio
async def test_unsupported_recipe_operation_has_no_execution_or_counterexample(
    tmp_path: Path,
) -> None:
    scripts = ProviderScripts(
        [
            {
                CHECK_PLAN: [plan(DIFFERENTIATE, degree=2)],
                CHECK_RECIPE: [check(DIFFERENTIATE, operation="poly.integrate")],
                POLYNOMIAL_FAMILY: [template(DIFFERENTIATE, degree=2)],
            }
        ]
    )
    async with _harness(tmp_path, scripts) as harness:
        created = await _author(harness.client, "Integrate a polynomial")
        assert created.status_code == 200, created.text
        record = created.json()
        _assert_fail_closed(record)
        assert record["check"] is None
        qualification = record.get("check_qualification")
        assert qualification is not None
        assert qualification["status"] == "error"
        assert qualification["phase"] == "compilation"
        assert qualification["execution_calls"] == 0
        assert qualification["counterexample"] is None
        assert any(
            "operation" in finding.lower() or "integrate" in finding.lower()
            for finding in record["findings"]
        )
        # Unsupported operations are compilation evidence, not a mathematical
        # execution and not a generated-code installation.
        assert (await harness.client.get(f"{BASE}/checks")).json() == []


@pytest.mark.asyncio
async def test_foreign_saved_check_reference_fails_closed(tmp_path: Path) -> None:
    owner_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    owner_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    scripts = ProviderScripts([solve_script(EQUATION, relation="eq")])
    async with _harness(tmp_path, scripts, owner=owner_a) as harness:
        created = await _author(harness.client, "Solve a quadratic equation")
        assert created.status_code == 200, created.text
        record_a = created.json()
        assert record_a["status"] == "ready", record_a["findings"]
        foreign_ref = record_a["check"]["check_ref"]

        scripts.scripts.append(solve_script(EQUATION, relation="eq", saved_check_ref=foreign_ref))
        harness.client.cookies.clear()
        app.dependency_overrides[get_current_user] = lambda: _User(UUID(owner_b))
        created = await _author(harness.client, "Reuse another owner's check")
        assert created.status_code == 200, created.text
        record_b = created.json()
        _assert_fail_closed(record_b)
        assert any(
            "check" in finding.lower()
            or "not found" in finding.lower()
            or "owner" in finding.lower()
            for finding in record_b["findings"]
        )
        # Owner B has no private check; owner A's catalogue is untouched.
        assert (await harness.client.get(f"{BASE}/checks")).json() == []

        app.dependency_overrides[get_current_user] = lambda: _User(UUID(owner_a))
        checks_a = await harness.client.get(f"{BASE}/checks")
        assert [item["check_ref"] for item in checks_a.json()] == [foreign_ref]
