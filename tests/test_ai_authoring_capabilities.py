"""Capabilities expose the effective check-pilot request bound."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from edcraft_backend.config.settings import AuthoringSettings, CheckPilotSettings
from edcraft_backend.main import app
from edcraft_backend.schemas.ai_authoring import (
    AuthoringCapabilitiesResponse,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.ai_authoring_runtime import JobRuntime
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.provider_endpoint import ResolvedConnection


@pytest.mark.asyncio
async def test_capabilities_publish_admission_plus_request_timeout(
    tmp_path: Path,
) -> None:
    service = AiAuthoringService(
        AuthoringSettings(storage_dir=tmp_path / "authoring"),
        check_pilot=CheckPilotSettings(
            admission_timeout_seconds=2.0,
            request_timeout_seconds=7.0,
        ),
    )

    capabilities = await service.capabilities()

    assert capabilities.check_pilot_timeout_seconds == 9.0


@pytest.mark.asyncio
async def test_capabilities_fall_back_to_default_pilot_bounds(
    tmp_path: Path,
) -> None:
    service = AiAuthoringService(AuthoringSettings(storage_dir=tmp_path / "authoring"))

    capabilities = await service.capabilities()

    # Default admission (5 s) plus default request bound (180 s).
    assert capabilities.check_pilot_timeout_seconds == 185.0


@pytest.mark.asyncio
async def test_submit_defaults_relevance_to_the_configured_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDCRAFT_JUDGE_PROVIDER", "soclaas")
    monkeypatch.setenv("EDCRAFT_JUDGE_MODEL", "x-test-3")
    service = AiAuthoringService(AuthoringSettings(storage_dir=tmp_path / "authoring"))
    author = ResolvedConnection.for_judge(provider="openai", model="gpt-5-mini")
    captured: dict[str, object] = {}

    def fake_prepare(request: SubmitAuthoringJobRequest, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(author=author)

    async def fake_submit(job: object, command: object, runtime: object) -> object:
        return job

    monkeypatch.setattr(JobRuntime, "prepare", staticmethod(fake_prepare))
    monkeypatch.setattr(service.lifecycle, "submit", fake_submit)

    await service.submit(
        uuid4(),
        SubmitAuthoringJobRequest(
            prompt="Create an arithmetic question",
            difficulty="easy",
            num_distractors=3,
            domain="code",
            model_profile="test-profile",
        ),
    )

    assert captured["judge_default"] == ResolvedConnection.for_judge(
        provider="soclaas", model="x-test-3"
    )


def test_an_invalid_configured_judge_yields_no_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDCRAFT_JUDGE_PROVIDER", "not-a-provider")
    service = AiAuthoringService(AuthoringSettings(storage_dir=tmp_path / "authoring"))

    assert service._configured_judge() is None


def _capabilities(timeout: float) -> AuthoringCapabilitiesResponse:
    return AuthoringCapabilitiesResponse(
        enabled_domains=[],
        difficulty_labels=[],
        model_profiles=[],
        max_quantity=1,
        min_distractors=1,
        max_distractors=3,
        default_distractors=1,
        check_pilot_timeout_seconds=timeout,
    )


def test_capabilities_pilot_timeout_must_be_positive_and_finite() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            _capabilities(value)


def test_capabilities_openapi_documents_the_combined_pilot_bound() -> None:
    schemas = app.openapi()["components"]["schemas"]
    field = schemas["AuthoringCapabilitiesResponse"]["properties"]["check_pilot_timeout_seconds"]

    assert field["type"] == "number"
    assert field["exclusiveMinimum"] == 0
