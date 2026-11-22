"""Capabilities and authoring defaults exposed by the service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.schemas.ai_authoring import SubmitAuthoringJobRequest
from edcraft_backend.services.ai_authoring_runtime import JobRuntime
from edcraft_backend.services.ai_authoring_service import AiAuthoringService
from edcraft_backend.services.ai_authoring_store import AuthoringStore
from edcraft_backend.services.provider_endpoint import ResolvedConnection


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_submit_uses_only_an_explicit_relevance_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    monkeypatch.setenv("EDCRAFT_JUDGE_PROVIDER", "soclaas")
    if configured:
        monkeypatch.setenv("EDCRAFT_JUDGE_MODEL", "x-test-3")
    else:
        monkeypatch.delenv("EDCRAFT_JUDGE_MODEL", raising=False)
    settings = AuthoringSettings(storage_dir=tmp_path / "authoring")
    service = AiAuthoringService(settings, store=AuthoringStore(settings))
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

    expected = (
        ResolvedConnection.for_judge(provider="soclaas", model="x-test-3") if configured else None
    )
    assert captured["judge_default"] == expected
    assert ((await service.capabilities()).judge is not None) == configured


def test_an_invalid_configured_judge_yields_no_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDCRAFT_JUDGE_PROVIDER", "not-a-provider")
    monkeypatch.setenv("EDCRAFT_JUDGE_MODEL", "x-test-3")
    settings = AuthoringSettings(storage_dir=tmp_path / "authoring")
    service = AiAuthoringService(settings, store=AuthoringStore(settings))

    assert service._configured_judge() is None
