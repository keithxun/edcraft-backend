"""Preflight and submission agree on the effective advisory models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from edcraft_validator.llm.forced_decision import ClassDecision
from edcraft_validator.llm.llm_errors import DecisionCapabilityError, GenerationTransportError

from edcraft_backend.dependencies import get_current_user
from edcraft_backend.main import app
from edcraft_backend.services import ai_authoring_compatibility
from tests.ai_authoring_helpers import StaticExecutor, make_checked_report
from tests.integration.test_ai_authoring_api import env_for, payload


async def test_judge_preflight_requires_sign_in(tmp_path: Path) -> None:
    async with env_for(tmp_path, StaticExecutor(make_checked_report)) as env:
        app.dependency_overrides.pop(get_current_user)
        response = await env.client.post(
            "/ai-authoring/judge-compatibility", json={"model_profile": "test-profile"}
        )
        assert response.status_code == 401


async def test_one_unsupported_judge_uses_verified_fallback_in_preflight_and_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDCRAFT_JUDGE_MODEL", raising=False)
    async with env_for(tmp_path, StaticExecutor(make_checked_report)) as env:
        calls = []

        class FakeProvider:
            def __init__(self, role: Any) -> None:
                self.role = role

            def decide(self, messages: list[dict[str, str]], labels: tuple[str, ...]) -> Any:
                calls.append((self.role.model, labels, messages))
                if self.role.model == "stub-model" and labels == ("yes", "no"):
                    raise DecisionCapabilityError("decision logprobs unavailable")
                return ClassDecision(
                    label=labels[0],
                    distribution={label: float(label == labels[0]) for label in labels},
                    confidence=1.0,
                )

        monkeypatch.setattr(ai_authoring_compatibility, "_open_probe_provider", FakeProvider)
        runtimes = []

        async def capture(job: Any, command: Any, runtime: Any) -> Any:
            runtimes.append(runtime)
            return job

        monkeypatch.setattr(env.service.lifecycle, "submit", capture)
        submitted = await env.client.post("/ai-authoring/jobs", json=payload())
        assert submitted.status_code == 202, submitted.text
        (runtime,) = runtimes
        assert runtime.author.model == "stub-model"
        assert runtime.difficulty.model == "stub-model"
        assert runtime.relevance.model == "x-test-2"

        options = {"model_profile": "test-profile"}
        public = {
            "difficulty": {"status": "compatible", "provider": "stub", "model": "stub-model"},
            "relevance": {"status": "fallback", "provider": "soclaas", "model": "x-test-2"},
        }
        responses = await asyncio.gather(
            *(env.client.post("/ai-authoring/judge-compatibility", json=options) for _ in range(2))
        )
        assert all(response.status_code == 200 for response in responses)
        assert [response.json() for response in responses] == [public, public]
        assert len(calls) == 3
        assert "Create an arithmetic tracing question" not in str(calls)


async def test_preflight_keeps_transient_primary_and_requires_a_valid_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDCRAFT_JUDGE_MODEL", raising=False)
    async with env_for(tmp_path, StaticExecutor(make_checked_report)) as env:
        calls = []

        class FakeProvider:
            def __init__(self, role: Any) -> None:
                self.role = role

            def decide(self, messages: list[dict[str, str]], labels: tuple[str, ...]) -> Any:
                calls.append((self.role.model, labels))
                if labels == ("yes", "no"):
                    raise GenerationTransportError("temporary quota error")
                if self.role.model == "stub-model":
                    raise DecisionCapabilityError("decision logprobs unavailable")
                return ClassDecision.model_construct(
                    label="easy",
                    distribution={"easy": 1.0, "medium": 0.3, "hard": 0.1},
                    confidence=1.0,
                )

        monkeypatch.setattr(ai_authoring_compatibility, "_open_probe_provider", FakeProvider)
        checked = await env.client.post(
            "/ai-authoring/judge-compatibility", json={"model_profile": "test-profile"}
        )
        assert checked.status_code == 200
        assert checked.json() == {
            "difficulty": {"status": "unavailable", "provider": "stub", "model": "stub-model"},
            "relevance": {"status": "unknown", "provider": "stub", "model": "stub-model"},
        }
        assert ("x-test-2", ("yes", "no")) not in calls

        submitted = await env.client.post("/ai-authoring/jobs", json=payload())
        assert submitted.status_code == 202
        await env.service.wait_for_idle()
        (runtime,) = env.executor.runtimes
        assert runtime.difficulty == runtime.relevance == runtime.author
