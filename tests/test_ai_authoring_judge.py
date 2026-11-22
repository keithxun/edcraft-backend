"""Verify authoring and advisory model wiring for each job."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from edcraft_validator.llm.forced_decision import ClassDecision
from edcraft_validator.llm.llm_contracts import TemplateProviderSelection

from edcraft_backend.services import ai_authoring_judge, ai_authoring_runner
from edcraft_backend.services.ai_authoring_judge import (
    SelectedModelDecisionProvider,
    build_model_tool_client,
    build_plan_provider,
    judge_settings_for,
)
from edcraft_backend.services.ai_authoring_providers import ManagedProvider
from edcraft_backend.services.ai_authoring_runtime import (
    AuthoringValidationError,
    JobRuntime,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection
from tests.ai_authoring_helpers import make_command


class StubProvider:
    """Minimal provider that records forced classification calls."""

    def __init__(
        self, *, provider: str = "openai", model: str = "gpt-4.1-mini"
    ) -> None:
        self.provider = provider
        self.model = model
        self.decisions: list[tuple[list[dict], tuple[str, ...]]] = []

    def generation_settings(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}

    def generate(self, request):  # pragma: no cover - not used by judges
        raise AssertionError("Judge wiring must not generate proposals")

    def decide(self, messages, labels) -> ClassDecision:
        self.decisions.append((messages, labels))
        return ClassDecision(
            label="yes",
            distribution={"yes": 0.8, "no": 0.2},
            confidence=0.8,
            served_model=self.model,
        )


def selection(provider: str, model: str) -> TemplateProviderSelection:
    return TemplateProviderSelection(provider=provider, model=model)


def test_managed_provider_forwards_decisions_and_usage() -> None:
    delegate = StubProvider()
    delegate.last_usage = {"input_tokens": 5}
    client = SimpleNamespace(close=lambda: None)
    provider = ManagedProvider(
        delegate, client, provider="openai", model="gpt-4.1-mini"
    )

    decision = provider.decide([], ("yes", "no"))

    assert decision.label == "yes"
    assert provider.last_usage == {"input_tokens": 5}


def test_managed_provider_rejects_delegates_without_decisions() -> None:
    class NoDecisions:
        pass

    provider = ManagedProvider(
        NoDecisions(), SimpleNamespace(close=lambda: None), provider="openai", model="x"
    )

    with pytest.raises(ValueError, match="token-probability"):
        provider.decide([], ("yes", "no"))


def test_judge_settings_for_keeps_server_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDCRAFT_JUDGE_MODEL", "")
    monkeypatch.setenv("EDCRAFT_RELEVANCE_THRESHOLD", "0.75")
    monkeypatch.setenv("EDCRAFT_JUDGE_REASONING_EFFORT", "high")

    settings = judge_settings_for("openai", "gpt-4.1-mini")

    assert settings.provider == "openai"
    assert settings.model == "gpt-4.1-mini"
    assert settings.relevance_threshold == 0.75
    assert settings.reasoning_effort == "high"


def test_selected_model_decision_provider_scores_with_its_model() -> None:
    settings = judge_settings_for("openai", "gpt-4.1-mini")
    provider = StubProvider()

    decision_provider = SelectedModelDecisionProvider(settings, provider)

    assert decision_provider.provider == "openai"
    assert decision_provider.model == "gpt-4.1-mini"
    assert decision_provider.threshold == settings.relevance_threshold
    assert decision_provider.score("build a loop", "write a loop") == pytest.approx(0.8)
    assert decision_provider.served_model == "gpt-4.1-mini"
    assert provider.decisions[0][1] == ("yes", "no")


def test_selected_model_decision_provider_rejects_providers_without_decisions() -> None:
    class NoDecisions:
        provider = "openai"
        model = "gpt-4.1-mini"

        def generation_settings(self) -> dict[str, str]:
            return {}

    with pytest.raises(ValueError, match="token-probability"):
        SelectedModelDecisionProvider(
            judge_settings_for("openai", "gpt-4.1-mini"), NoDecisions()
        )


def test_resolved_connection_repr_hides_api_key() -> None:
    connection = ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-secret-value",
        base_url="https://api.openai.com/v1",
    )

    rendered = repr(connection)

    assert "sk-secret-value" not in rendered
    assert "openai" in rendered
    assert "gpt-4.1-mini" in rendered
    assert "https://api.openai.com/v1" in rendered


def test_build_plan_provider_prefers_byok_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    managed = StubProvider()
    monkeypatch.setattr(ResolvedConnection, "open_provider", lambda _self: managed)

    assert build_plan_provider(role) is managed


def test_build_plan_provider_uses_a_server_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = StubProvider(provider="soclaas", model="x-test-2")
    role = ResolvedConnection(selection=selection("soclaas", "x-test-2"))
    monkeypatch.setattr(
        ai_authoring_judge, "create_model_provider", lambda _selection: managed
    )

    assert build_plan_provider(role) is managed


def test_model_tool_client_wires_different_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    server = object()

    def fake_server(**kwargs):
        captured.update(kwargs)
        return server

    difficulty_provider = StubProvider(provider="soclaas", model="x-test-2")
    relevance_provider = StubProvider(provider="openai", model="gpt-4.1-mini")
    providers = {
        ("soclaas", "x-test-2"): difficulty_provider,
        ("openai", "gpt-4.1-mini"): relevance_provider,
    }

    monkeypatch.setattr(ai_authoring_judge, "create_validation_server", fake_server)
    monkeypatch.setattr(
        ai_authoring_judge,
        "FastMcpToolClient",
        lambda created: SimpleNamespace(server=created),
    )
    monkeypatch.setattr(
        ai_authoring_judge,
        "build_plan_provider",
        lambda role: providers[(role.selection.provider, role.selection.model)],
    )

    client = build_model_tool_client(
        difficulty=ResolvedConnection(selection=selection("soclaas", "x-test-2")),
        relevance=ResolvedConnection(selection=selection("openai", "gpt-4.1-mini")),
    )

    assert client.server is server
    assert captured["judge_settings"].provider == "soclaas"
    assert captured["judge_settings"].model == "x-test-2"
    assert captured["difficulty_provider_factory"]() is difficulty_provider

    decision_provider = captured["decision_provider_factory"]()
    assert decision_provider.delegate is relevance_provider
    assert decision_provider.provider == "openai"
    assert decision_provider.model == "gpt-4.1-mini"


def test_model_tool_client_requires_named_models() -> None:
    with pytest.raises(AuthoringValidationError, match="name a model"):
        build_model_tool_client(
            difficulty=ResolvedConnection(
                selection=TemplateProviderSelection(provider="soclaas")
            ),
            relevance=ResolvedConnection(selection=selection("soclaas", "x-test-2")),
        )


def test_runner_profile_path_uses_server_judge_tool_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = make_command()
    difficulty = ResolvedConnection(selection=selection("soclaas", "x-test-2"))
    relevance = ResolvedConnection(selection=selection("openai", "gpt-4.1-mini"))
    runtime = JobRuntime(
        author=difficulty,
        difficulty=difficulty,
        relevance=relevance,
        executor=lambda *_args: None,
    )
    tool_client = object()
    record = object()
    captured: dict = {}

    def fake_run(_command, **kwargs):
        captured.update(kwargs)
        return record

    def fake_tool_client(**kwargs):
        captured["tool_plan"] = kwargs
        return tool_client

    monkeypatch.setattr(
        ai_authoring_runner, "build_model_tool_client", fake_tool_client
    )
    monkeypatch.setattr(ai_authoring_runner, "run_authoring_job_sync", fake_run)
    monkeypatch.setattr(
        ai_authoring_runner,
        "TemplateApplication",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = ai_authoring_runner.run_library_authoring_job(command, runtime, None)

    assert result is record
    assert captured["tool_plan"] == {
        "difficulty": runtime.difficulty,
        "relevance": runtime.relevance,
    }
    assert captured["application_factory"]().tool_client is tool_client
    assert "provider_factory" not in captured


def test_runner_byok_path_uses_author_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = make_command()
    author = ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    runtime = JobRuntime(
        author=author,
        difficulty=author,
        relevance=author,
        executor=lambda *_args: None,
    )
    tool_client = object()
    managed = object()
    record = object()
    captured: dict = {}

    def fake_run(_command, **kwargs):
        captured.update(kwargs)
        return record

    monkeypatch.setattr(
        ai_authoring_runner, "build_model_tool_client", lambda **kwargs: tool_client
    )
    monkeypatch.setattr(ResolvedConnection, "open_provider", lambda _self: managed)
    monkeypatch.setattr(ai_authoring_runner, "run_authoring_job_sync", fake_run)
    monkeypatch.setattr(
        ai_authoring_runner,
        "TemplateApplication",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = ai_authoring_runner.run_library_authoring_job(command, runtime, None)

    assert result is record
    assert captured["application_factory"]().tool_client is tool_client
    assert captured["provider_factory"](command.author) is managed
