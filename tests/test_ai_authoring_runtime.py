"""Resolve every authoring role through one runtime seam."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from edcraft_backend.config.settings import AuthoringProfileSettings
from edcraft_backend.schemas.ai_authoring import (
    AuthoringJudgeSelections,
    AuthoringModelSelection,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.ai_authoring_runtime import (
    AuthoringValidationError,
    JobRuntime,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection
from tests.ai_authoring_helpers import make_command

PROFILE = AuthoringProfileSettings(
    id="test-profile",
    label="Test author",
    provider="soclaas",
    model="profile-model",
)


def _request(
    *,
    model_profile: str | None = "test-profile",
    connection_id: UUID | None = None,
    model: str | None = None,
    judges: AuthoringJudgeSelections | None = None,
) -> SubmitAuthoringJobRequest:
    return SubmitAuthoringJobRequest(
        prompt="Create an arithmetic tracing question",
        difficulty="easy",
        num_distractors=3,
        model_profile=model_profile,
        connection_id=connection_id,
        model=model,
        judges=judges,
    )


def _connection(model: str = "judge-model") -> ResolvedConnection:
    return ResolvedConnection.for_connection(
        connection_id=uuid4(),
        provider="openai",
        model=model,
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )


def _unused(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("The executor must not run during prepare")


def test_prepare_uses_the_configured_judge_for_relevance_by_default() -> None:
    judge = ResolvedConnection.for_judge(provider="soclaas", model="x-test-3")
    runtime = JobRuntime.prepare(
        _request(),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=_unused,
        judge_default=judge,
    )

    assert runtime.author.selection.provider == "soclaas"
    assert runtime.author.selection.model == "profile-model"
    assert runtime.author.is_byok is False
    assert runtime.difficulty == runtime.author
    assert runtime.relevance == judge
    assert runtime.relevance.is_byok is False
    assert runtime.relevance.selection.model == "x-test-3"


def test_prepare_falls_back_to_the_author_without_a_judge_default() -> None:
    runtime = JobRuntime.prepare(
        _request(),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=_unused,
    )

    assert runtime.difficulty == runtime.author
    assert runtime.relevance == runtime.author


def test_prepare_uses_the_pre_resolved_judge_connection() -> None:
    difficulty = _connection("judge-difficulty")
    judges = AuthoringJudgeSelections(
        difficulty=AuthoringModelSelection(
            connection_id=difficulty.connection_id, model="judge-difficulty"
        )
    )
    judge_default = ResolvedConnection.for_judge(provider="soclaas", model="x-test-3")

    runtime = JobRuntime.prepare(
        _request(judges=judges),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=_unused,
        difficulty_connection=difficulty,
        judge_default=judge_default,
    )

    assert runtime.difficulty is difficulty
    assert runtime.relevance == judge_default


def test_prepare_keeps_an_explicit_relevance_connection_over_the_default() -> None:
    relevance = _connection("judge-relevance")
    judges = AuthoringJudgeSelections(
        relevance=AuthoringModelSelection(
            connection_id=relevance.connection_id, model="judge-relevance"
        )
    )
    judge_default = ResolvedConnection.for_judge(provider="soclaas", model="x-test-3")

    runtime = JobRuntime.prepare(
        _request(judges=judges),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=_unused,
        relevance_connection=relevance,
        judge_default=judge_default,
    )

    assert runtime.relevance is relevance


def test_prepare_allows_a_profile_judge_over_a_byok_author() -> None:
    author = _connection("author-model")
    judges = AuthoringJudgeSelections(
        difficulty=AuthoringModelSelection(model_profile="test-profile")
    )

    runtime = JobRuntime.prepare(
        _request(
            model_profile=None,
            connection_id=author.connection_id,
            model="author-model",
            judges=judges,
        ),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=_unused,
        author_connection=author,
    )

    assert runtime.author is author
    assert runtime.difficulty.selection.provider == "soclaas"
    assert runtime.difficulty.is_byok is False


def test_prepare_rejects_a_missing_connection() -> None:
    connection_id = uuid4()
    judges = AuthoringJudgeSelections(
        difficulty=AuthoringModelSelection(connection_id=connection_id, model="m")
    )

    with pytest.raises(AuthoringValidationError, match="Difficulty model connection"):
        JobRuntime.prepare(
            _request(judges=judges),
            profile_lookup=lambda _profile_id: PROFILE,
            executor=_unused,
        )


def test_prepare_rejects_an_unknown_author_profile() -> None:
    def unknown(_profile_id: str) -> AuthoringProfileSettings:
        raise AuthoringValidationError("Unknown model profile.")

    with pytest.raises(AuthoringValidationError, match="Unknown model profile"):
        JobRuntime.prepare(
            _request(),
            profile_lookup=unknown,
            executor=_unused,
        )


def test_run_delegates_to_the_executor_seam() -> None:
    sentinel = object()
    calls: list[tuple[Any, JobRuntime, Any]] = []
    progress: list[Any] = []

    def executor(
        command: Any, runtime: JobRuntime, on_progress: Any
    ) -> object:
        calls.append((command, runtime, on_progress))
        return sentinel

    runtime = JobRuntime.prepare(
        _request(),
        profile_lookup=lambda _profile_id: PROFILE,
        executor=executor,
    )
    command = make_command()

    assert runtime.run(command, progress.append) is sentinel
    assert calls == [(command, runtime, progress.append)]
