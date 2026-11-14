"""Resolve the authoring model roles and run one job behind a single runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress

from edcraft_backend.config.settings import AuthoringProfileSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.schemas.ai_authoring import (
    AuthoringModelSelection,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection

ProgressCallback = Callable[[AuthoringProgress], None]


class AuthoringValidationError(EdCraftBaseException):
    """The requested model configuration cannot form an authoring job."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 422)


class JobExecutor(Protocol):
    """Synchronous library boundary executed inside a worker thread."""

    def __call__(
        self,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
        on_progress: ProgressCallback | None,
    ) -> AuthoringJobRecord: ...


class JobRuntime:
    """The three resolved model roles for exactly one authoring job."""

    def __init__(
        self,
        *,
        author: ResolvedConnection,
        difficulty: ResolvedConnection,
        relevance: ResolvedConnection,
        executor: JobExecutor,
    ) -> None:
        self.author = author
        self.difficulty = difficulty
        self.relevance = relevance
        self._executor = executor

    @classmethod
    def prepare(
        cls,
        request: SubmitAuthoringJobRequest,
        *,
        profile_lookup: Callable[[str], AuthoringProfileSettings],
        executor: JobExecutor,
        author_connection: ResolvedConnection | None = None,
        difficulty_connection: ResolvedConnection | None = None,
        relevance_connection: ResolvedConnection | None = None,
        judge_default: ResolvedConnection | None = None,
    ) -> JobRuntime:
        """Resolve all three roles from the request and pre-resolved connections.

        ``judge_default`` is the server-configured advisory judge used when the
        request does not override relevance. It is separate from the author
        because a decision judge needs token probabilities, which not every
        author model exposes.
        """
        author = cls._author_role(
            request, profile_lookup=profile_lookup, connection=author_connection
        )
        judges = request.judges
        return cls(
            author=author,
            difficulty=cls._judge_role(
                judges.difficulty if judges else None,
                default=author,
                connection=difficulty_connection,
                profile_lookup=profile_lookup,
                label="Difficulty",
            ),
            relevance=cls._judge_role(
                judges.relevance if judges else None,
                default=judge_default or author,
                connection=relevance_connection,
                profile_lookup=profile_lookup,
                label="Relevance",
            ),
            executor=executor,
        )

    def run(
        self,
        command: AuthoringJobCommand,
        on_progress: ProgressCallback | None = None,
    ) -> AuthoringJobRecord:
        """Run the validator through the injected executor seam."""
        return self._executor(command, self, on_progress)

    @staticmethod
    def _author_role(
        request: SubmitAuthoringJobRequest,
        *,
        profile_lookup: Callable[[str], AuthoringProfileSettings],
        connection: ResolvedConnection | None,
    ) -> ResolvedConnection:
        if request.connection_id is not None:
            if connection is None:
                raise AuthoringValidationError("Model connection is unavailable.")
            return connection
        if request.model_profile is None:
            raise AuthoringValidationError("Choose a question-writing model.")
        return ResolvedConnection.for_profile(profile_lookup(request.model_profile))

    @staticmethod
    def _judge_role(
        selection: AuthoringModelSelection | None,
        *,
        default: ResolvedConnection,
        connection: ResolvedConnection | None,
        profile_lookup: Callable[[str], AuthoringProfileSettings],
        label: str,
    ) -> ResolvedConnection:
        """Resolve one advisory override, inheriting the author by default."""
        if selection is None:
            return default
        if selection.connection_id is not None:
            if connection is None:
                raise AuthoringValidationError(
                    f"{label} model connection is unavailable."
                )
            return connection
        if selection.model_profile is None:
            raise AuthoringValidationError(f"Choose a {label.lower()} model.")
        return ResolvedConnection.for_profile(profile_lookup(selection.model_profile))


__all__ = [
    "AuthoringValidationError",
    "JobExecutor",
    "JobRuntime",
    "ProgressCallback",
]
