"""Resolve the authoring model roles and run one job behind a single runtime."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from collections.abc import Callable
from typing import Literal, Protocol
from uuid import UUID

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress
from edcraft_validator.llm.llm_contracts import TemplateProviderSelection
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from edcraft_backend.config.settings import AuthoringProfileSettings, AuthoringSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.schemas.ai_authoring import (
    AuthoringModelSelection,
    SubmitAuthoringJobRequest,
)
from edcraft_backend.services.provider_endpoint import (
    DEFAULT_TIMEOUT_SECONDS,
    ProviderEndpoint,
    ResolvedConnection,
)
from edcraft_backend.services.provider_errors import ProviderConnectionInvalid
from edcraft_backend.services.provider_secrets import ProviderSecretCipher, ProviderSecretError

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
                raise AuthoringValidationError(f"{label} model connection is unavailable.")
            return connection
        if selection.model_profile is None:
            raise AuthoringValidationError(f"Choose a {label.lower()} model.")
        return ResolvedConnection.for_profile(profile_lookup(selection.model_profile))


class ExecutionRole(BaseModel):
    """One role serialized inside the encrypted execution payload."""

    model_config = ConfigDict(hide_input_in_errors=True)

    provider: str
    model: str
    connection_id: UUID | None = None
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    timeout_seconds: float | None = None

    @classmethod
    def snapshot(
        cls, role: ResolvedConnection, *, remote: bool, required: bool = True
    ) -> ExecutionRole:
        provider = role.selection.provider
        api_key, base_url = role.api_key, role.base_url
        timeout: float | None = None
        if remote:
            if provider not in DEFAULT_TIMEOUT_SECONDS:
                raise AuthoringValidationError("This provider cannot run in an authoring worker.")
            timeout = DEFAULT_TIMEOUT_SECONDS[provider]
            if api_key is None:
                prefix = provider.upper()
                api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
                base_url = os.getenv(f"{prefix}_BASE_URL")
                try:
                    timeout = float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", str(timeout)))
                except ValueError:
                    timeout = 0.0
            try:
                if not api_key or not math.isfinite(timeout) or timeout <= 0:
                    raise AuthoringValidationError(
                        "An authoring worker provider is not configured."
                    )
                base_url = ProviderEndpoint.parse(provider, base_url).base_url
            except (AuthoringValidationError, ProviderConnectionInvalid):
                if required or role.is_byok:
                    raise
                # A misconfigured advisory profile is unavailable, as in inline mode.
                api_key, base_url, timeout = None, None, None
        return cls(
            provider=provider,
            model=role.model,
            connection_id=role.connection_id,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout,
        )

    def connection(self) -> ResolvedConnection:
        return ResolvedConnection(
            selection=TemplateProviderSelection(provider=self.provider, model=self.model),
            connection_id=self.connection_id,
            api_key=self.api_key,
            base_url=self.base_url,
        )


class ExecutionPayload(BaseModel):
    """Versioned job-local inputs, encrypted as one value in PostgreSQL."""

    model_config = ConfigDict(hide_input_in_errors=True)

    version: Literal[1] = 1
    command: AuthoringJobCommand = Field(repr=False)
    author: ExecutionRole
    difficulty: ExecutionRole
    relevance: ExecutionRole

    @classmethod
    def capture(
        cls, command: AuthoringJobCommand, runtime: JobRuntime, *, remote: bool
    ) -> ExecutionPayload:
        return cls(
            command=command,
            author=ExecutionRole.snapshot(runtime.author, remote=remote),
            difficulty=ExecutionRole.snapshot(runtime.difficulty, remote=remote, required=False),
            relevance=ExecutionRole.snapshot(runtime.relevance, remote=remote, required=False),
        )

    def runtime(self, executor: JobExecutor) -> JobRuntime:
        return JobRuntime(
            author=self.author.connection(),
            difficulty=self.difficulty.connection(),
            relevance=self.relevance.connection(),
            executor=executor,
        )

    def worker_context(self, library_revision: str, remaining_seconds: float) -> dict[str, object]:
        return {
            "job_id": str(self.command.job_id),
            "command": self.command.model_dump(mode="json"),
            "library_revision": library_revision,
            "timeout_seconds": remaining_seconds,
            "plan": {
                name: role.model_dump(mode="json", exclude={"connection_id"})
                for name, role in (
                    ("author", self.author),
                    ("difficulty", self.difficulty),
                    ("relevance", self.relevance),
                )
            },
        }


class ExecutionSecrets:
    """Keep queued credentials encrypted and authenticate a single worker attempt."""

    def __init__(self, settings: AuthoringSettings, installation_secret: str) -> None:
        configured = settings.encryption_key
        shared = configured.get_secret_value() if configured else installation_secret
        self._cipher = ProviderSecretCipher(
            hmac.new(shared.encode(), b"edcraft/authoring/execution/v1", hashlib.sha256).hexdigest()
        )
        self._worker_key = hmac.new(
            shared.encode(), b"edcraft/authoring/worker/v1", hashlib.sha256
        ).digest()

    def encrypt(self, payload: ExecutionPayload) -> str:
        return self._cipher.encrypt(payload.model_dump_json())

    def decrypt(self, ciphertext: str | None) -> ExecutionPayload:
        if ciphertext is not None:
            try:
                return ExecutionPayload.model_validate_json(self._cipher.decrypt(ciphertext))
            except (ProviderSecretError, ValidationError):
                pass
        raise ProviderSecretError("Authoring execution inputs are unavailable.")

    def worker_token(self, job_id: UUID, attempt_id: UUID) -> str:
        return hmac.new(
            self._worker_key, job_id.bytes + attempt_id.bytes, hashlib.sha256
        ).hexdigest()

    def valid_worker_token(self, job_id: UUID, attempt_id: UUID, token: str) -> bool:
        return (
            len(token) == 64
            and token.isascii()
            and hmac.compare_digest(token, self.worker_token(job_id, attempt_id))
        )


__all__ = [
    "AuthoringValidationError",
    "ExecutionPayload",
    "ExecutionSecrets",
    "JobExecutor",
    "JobRuntime",
    "ProgressCallback",
]
