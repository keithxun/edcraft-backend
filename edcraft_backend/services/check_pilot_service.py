"""Thin, bounded integration with the AI check-pilot application.

The backend owns provider selection, admission and HTTP error mapping. The AI
library owns every pilot semantic: recipes, qualification, storage and
generation. This module only runs one worker-thread call at a time under a
bounded admission/timeout policy and closes the resolved provider on all paths.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID

from edcraft_validator.application.check_pilot import (
    CheckRecord,
    PilotBatch,
    PilotConflict,
    PilotError,
    PilotInputError,
    PilotNotFound,
    PilotTemplateRecord,
)
from edcraft_validator.llm.llm_contracts import ModelProvider
from edcraft_validator.llm.usage import close_provider

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    CheckPilotSettings,
)
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.schemas.check_pilot import (
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)
from edcraft_backend.services.ai_authoring_judge import build_plan_provider
from edcraft_backend.services.provider_endpoint import ResolvedConnection

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CheckPilotError(EdCraftBaseException):
    """Base class for safe check-pilot HTTP failures."""


class CheckPilotNotFound(CheckPilotError):
    """The owner has no such private pilot artifact."""

    def __init__(
        self, message: str = "The requested check pilot record was not found."
    ) -> None:
        super().__init__(message, 404)


class CheckPilotConflict(CheckPilotError):
    """A stale digest, rejected template or immutable-version conflict."""

    def __init__(
        self, message: str = "The check pilot record changed; reload it and try again."
    ) -> None:
        super().__init__(message, 409)


class CheckPilotInputError(CheckPilotError):
    """The request is well-formed HTTP but unsupported by the pilot."""

    def __init__(
        self, message: str = "The check pilot request could not be accepted."
    ) -> None:
        super().__init__(message, 422)


class CheckPilotAdmissionFull(CheckPilotError):
    """The bounded worker capacity was not free within the admission window."""

    def __init__(
        self,
        message: str = "Check authoring is busy; try again when a request finishes.",
    ) -> None:
        super().__init__(message, 429)


class CheckPilotTimeout(CheckPilotError):
    """The worker or provider exceeded the configured wall-clock bound."""

    def __init__(
        self, message: str = "Check authoring timed out before it produced a result."
    ) -> None:
        super().__init__(message, 504)


class CheckPilotUnavailable(CheckPilotError):
    """A worker, provider or storage failure; never a passing result."""

    def __init__(
        self,
        message: str = "Check authoring could not complete. Try again or choose another model.",
    ) -> None:
        super().__init__(message, 503)


class PilotApplication(Protocol):
    """The subset of ``CheckPilotApplication`` the backend invokes."""

    def author_template(
        self,
        owner_id: str,
        *,
        prompt: str,
        provider: ModelProvider,
        num_distractors: int = 3,
    ) -> PilotTemplateRecord: ...

    def list_checks(self, owner_id: str) -> list[CheckRecord]: ...

    def get_check(self, owner_id: str, check_id: str) -> CheckRecord: ...

    def list_templates(self, owner_id: str) -> list[PilotTemplateRecord]: ...

    def get_template(self, owner_id: str, template_id: str) -> PilotTemplateRecord: ...

    def decide(
        self,
        owner_id: str,
        template_id: str,
        *,
        template_sha256: str,
        check_sha256: str,
        decision: str,
    ) -> PilotTemplateRecord: ...

    def generate(
        self, owner_id: str, template_id: str, *, seed: int, quantity: int
    ) -> PilotBatch: ...


def _consume_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a worker outcome, including a late one after a request timeout."""
    if not task.cancelled():
        task.exception()


def _safe_message(error: Exception, limit: int = 500) -> str:
    """Forward only the AI-owned safe message, capped for the HTTP surface."""
    message = str(error).strip()
    if not message:
        return "The check pilot request could not be completed."
    return message[:limit]


class CheckPilotService:
    """Owner-scoped pilot calls behind a bounded worker and provider lifecycle."""

    def __init__(
        self,
        *,
        authoring: AuthoringSettings,
        settings: CheckPilotSettings,
        application_factory: Callable[[], PilotApplication] | None = None,
        provider_builder: Callable[
            [ResolvedConnection], ModelProvider
        ] = build_plan_provider,
    ) -> None:
        self._authoring = authoring
        self._settings = settings
        self._application_factory = application_factory or self._new_application
        self._provider_builder = provider_builder
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def root(self) -> Path:
        """Private pilot storage, derived from the existing authoring directory."""
        return self._authoring.storage_dir / "check-pilot"

    async def author_template(
        self,
        owner_id: UUID,
        request: PilotTemplateRequest,
        *,
        connection: ResolvedConnection | None,
    ) -> PilotTemplateRecord:
        """Run one author/check-author/provider call for the owner."""
        role = self._role(request, connection=connection)
        owner = str(owner_id)
        prompt = request.prompt
        num_distractors = request.num_distractors

        def operation() -> Any:
            provider = self._provider_builder(role)
            try:
                return self._application_factory().author_template(
                    owner,
                    prompt=prompt,
                    provider=provider,
                    num_distractors=num_distractors,
                )
            finally:
                close_provider(provider)

        return await self._execute(operation, label="author_template")

    async def list_checks(self, owner_id: UUID) -> list[CheckRecord]:
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().list_checks(owner),
            label="list_checks",
        )

    async def get_check(self, owner_id: UUID, check_id: str) -> CheckRecord:
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().get_check(owner, check_id),
            label="get_check",
        )

    async def list_templates(self, owner_id: UUID) -> list[PilotTemplateRecord]:
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().list_templates(owner),
            label="list_templates",
        )

    async def get_template(self, owner_id: UUID, template_id: str) -> PilotTemplateRecord:
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().get_template(owner, template_id),
            label="get_template",
        )

    async def decide(
        self,
        owner_id: UUID,
        template_id: str,
        request: PilotApprovalRequest,
    ) -> PilotTemplateRecord:
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().decide(
                owner,
                template_id,
                template_sha256=request.template_sha256,
                check_sha256=request.check_sha256,
                decision=request.decision,
            ),
            label="decide",
        )

    async def generate(
        self,
        owner_id: UUID,
        template_id: str,
        request: PilotGenerateRequest,
    ) -> PilotBatch:
        limit = min(self._authoring.max_quantity, 50)
        if request.quantity > limit:
            raise CheckPilotInputError(
                f"Quantity must be at most {limit} for this server."
            )
        owner = str(owner_id)
        return await self._execute(
            lambda: self._application_factory().generate(
                owner, template_id, seed=request.seed, quantity=request.quantity
            ),
            label="generate",
        )

    def _role(
        self,
        request: PilotTemplateRequest,
        *,
        connection: ResolvedConnection | None,
    ) -> ResolvedConnection:
        if request.connection_id is not None:
            if connection is None:
                raise CheckPilotInputError("Model connection is unavailable.")
            return connection
        if request.model_profile is None:
            raise CheckPilotInputError("Choose a question-writing model.")
        return ResolvedConnection.for_profile(self._profile(request.model_profile))

    def _profile(self, profile_id: str) -> AuthoringProfileSettings:
        for profile in self._authoring.model_profiles:
            if profile.id == profile_id:
                return profile
        raise CheckPilotInputError("Unknown model profile.")

    def _new_application(self) -> PilotApplication:
        # Imported lazily so this module (and the HTTP app) never creates
        # storage or a provider at import time, and so the pinned wheel stays
        # usable until the orchestrator updates the dependency revision.
        from edcraft_validator.application.check_pilot import CheckPilotApplication

        return cast(PilotApplication, CheckPilotApplication(self.root))

    async def _execute(
        self,
        operation: Callable[[], T],
        *,
        label: str,
    ) -> T:
        """Run one blocking application call under admission and a deadline."""
        try:
            return await self._run_blocking(operation, label=label)
        except CheckPilotError:
            raise
        except PilotNotFound as exc:
            raise CheckPilotNotFound(_safe_message(exc)) from exc
        except PilotConflict as exc:
            raise CheckPilotConflict(_safe_message(exc)) from exc
        except PilotInputError as exc:
            raise CheckPilotInputError(_safe_message(exc)) from exc
        except PilotError as exc:
            logger.exception("Check pilot error", extra={"operation": label})
            raise CheckPilotUnavailable(_safe_message(exc)) from exc
        except Exception as exc:
            logger.exception("Check pilot call failed", extra={"operation": label})
            raise CheckPilotUnavailable() from exc

    async def _run_blocking(
        self,
        operation: Callable[[], T],
        *,
        label: str,
    ) -> T:
        self._ensure_loop_state()
        semaphore = self._semaphore
        if semaphore is None:
            raise CheckPilotUnavailable()
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self._settings.admission_timeout_seconds,
            )
        except TimeoutError as exc:
            raise CheckPilotAdmissionFull() from exc

        task: asyncio.Task[T] = asyncio.create_task(asyncio.to_thread(operation))
        task.add_done_callback(_consume_result)
        task.add_done_callback(lambda _: semaphore.release())
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._settings.request_timeout_seconds,
            )
        except TimeoutError as exc:
            logger.warning(
                "Check pilot operation timed out",
                extra={
                    "operation": label,
                    "timeout_seconds": self._settings.request_timeout_seconds,
                },
            )
            raise CheckPilotTimeout() from exc

    def _ensure_loop_state(self) -> None:
        loop = asyncio.get_running_loop()
        if loop is self._loop:
            return
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)


__all__ = [
    "CheckPilotAdmissionFull",
    "CheckPilotConflict",
    "CheckPilotError",
    "CheckPilotInputError",
    "CheckPilotNotFound",
    "CheckPilotService",
    "CheckPilotTimeout",
    "CheckPilotUnavailable",
    "PilotApplication",
]
