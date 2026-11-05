"""Business rules for owner-scoped AI authoring, review and generation."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    PublicAuthoringError,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringProgress,
    AuthoringResult,
    GenerationAttempt,
)
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.domains.code.code_schemas import (
    CodeQuestionInstance,
    CodeTemplateCandidate,
    TemplateValidationSummary,
    ValidatedCodeTemplate,
)
from edcraft_validator.domains.code.code_types import (
    DEFAULT_DISTRACTORS,
    MAX_DISTRACTORS,
    MIN_DISTRACTORS,
)
from edcraft_validator.domains.domain_registry import create_domain
from edcraft_validator.llm.judge_configuration import load_judge_settings
from edcraft_validator.llm.llm_contracts import TemplateProviderSelection
from pydantic import ValidationError

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    NomadSettings,
)
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.executors.nomad_authoring import (
    AUTHORING_JOB_PREFIX,
    NomadAuthoringExecutor,
)
from edcraft_backend.schemas.ai_authoring import (
    ApprovalRequest,
    ApprovalResponse,
    AuthoringCapabilitiesResponse,
    AuthoringJobStatusResponse,
    AuthoringJobSummary,
    AuthoringModelProfile,
    AuthoringModelSelection,
    AuthoringRequestView,
    AuthoringReviewResponse,
    GeneratedQuestionsBatch,
    GenerateRequest,
    GenerationMetadata,
    JudgeDisplay,
    PreviewResponse,
    ReviewAttempt,
    ReviewCheck,
    ReviewFinding,
    ReviewTemplate,
    StoredReviewerApproval,
    SubmitAuthoringJobRequest,
    ValidationLimits,
)
from edcraft_backend.schemas.provider_connection import ProviderConnectionResponse
from edcraft_backend.services.ai_authoring_providers import (
    JobCredentials,
    JobModelPlan,
    ModelPlan,
    ProviderCredentials,
)
from edcraft_backend.services.ai_authoring_runner import (
    AuthoringRunner,
    JobExecutor,
    RemoteJobExecutor,
    WorkerChannel,
)
from edcraft_backend.services.ai_authoring_store import (
    ApprovalRecord,
    AuthoringConflict,
    AuthoringNotFound,
    AuthoringRequestRecord,
    AuthoringStorageUnavailable,
    AuthoringStore,
    StoredJob,
    require_bound_report,
    require_reusable_evidence,
)

ApprovalView = StoredReviewerApproval

logger = logging.getLogger(__name__)

LIBRARY_REVISION = "0392a726dac501e4f5ec4f2f3ea741d9785a3375"
MAX_FINDING_TEXT = 2_000


class AuthoringValidationError(EdCraftBaseException):
    """A safe request/config selection error (HTTP 422)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 422)


class AuthoringWorkerUnauthorized(EdCraftBaseException):
    """The worker callback token is unknown or expired (HTTP 401)."""

    def __init__(self) -> None:
        super().__init__("Authoring worker token is invalid or expired.", 401)


def hash_worker_token(token: str) -> str:
    """Return the only representation of a worker token that may be stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AiAuthoringService:
    """Thin integration layer over the pinned AI library and file store."""

    def __init__(
        self,
        settings: AuthoringSettings,
        *,
        store: AuthoringStore | None = None,
        executor: JobExecutor | None = None,
        remote_executor: RemoteJobExecutor | None = None,
        nomad_settings: NomadSettings | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or AuthoringStore(settings)
        self._worker_plans: dict[UUID, JobModelPlan] = {}
        self._worker_commands: dict[UUID, dict[str, Any]] = {}
        self._remote_mode = settings.executor == "nomad" and executor is None
        self._remote_configuration_error: str | None = None
        resolved_remote = remote_executor if self._remote_mode else None
        if self._remote_mode and resolved_remote is None:
            if settings.nomad_container_image:
                resolved_remote = self._build_nomad_executor(nomad_settings)
            else:
                # Never fall back to the builder image: it has no
                # ``edcraft_validator.worker`` module and would fail at runtime.
                self._remote_configuration_error = (
                    "AI authoring is not configured for Nomad execution: set "
                    "AUTHORING_NOMAD_CONTAINER_IMAGE to the authoring worker image."
                )
        self.runner = AuthoringRunner(
            self.store,
            settings,
            executor,
            remote_executor=resolved_remote,
            job_timeout_seconds=settings.job_timeout_seconds,
            on_remote_finished=self._discard_worker_context,
        )

    def _build_nomad_executor(
        self, nomad_settings: NomadSettings | None
    ) -> NomadAuthoringExecutor:
        """Build the production executor from Nomad + authoring settings."""
        if nomad_settings is None:
            from edcraft_backend.config import settings as app_settings

            nomad_settings = app_settings.nomad
        image = self.settings.nomad_container_image
        if image is None:  # Guarded by the caller; keeps the type narrow.
            raise AuthoringValidationError(
                "AI authoring is not configured for Nomad execution: set "
                "AUTHORING_NOMAD_CONTAINER_IMAGE to the authoring worker image."
            )
        return NomadAuthoringExecutor(
            host=nomad_settings.host,
            port=nomad_settings.port,
            token=nomad_settings.token,
            namespace=nomad_settings.namespace,
            datacenters=list(nomad_settings.datacenters),
            cpu_mhz=nomad_settings.cpu_mhz,
            memory_mb=nomad_settings.memory_mb,
            container_image=image,
            force_pull=self.settings.nomad_force_pull,
            library_revision=LIBRARY_REVISION,
        )

    def _discard_worker_context(self, job_id: UUID) -> None:
        """Drop in-memory credentials once a remote job is terminal."""
        self._worker_plans.pop(job_id, None)
        self._worker_commands.pop(job_id, None)

    async def startup(self) -> None:
        """Prepare durable storage, reconcile interruptions and bind the runner."""
        await asyncio.to_thread(self.store.ensure_ready)
        await asyncio.to_thread(self.store.reconcile_startup)
        await self.runner.start()

    async def shutdown(self, timeout: float | None = None) -> None:
        """Allow admitted work a bounded chance to finish before process exit."""
        await self.runner.shutdown(
            self.settings.shutdown_timeout_seconds if timeout is None else timeout
        )

    async def capabilities(
        self, connections: list[ProviderConnectionResponse] | None = None
    ) -> AuthoringCapabilitiesResponse:
        """Return safe server profiles plus the caller's own connections."""
        profiles = [
            AuthoringModelProfile(
                id=profile.id,
                label=profile.label,
                provider=profile.provider,
                model=profile.model,
            )
            for profile in self.settings.model_profiles
        ]
        judge: JudgeDisplay | None
        try:
            configured = load_judge_settings()
            judge = JudgeDisplay(
                label=f"{configured.provider} · {configured.model}",
                provider=configured.provider,
                model=configured.model,
            )
        except (ValueError, ValidationError):
            logger.warning("Configured advisory judge settings are invalid")
            judge = None
        return AuthoringCapabilitiesResponse(
            enabled_domains=["code"],
            difficulty_labels=["easy", "medium", "hard"],
            model_profiles=profiles,
            # Disabled connections stay visible in settings but cannot write questions.
            connections=[item for item in (connections or []) if item.enabled],
            judge=judge,
            max_quantity=min(self.settings.max_quantity, 50),
            min_distractors=MIN_DISTRACTORS,
            max_distractors=MAX_DISTRACTORS,
            default_distractors=DEFAULT_DISTRACTORS,
        )

    async def submit(
        self,
        owner_id: UUID,
        request: SubmitAuthoringJobRequest,
        credentials: JobCredentials | None = None,
    ) -> StoredJob:
        """Admit and persist one queued authoring job."""
        if self._remote_configuration_error is not None:
            raise AuthoringValidationError(self._remote_configuration_error)
        resolved = credentials or JobCredentials()
        if request.connection_id is not None:
            if resolved.author is None:
                raise AuthoringValidationError("Model connection is unavailable.")
            author = resolved.author
            author_plan = ModelPlan(
                selection=TemplateProviderSelection(
                    provider=author.provider, model=author.model
                ),
                credentials=author,
            )
            provider = author.provider
            model = author.model
            profile_id = None
            connection_id = request.connection_id
        else:
            if request.model_profile is None:
                raise AuthoringValidationError("Choose a question-writing model.")
            profile = self._profile(request.model_profile)
            author_plan = ModelPlan(
                selection=TemplateProviderSelection(
                    provider=profile.provider, model=profile.model
                )
            )
            provider = profile.provider
            model = profile.model
            profile_id = profile.id
            connection_id = None

        difficulty_plan = self._judge_model_plan(
            request.judges.difficulty if request.judges else None,
            default=author_plan,
            credentials=resolved.difficulty,
            label="Difficulty",
        )
        relevance_plan = self._judge_model_plan(
            request.judges.relevance if request.judges else None,
            default=author_plan,
            credentials=resolved.relevance,
            label="Relevance",
        )
        plan = JobModelPlan(
            author=author_plan,
            difficulty=difficulty_plan,
            relevance=relevance_plan,
        )

        await self.runner.reserve(owner_id)
        job_id = uuid4()
        now = _now()
        command = AuthoringJobCommand(
            job_id=job_id,
            domain="code",
            author=author_plan.selection,
            request={
                "prompt": request.prompt,
                "difficulty": request.difficulty,
                "num_distractors": request.num_distractors,
            },
        )
        job = StoredJob(
            job_id=job_id,
            owner_id=owner_id,
            request=AuthoringRequestRecord(
                prompt=request.prompt,
                difficulty=request.difficulty,
                num_distractors=request.num_distractors,
                domain="code",
                model_profile=profile_id,
                connection_id=connection_id,
                model=model,
                judges=request.judges,
            ),
            provider=provider,
            model=model,
            connection_id=connection_id,
            library_revision=LIBRARY_REVISION,
            state="queued",
            created_at=now,
            updated_at=now,
        )
        try:
            await asyncio.to_thread(self.store.create_job, job)
        except Exception:
            await self.runner.release(owner_id)
            raise

        worker_channel: WorkerChannel | None = None
        if self._remote_mode:
            raw_token = secrets.token_urlsafe(32)
            nomad_job_id = f"{AUTHORING_JOB_PREFIX}{job_id}"
            expires_at = now + timedelta(
                seconds=self.settings.worker_token_ttl_seconds
            )
            try:
                job = await asyncio.to_thread(
                    self.store.set_worker_channel,
                    job_id,
                    hash_worker_token(raw_token),
                    expires_at,
                    nomad_job_id,
                )
            except Exception:
                await asyncio.to_thread(
                    self.store.fail_job,
                    job_id,
                    PublicAuthoringError(
                        code="AUTHORING_FAILED",
                        message="Authoring could not be scheduled.",
                    ),
                )
                await self.runner.release(owner_id)
                raise
            worker_channel = self._worker_channel(raw_token)
            self._worker_plans[job_id] = plan
            self._worker_commands[job_id] = command.model_dump(mode="json")

        try:
            self.runner.start_job(
                job_id,
                command,
                plan,
                owner_id,
                worker_channel=worker_channel,
            )
        except Exception:
            self._discard_worker_context(job_id)
            await asyncio.to_thread(
                self.store.fail_job,
                job_id,
                PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message="Authoring could not be scheduled.",
                ),
            )
            await self.runner.release(owner_id)
            raise
        return job

    async def list_jobs(self, owner_id: UUID) -> list[StoredJob]:
        return await asyncio.to_thread(self.store.list_jobs, owner_id)

    async def get_job(self, owner_id: UUID, job_id: UUID) -> StoredJob:
        return await asyncio.to_thread(self.store.get_owned_job, job_id, owner_id)

    async def wait_for_idle(self) -> None:
        """Expose deterministic completion to focused tests."""
        await self.runner.wait_for_idle()

    async def worker_context(self, token: str) -> dict[str, Any]:
        """Return the command and decrypted plan for one authenticated worker."""
        job = await self._require_worker_job(token)
        if job.state not in {"queued", "running"}:
            raise AuthoringConflict("Authoring job is already finished.")
        command = self._worker_commands.get(job.job_id)
        plan = self._worker_plans.get(job.job_id)
        if command is None or plan is None:
            raise AuthoringConflict("Authoring worker context is no longer available.")
        await asyncio.to_thread(self.store.mark_worker_context_issued, job.job_id)
        return {
            "job_id": str(job.job_id),
            "library_revision": LIBRARY_REVISION,
            "timeout_seconds": self.settings.job_timeout_seconds,
            "command": command,
            "plan": self._context_plan(plan),
        }

    async def update_worker_progress(
        self, token: str, progress: AuthoringProgress
    ) -> None:
        """Persist best-effort worker progress without exposing the raw token."""
        job = await self._require_worker_job(token)
        if job.state not in {"queued", "running"}:
            return
        await asyncio.to_thread(
            self.store.update_progress_by_token,
            hash_worker_token(token),
            progress,
        )

    async def complete_worker_result(
        self, token: str, report: AuthoringResult
    ) -> None:
        """Resolve the runner future; the runner persists the single result."""
        job = await self._require_worker_job(token)
        if job.state == "completed":
            await self._require_matching_worker_report(job.job_id, report)
            return
        if job.state == "failed":
            raise AuthoringConflict("Authoring job already failed.")
        resolution = self.runner.resolve(job.job_id, report)
        if resolution in {"accepted", "duplicate"}:
            return
        refreshed = await asyncio.to_thread(self.store.get_job, job.job_id)
        if refreshed.state == "completed":
            await self._require_matching_worker_report(job.job_id, report)
            return
        raise AuthoringConflict("Authoring job is not awaiting a worker result.")

    async def fail_worker_job(self, token: str, error: PublicAuthoringError) -> None:
        """Reject the runner future with a worker-reported fatal error."""
        job = await self._require_worker_job(token)
        if job.state in {"completed", "failed"}:
            raise AuthoringConflict("Authoring job is already finished.")
        resolution = self.runner.reject(job.job_id, error)
        if resolution == "accepted":
            return
        refreshed = await asyncio.to_thread(self.store.get_job, job.job_id)
        if refreshed.state in {"completed", "failed"}:
            raise AuthoringConflict("Authoring job is already finished.")
        raise AuthoringConflict("Authoring job is not awaiting a worker result.")

    async def _require_worker_job(self, token: str) -> StoredJob:
        try:
            job = await asyncio.to_thread(
                self.store.get_job_by_worker_token, hash_worker_token(token)
            )
        except AuthoringNotFound as exc:
            raise AuthoringWorkerUnauthorized() from exc
        expires_at = job.worker_token_expires_at
        if expires_at is None or expires_at <= _now():
            raise AuthoringWorkerUnauthorized()
        return job

    async def _require_matching_worker_report(
        self, job_id: UUID, report: AuthoringResult
    ) -> None:
        try:
            stored = await asyncio.to_thread(self.store.read_report, job_id)
        except AuthoringConflict as exc:
            raise AuthoringConflict(
                "Authoring job is already finished with different evidence."
            ) from exc
        if stored.model_dump(mode="json") != report.model_dump(mode="json"):
            raise AuthoringConflict(
                "Authoring job is already completed with a different result."
            )

    def _worker_channel(self, token: str) -> WorkerChannel:
        base_url = self.settings.nomad_callback_base_url
        if not base_url:
            from edcraft_backend.config import settings as app_settings

            base_url = app_settings.nomad.callback_base_url
        prefix = f"{base_url.rstrip('/')}/ai-authoring/worker/{token}"
        return WorkerChannel(
            token=token,
            context_url=f"{prefix}/context",
            progress_url=f"{prefix}/progress",
            callback_url=f"{prefix}/result",
            error_url=f"{prefix}/error",
        )

    def _context_plan(self, plan: JobModelPlan) -> dict[str, dict[str, Any]]:
        return {
            "author": self._role_context(plan.author),
            "difficulty": self._role_context(plan.difficulty),
            "relevance": self._role_context(plan.relevance),
        }

    def _role_context(self, role: ModelPlan) -> dict[str, Any]:
        credentials = role.credentials or self._server_role_credentials(role)
        return {
            "provider": credentials.provider,
            "model": credentials.model,
            "api_key": credentials.api_key,
            "base_url": credentials.base_url,
            "timeout_seconds": credentials.timeout_seconds,
        }

    def _server_role_credentials(self, role: ModelPlan) -> ProviderCredentials:
        """Read a server-profile key from the environment only for Nomad mode."""
        provider = role.selection.provider
        model = role.selection.model
        if model is None:
            raise AuthoringValidationError("Choose a model for the server profile.")
        prefix = provider.upper()
        api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
        if not api_key:
            raise AuthoringValidationError(
                f"Server credentials for {provider} are not configured."
            )
        base_url = os.getenv(f"{prefix}_BASE_URL") or None
        return ProviderCredentials(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url.strip() if base_url else None,
        )

    async def review(self, owner_id: UUID, job_id: UUID) -> AuthoringReviewResponse:
        """Return sanitized evidence sufficient for an educator decision."""
        job = await self.get_job(owner_id, job_id)
        report: AuthoringResult | None = None
        if job.state in {"completed", "failed"}:
            try:
                report = await asyncio.to_thread(self.store.read_report, job_id)
            except AuthoringConflict:
                if job.artifact_sha256 is not None or job.artifact_id is not None:
                    raise
                report = None

        artifact: ValidatedCodeTemplate | None = None
        if job.artifact_sha256 and job.artifact_id:
            _, artifact = await self._verified_artifact(job)
            require_bound_report(
                job,
                expected_artifact_sha256=job.artifact_sha256,
                report=report,
            )

        attempts = [
            self._project_attempt(attempt) for attempt in (report.attempts if report else [])
        ]
        checks = attempts[-1].checks if attempts else []
        candidate = attempts[-1].candidate if attempts else None
        limits = self._limits(artifact.validation) if artifact is not None else None
        summary = job.result
        return AuthoringReviewResponse(
            job_id=job.job_id,
            state=job.state,
            artifact_id=job.artifact_id,
            artifact_sha256=job.artifact_sha256,
            library_revision=job.library_revision,
            request=AuthoringRequestView.model_validate(job.request.model_dump()),
            template=(
                ReviewTemplate(
                    template_id=artifact.template.template_id,
                    candidate=artifact.template,
                )
                if artifact is not None
                else None
            ),
            checks=checks,
            attempts=attempts,
            fixed_plan=list(report.fixed_plan) if report else [],
            validation_limits=limits,
            candidate=candidate,
            candidate_is_historical_attempt=candidate is not None,
            difficulty=summary.difficulty if summary else None,
            relevance=summary.relevance if summary else None,
            warnings=list(summary.warnings) if summary else [],
            approval=self._approval_view(job.approval),
            error=job.error,
        )

    async def preview(self, owner_id: UUID, job_id: UUID, seed: int) -> PreviewResponse:
        """Generate one deterministic draft locally without model or MCP calls."""
        job = await self.get_job(owner_id, job_id)
        await self._require_review_evidence(job)
        _, artifact = await self._verified_artifact(job)
        question = self._generate(artifact, seed)
        if job.artifact_id is None or job.artifact_sha256 is None:
            raise AuthoringConflict("The validated artifact is unavailable.")
        return PreviewResponse(
            job_id=job.job_id,
            artifact_id=job.artifact_id,
            artifact_sha256=job.artifact_sha256,
            seed=seed,
            question=question,
        )

    async def decide(
        self,
        owner_id: UUID,
        job_id: UUID,
        request: ApprovalRequest,
    ) -> ApprovalResponse:
        """Persist the current decision when the exact artifact is usable."""
        job = await self.get_job(owner_id, job_id)
        await self._require_review_evidence(job)
        if request.artifact_sha256 != job.artifact_sha256:
            raise AuthoringConflict("The artifact changed; refresh the review before deciding.")
        await self._verified_artifact(job)
        approval = ApprovalRecord(
            decision=request.decision,
            artifact_sha256=request.artifact_sha256,
            reviewer_id=owner_id,
            reviewer=str(owner_id),
            note=request.note,
            decided_at=_now(),
        )
        updated = await asyncio.to_thread(self.store.save_approval, job_id, approval)
        view = self._approval_view(updated.approval)
        if view is None:
            raise AuthoringConflict("The approval decision could not be saved.")
        return ApprovalResponse(job_id=job_id, **view.model_dump())

    async def generate(
        self,
        owner_id: UUID,
        job_id: UUID,
        request: GenerateRequest,
    ) -> GeneratedQuestionsBatch:
        """Expand an approved artifact across consecutive deterministic seeds."""
        job = await self.get_job(owner_id, job_id)
        await self._require_review_evidence(job)
        limit = min(self.settings.max_quantity, 50)
        if request.quantity > limit:
            raise AuthoringValidationError(f"Quantity must be at most {limit} for this server.")
        if request.artifact_sha256 != job.artifact_sha256:
            raise AuthoringConflict("The artifact changed; refresh the review before generating.")
        approval = job.approval
        if (
            approval is None
            or approval.decision != "approved"
            or approval.artifact_sha256 != request.artifact_sha256
        ):
            raise AuthoringConflict(
                "The exact artifact must be approved before generating questions."
            )
        _, artifact = await self._verified_artifact(job)

        questions: list[CodeQuestionInstance] = []
        warnings: list[str] = []
        for offset in range(request.quantity):
            seed = request.seed + offset
            try:
                questions.append(self._generate_raw(artifact, seed))
            except (ValidationError, ValueError):
                warnings.append(
                    f"Seed {seed} did not produce a usable question from the "
                    "finite template domain."
                )
        fingerprints = {question.model_dump_json(exclude={"seed"}) for question in questions}
        distinct_count = len(fingerprints)
        duplicates = len(questions) - distinct_count
        if duplicates:
            warnings.append(
                f"{duplicates} duplicate question(s) came from a finite parameter "
                "domain; every retained instance still has a distinct seed."
            )
        if job.artifact_id is None or job.artifact_sha256 is None:
            raise AuthoringConflict("The validated artifact is unavailable.")
        created_at = _now()
        await asyncio.to_thread(
            self.store.save_generation,
            job.job_id,
            GenerationMetadata(
                artifact_sha256=job.artifact_sha256,
                base_seed=request.seed,
                requested_quantity=request.quantity,
                actual_quantity=len(questions),
                distinct_count=distinct_count,
                duplicates=duplicates,
                warnings=warnings,
                created_at=created_at,
            ),
        )
        return GeneratedQuestionsBatch(
            job_id=job.job_id,
            artifact_id=job.artifact_id,
            artifact_sha256=job.artifact_sha256,
            base_seed=request.seed,
            requested_quantity=request.quantity,
            actual_quantity=len(questions),
            distinct_count=distinct_count,
            duplicates=duplicates,
            questions=questions,
            warnings=warnings,
        )

    def job_status(self, job: StoredJob) -> AuthoringJobStatusResponse:
        return AuthoringJobStatusResponse(
            job_id=job.job_id,
            state=job.state,
            progress=job.progress,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            approval=self._approval_view(job.approval),
            last_generation=job.last_generation,
        )

    def job_summary(self, job: StoredJob) -> AuthoringJobSummary:
        return AuthoringJobSummary(
            job_id=job.job_id,
            state=job.state,
            result_status=job.result.status if job.result is not None else None,
            created_at=job.created_at,
            prompt_preview=_truncate(job.request.prompt, 160),
            difficulty=job.request.difficulty,
            approval=self._approval_view(job.approval),
        )

    def _profile(self, profile_id: str) -> AuthoringProfileSettings:
        for profile in self.settings.model_profiles:
            if profile.id == profile_id:
                return profile
        raise AuthoringValidationError("Unknown model profile.")

    def _judge_model_plan(
        self,
        selection: AuthoringModelSelection | None,
        *,
        default: ModelPlan,
        credentials: ProviderCredentials | None,
        label: str,
    ) -> ModelPlan:
        """Resolve one optional advisory override, inheriting the author by default."""
        if selection is None:
            return default
        if selection.connection_id is not None:
            if credentials is None:
                raise AuthoringValidationError(
                    f"{label} model connection is unavailable."
                )
            return ModelPlan(
                selection=TemplateProviderSelection(
                    provider=credentials.provider, model=credentials.model
                ),
                credentials=credentials,
            )
        if selection.model_profile is None:
            raise AuthoringValidationError(f"Choose a {label.lower()} model.")
        profile = self._profile(selection.model_profile)
        return ModelPlan(
            selection=TemplateProviderSelection(
                provider=profile.provider, model=profile.model
            )
        )

    async def _require_review_evidence(self, job: StoredJob) -> None:
        """Require the saved report and bind its artifact to the job."""
        if job.artifact_sha256 is None:
            raise AuthoringConflict(
                "Only a technically checked artifact with saved review evidence "
                "can be previewed, approved or generated."
            )
        report = await asyncio.to_thread(self.store.read_report, job.job_id)
        require_reusable_evidence(
            job,
            expected_artifact_sha256=job.artifact_sha256,
            report=report,
        )

    async def _verified_artifact(self, job: StoredJob) -> tuple[bytes, ValidatedCodeTemplate]:
        if job.artifact_sha256 is None or job.artifact_id is None:
            raise AuthoringConflict("The validated artifact is unavailable.")
        data = await asyncio.to_thread(self.store.read_artifact, job.job_id)
        digest = hashlib.sha256(data).hexdigest()
        if digest != job.artifact_sha256:
            raise AuthoringConflict(
                "The stored artifact failed its integrity check; refresh the review."
            )
        try:
            artifact = ValidatedCodeTemplate.model_validate_json(data)
        except (ValidationError, ValueError) as exc:
            raise AuthoringConflict("The stored artifact is not usable.") from exc
        if artifact.artifact_id != job.artifact_id:
            raise AuthoringConflict("The stored artifact identity does not match.")
        return data, artifact

    @staticmethod
    def _generate_raw(artifact: ValidatedCodeTemplate, seed: int) -> CodeQuestionInstance:
        result = TemplateApplication().generate_question(
            artifact, domain=create_domain("code"), seed=seed
        )
        return CodeQuestionInstance.model_validate(result.model_dump())

    @classmethod
    def _generate(cls, artifact: ValidatedCodeTemplate, seed: int) -> CodeQuestionInstance:
        try:
            return cls._generate_raw(artifact, seed)
        except (ValidationError, ValueError) as exc:
            raise AuthoringConflict(
                "The artifact could not produce a question for this seed."
            ) from exc

    @staticmethod
    def _project_attempt(attempt: GenerationAttempt) -> ReviewAttempt:
        return ReviewAttempt(
            number=attempt.number,
            candidate=_candidate(attempt.candidate),
            pending=list(attempt.pending),
            complete=attempt.complete,
            passed=attempt.passed,
            checks=[_project_check(execution) for execution in attempt.executions],
        )

    @staticmethod
    def _limits(validation: TemplateValidationSummary) -> ValidationLimits:
        return ValidationLimits(
            validator_version=validation.validator_version,
            checking_mode=validation.checking_mode,
            examples_checked=validation.examples_checked,
            execution_calls=validation.execution_calls,
            max_examples=validation.max_examples,
            max_execution_calls=validation.max_execution_calls,
            timeout_seconds=validation.timeout_seconds,
            seed=validation.seed,
        )

    @staticmethod
    def _approval_view(approval: ApprovalRecord | None) -> ApprovalView | None:
        if approval is None:
            return None
        return ApprovalView(
            decision=approval.decision,
            artifact_sha256=approval.artifact_sha256,
            reviewer=approval.reviewer,
            note=approval.note,
            decided_at=approval.decided_at,
        )


def _project_check(execution: Any) -> ReviewCheck:
    evidence = execution.evidence
    if evidence is not None:
        status = evidence.status
        findings = [
            ReviewFinding(
                code=finding.code,
                message=_truncate(finding.message, MAX_FINDING_TEXT),
                field=finding.field,
            )
            for finding in evidence.findings
        ]
        advisory = execution.advisory or evidence.details.get("advisory") is True
    else:
        status = "error" if execution.error is not None else "skipped"
        findings = []
        advisory = bool(execution.advisory)
    return ReviewCheck(
        tool=execution.tool,
        status=status,
        advisory=advisory,
        findings=findings,
        error="Check did not complete." if execution.error is not None else None,
    )


def _candidate(value: Any) -> CodeTemplateCandidate | None:
    try:
        return CodeTemplateCandidate.model_validate(value)
    except (ValidationError, ValueError):
        return None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "AiAuthoringService",
    "AuthoringNotFound",
    "AuthoringStorageUnavailable",
    "AuthoringValidationError",
    "AuthoringWorkerUnauthorized",
    "LIBRARY_REVISION",
    "hash_worker_token",
]
