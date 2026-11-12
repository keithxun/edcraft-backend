"""Business rules for owner-scoped AI authoring, review and generation."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from edcraft_validator.application.authoring_api import AuthoringJobCommand
from edcraft_validator.application.authoring_contracts import (
    AuthoringResult,
    GenerationAttempt,
)
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.artifact_contracts import ValidatedTemplateArtifact
from edcraft_validator.domains.code.code_types import (
    DEFAULT_DISTRACTORS,
    MAX_DISTRACTORS,
    MIN_DISTRACTORS,
)
from edcraft_validator.domains.domain_contract import DomainModule
from edcraft_validator.domains.domain_registry import available_domains, create_domain
from edcraft_validator.llm.judge_configuration import load_judge_settings
from pydantic import ValidationError

from edcraft_backend.config.settings import (
    AuthoringProfileSettings,
    AuthoringSettings,
    CheckPilotSettings,
)
from edcraft_backend.schemas.ai_authoring import (
    ApprovalRequest,
    ApprovalResponse,
    AuthoringCapabilitiesResponse,
    AuthoringJobStatusResponse,
    AuthoringJobSummary,
    AuthoringModelProfile,
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
from edcraft_backend.services.ai_authoring_lifecycle import AuthoringLifecycle
from edcraft_backend.services.ai_authoring_runner import run_library_authoring_job
from edcraft_backend.services.ai_authoring_runtime import (
    AuthoringValidationError,
    JobExecutor,
    JobRuntime,
)
from edcraft_backend.services.ai_authoring_store import (
    ApprovalRecord,
    AuthoringConflict,
    AuthoringNotFound,
    AuthoringRequestRecord,
    AuthoringStorageUnavailable,
    AuthoringStore,
    EvidenceRequirement,
    StoredJob,
)
from edcraft_backend.services.provider_endpoint import ResolvedConnection

ApprovalView = StoredReviewerApproval

logger = logging.getLogger(__name__)

# Must match [tool.uv.sources].edcraft-validator.rev in pyproject.toml.
LIBRARY_REVISION = "61ce11ff1d6c1cd45badeaa440ea7da51288dfb9"
MAX_FINDING_TEXT = 2_000


class AiAuthoringService:
    """Thin integration layer over the pinned AI library and file store."""

    def __init__(
        self,
        settings: AuthoringSettings,
        *,
        store: AuthoringStore | None = None,
        executor: JobExecutor | None = None,
        check_pilot: CheckPilotSettings | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or AuthoringStore(settings)
        self._executor: JobExecutor = executor or run_library_authoring_job
        self.lifecycle = AuthoringLifecycle(self.store, settings)
        self._check_pilot = check_pilot or CheckPilotSettings()

    async def startup(self) -> None:
        """Prepare storage, reconcile interruptions and bind the lifecycle."""
        await self.lifecycle.start()

    async def shutdown(self, timeout: float | None = None) -> None:
        """Allow admitted work a bounded chance to finish before process exit."""
        await self.lifecycle.shutdown(
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
            enabled_domains=list(available_domains()),
            difficulty_labels=["easy", "medium", "hard"],
            model_profiles=profiles,
            # Disabled connections stay visible in settings but cannot write questions.
            connections=[item for item in (connections or []) if item.enabled],
            judge=judge,
            max_quantity=min(self.settings.max_quantity, 50),
            min_distractors=MIN_DISTRACTORS,
            max_distractors=MAX_DISTRACTORS,
            default_distractors=DEFAULT_DISTRACTORS,
            check_pilot_timeout_seconds=(
                self._check_pilot.admission_timeout_seconds
                + self._check_pilot.request_timeout_seconds
            ),
        )

    async def submit(
        self,
        owner_id: UUID,
        request: SubmitAuthoringJobRequest,
        *,
        author_connection: ResolvedConnection | None = None,
        difficulty_connection: ResolvedConnection | None = None,
        relevance_connection: ResolvedConnection | None = None,
    ) -> StoredJob:
        """Admit and persist one queued authoring job."""
        domain_request = _domain_request(request)
        runtime = JobRuntime.prepare(
            request,
            profile_lookup=self._profile,
            executor=self._executor,
            author_connection=author_connection,
            difficulty_connection=difficulty_connection,
            relevance_connection=relevance_connection,
        )
        author = runtime.author
        provider = author.selection.provider
        model = author.model
        connection_id = request.connection_id
        profile_id = request.model_profile if connection_id is None else None

        job_id = uuid4()
        now = _now()
        command = AuthoringJobCommand(
            job_id=job_id,
            domain=request.domain,
            author=author.selection,
            request=domain_request.model_dump(mode="json"),
        )
        job = StoredJob(
            job_id=job_id,
            owner_id=owner_id,
            request=AuthoringRequestRecord(
                prompt=request.prompt,
                difficulty=request.difficulty,
                num_distractors=request.num_distractors,
                domain=request.domain,
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
        return await self.lifecycle.submit(job, command, runtime)

    async def list_jobs(self, owner_id: UUID) -> list[StoredJob]:
        return await asyncio.to_thread(self.store.list_jobs, owner_id)

    async def get_job(self, owner_id: UUID, job_id: UUID) -> StoredJob:
        return await asyncio.to_thread(self.store.get_owned_job, job_id, owner_id)

    async def wait_for_idle(self) -> None:
        """Expose deterministic completion to focused tests."""
        await self.lifecycle.wait_for_idle()

    async def review(self, owner_id: UUID, job_id: UUID) -> AuthoringReviewResponse:
        """Return sanitized evidence sufficient for an educator decision."""
        job = await self.get_job(owner_id, job_id)
        domain = _require_domain(job.request.domain)
        report: AuthoringResult | None = None
        artifact: ValidatedTemplateArtifact | None = None
        if job.artifact_sha256 and job.artifact_id:
            evidence = await asyncio.to_thread(
                self.store.read_evidence,
                job_id,
                owner_id=owner_id,
                requirement=EvidenceRequirement.REVIEW,
            )
            job = evidence.job
            report = evidence.report
            artifact = _validated_artifact(domain, evidence.artifact_bytes)
        elif job.state in {"completed", "failed"}:
            try:
                report = await asyncio.to_thread(self.store.read_report, job_id)
            except AuthoringConflict:
                report = None

        attempts = [
            self._project_attempt(attempt) for attempt in (report.attempts if report else [])
        ]
        checks = attempts[-1].checks if attempts else []
        candidate = attempts[-1].candidate if attempts else None
        review = domain.review_view(artifact) if artifact is not None else None
        template = (
            ReviewTemplate(
                template_id=review.template_id,
                candidate=review.candidate,
                language=review.language,
            )
            if review is not None
            else None
        )
        limits = _limits(review.limits) if review is not None else None
        summary = job.result
        return AuthoringReviewResponse(
            job_id=job.job_id,
            state=job.state,
            domain=job.request.domain,
            artifact_id=job.artifact_id,
            artifact_sha256=job.artifact_sha256,
            library_revision=job.library_revision,
            request=AuthoringRequestView.model_validate(job.request.model_dump()),
            template=template,
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
        evidence = await asyncio.to_thread(
            self.store.read_evidence,
            job_id,
            owner_id=owner_id,
            requirement=EvidenceRequirement.REUSABLE,
        )
        job = evidence.job
        question = self._generate(evidence.artifact, _require_domain(job.request.domain), seed)
        if job.artifact_id is None or job.artifact_sha256 is None:
            raise AuthoringConflict("The validated artifact is unavailable.")
        return PreviewResponse(
            job_id=job.job_id,
            domain=job.request.domain,
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
        await asyncio.to_thread(
            self.store.read_evidence,
            job_id,
            owner_id=owner_id,
            requirement=EvidenceRequirement.REUSABLE,
            expected_artifact_sha256=request.artifact_sha256,
        )
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
        limit = min(self.settings.max_quantity, 50)
        if request.quantity > limit:
            raise AuthoringValidationError(f"Quantity must be at most {limit} for this server.")
        evidence = await asyncio.to_thread(
            self.store.read_evidence,
            job_id,
            owner_id=owner_id,
            requirement=EvidenceRequirement.APPROVED,
            expected_artifact_sha256=request.artifact_sha256,
        )
        job = evidence.job
        artifact = evidence.artifact
        domain = _require_domain(job.request.domain)

        questions: list[dict[str, Any]] = []
        warnings: list[str] = []
        for offset in range(request.quantity):
            seed = request.seed + offset
            try:
                questions.append(self._generate_raw(artifact, domain, seed))
            except (ValidationError, ValueError):
                warnings.append(
                    f"Seed {seed} did not produce a usable question from the "
                    "finite template domain."
                )
        fingerprints = {
            json.dumps(_without_seed(question), sort_keys=True) for question in questions
        }
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
            domain=job.request.domain,
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
            domain=job.request.domain,
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

    @staticmethod
    def _generate_raw(
        artifact: ValidatedTemplateArtifact, domain: DomainModule, seed: int
    ) -> dict[str, Any]:
        result = TemplateApplication().generate_question(artifact, domain=domain, seed=seed)
        question = domain.question_model.model_validate(result)
        return cast(dict[str, Any], question.model_dump(mode="json"))

    @classmethod
    def _generate(
        cls, artifact: ValidatedTemplateArtifact, domain: DomainModule, seed: int
    ) -> dict[str, Any]:
        try:
            return cls._generate_raw(artifact, domain, seed)
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


def _domain_request(request: SubmitAuthoringJobRequest) -> Any:
    """Validate the portable request fields through the selected domain."""
    try:
        domain = create_domain(request.domain)
        return domain.request_model.model_validate(
            {
                "prompt": request.prompt,
                "difficulty": request.difficulty,
                "num_distractors": request.num_distractors,
            }
        )
    except (ValidationError, ValueError) as exc:
        raise AuthoringValidationError("The request does not satisfy the selected domain.") from exc


def _require_domain(name: str) -> DomainModule:
    try:
        return create_domain(name)
    except ValueError as exc:
        raise AuthoringConflict("The domain for this authoring job is not available.") from exc


def _validated_artifact(domain: DomainModule, artifact_bytes: bytes) -> ValidatedTemplateArtifact:
    try:
        return domain.validated_model.model_validate_json(artifact_bytes)
    except (ValidationError, ValueError) as exc:
        raise AuthoringConflict("The stored artifact is not usable.") from exc


def _template_view(
    artifact: ValidatedTemplateArtifact | None, domain: DomainModule
) -> ReviewTemplate | None:
    if artifact is None:
        return None
    template = getattr(artifact, "template", None)
    template_id = getattr(template, "template_id", None)
    candidate = _candidate(template)
    if template_id is None or candidate is None:
        return None
    return ReviewTemplate(
        template_id=template_id,
        candidate=candidate,
        language=_template_language(domain),
    )


def _template_language(domain: DomainModule) -> str | None:
    """Read the optional review label without requiring it from older domains."""
    field = "template_language"
    return getattr(domain, field, None)


_LIMIT_FIELDS = (
    "examples_checked",
    "execution_calls",
    "max_examples",
    "max_execution_calls",
    "timeout_seconds",
    "combinations_checked",
    "max_combinations",
)


def _limits(recorded: dict[str, Any] | None) -> ValidationLimits | None:
    """Project the domain-owned validation summary into the response shape."""
    if recorded is None:
        return None
    if not {"validator_version", "checking_mode", "seed"} <= recorded.keys():
        return None
    return ValidationLimits(
        validator_version=recorded["validator_version"],
        checking_mode=recorded["checking_mode"],
        seed=recorded["seed"],
        **{field: recorded.get(field) for field in _LIMIT_FIELDS},
    )


def _candidate(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if dump is None:
        return None
    try:
        return cast(dict[str, Any], dump(mode="json"))
    except (TypeError, ValueError):
        return None


def _without_seed(question: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in question.items() if key != "seed"}


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
    "LIBRARY_REVISION",
]
