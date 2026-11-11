"""Request and response contracts for AI template authoring."""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from edcraft_validator.application.authoring_api import (
    AuthoringSummary,
    PublicAuthoringError,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringProgress,
    DifficultyOutcome,
    RelevanceOutcome,
)
from edcraft_validator.domains.code.code_types import (
    DEFAULT_DISTRACTORS,
    MAX_DISTRACTORS,
    MIN_DISTRACTORS,
)
from edcraft_validator.domains.domain_registry import available_domains
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edcraft_backend.schemas.provider_connection import ProviderConnectionResponse

Difficulty = Literal["easy", "medium", "hard"]
MAX_SEED = 2_147_483_647
StrictSeed = Annotated[int, Field(strict=True, ge=0, le=MAX_SEED)]
StrictQuantity = Annotated[int, Field(strict=True, ge=1, le=50)]
DistractorCount = Annotated[int, Field(strict=True, ge=MIN_DISTRACTORS, le=MAX_DISTRACTORS)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ErrorResponse(BaseModel):
    """Sanitized error envelope used by all AI authoring endpoints."""

    detail: str
    error_type: str | None = None


class AuthoringModelProfile(BaseModel):
    """Safe display metadata for a server-configured authoring model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=120)


class JudgeDisplay(BaseModel):
    """Safe display metadata for the separately configured advisory judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=120)


class AuthoringCapabilitiesResponse(BaseModel):
    """Enabled domains and server-configured model choices."""

    enabled_domains: list[str]
    difficulty_labels: list[Difficulty]
    model_profiles: list[AuthoringModelProfile]
    connections: list[ProviderConnectionResponse] = Field(default_factory=list)
    judge: JudgeDisplay | None = None
    max_quantity: int
    min_distractors: int
    max_distractors: int
    default_distractors: int


class AuthoringModelSelection(BaseModel):
    """Exactly one configured model source for one authoring role."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Legacy server-configured profile. New clients use connection_id + model.
    model_profile: str | None = Field(default=None, min_length=1, max_length=64)
    connection_id: UUID | None = Field(default=None, strict=False)
    model: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def select_one_model_source(self) -> "AuthoringModelSelection":
        using_connection = self.connection_id is not None
        if using_connection and self.model_profile is not None:
            raise ValueError("Choose either a model connection or a server model profile.")
        if not using_connection and self.model_profile is None:
            raise ValueError("Choose a model connection or a server model profile.")
        if using_connection and self.model is None:
            raise ValueError("Choose a model for this connection.")
        if not using_connection and self.model is not None:
            raise ValueError("A model requires a model connection.")
        return self


class AuthoringJudgeSelections(BaseModel):
    """Optional per-check model overrides; omitted roles use the question model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    difficulty: AuthoringModelSelection | None = None
    relevance: AuthoringModelSelection | None = None


class SubmitAuthoringJobRequest(AuthoringModelSelection):
    """One request to author a reusable question template."""

    domain: str = "code"
    prompt: str = Field(min_length=1, max_length=4000)
    difficulty: Difficulty
    num_distractors: DistractorCount = DEFAULT_DISTRACTORS
    judges: AuthoringJudgeSelections | None = None

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("domain")
    @classmethod
    def require_registered_domain(cls, value: str) -> str:
        if value not in available_domains():
            supported = ", ".join(available_domains())
            raise ValueError(f"Unsupported domain {value!r}; choose one of: {supported}")
        return value


class AuthoringJobSubmittedResponse(BaseModel):
    """Accepted job identity and polling URL."""

    job_id: UUID
    status_url: str


class StoredReviewerApproval(BaseModel):
    """Current human decision bound to one exact artifact digest."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]
    artifact_sha256: Sha256
    reviewer: str
    note: str | None = None
    decided_at: datetime


class GenerationMetadata(BaseModel):
    """Minimal server-persisted record of the latest deterministic batch."""

    model_config = ConfigDict(extra="forbid")

    artifact_sha256: Sha256
    base_seed: StrictSeed
    requested_quantity: StrictQuantity
    actual_quantity: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class AuthoringJobStatusResponse(BaseModel):
    """Owner-scoped authoring job state and safe public summary."""

    job_id: UUID
    state: Literal["queued", "running", "completed", "failed"]
    progress: AuthoringProgress | None = None
    result: AuthoringSummary | None = None
    error: PublicAuthoringError | None = None
    created_at: datetime
    approval: StoredReviewerApproval | None = None
    last_generation: GenerationMetadata | None = None


class AuthoringJobSummary(BaseModel):
    """Small owned-job item used to reopen saved work."""

    job_id: UUID
    domain: str
    state: Literal["queued", "running", "completed", "failed"]
    result_status: Literal["checked", "needs_review", "error"] | None = None
    created_at: datetime
    prompt_preview: str
    difficulty: Difficulty
    approval: StoredReviewerApproval | None = None


class AuthoringRequestView(BaseModel):
    """Original author request returned during review."""

    prompt: str
    difficulty: Difficulty
    num_distractors: DistractorCount
    domain: str
    model_profile: str | None = None
    connection_id: UUID | None = None
    model: str | None = None
    judges: AuthoringJudgeSelections | None = None


class ReviewFinding(BaseModel):
    """One safe check finding; provider details and raw exceptions are omitted."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    field: str | None = None


class ReviewCheck(BaseModel):
    """One technical or advisory check projected from the retained report."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    status: Literal["passed", "failed", "error", "skipped"]
    advisory: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    error: str | None = None


class ReviewAttempt(BaseModel):
    """One historical candidate attempt with its check findings."""

    model_config = ConfigDict(extra="forbid")

    number: int
    candidate: dict[str, Any] | None = None
    pending: list[str] = Field(default_factory=list)
    complete: bool
    passed: bool
    checks: list[ReviewCheck] = Field(default_factory=list)


class ReviewTemplate(BaseModel):
    """Exact technically checked template an educator should review."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    candidate: dict[str, Any]
    language: str | None = None


class ValidationLimits(BaseModel):
    """Check budget recorded by the AI library, whichever mode it used.

    Code templates record a sampled budget; math templates record an exhaustive
    bounded-combination budget. Fields for the other mode stay null so clients
    can keep reading one stable response shape.
    """

    model_config = ConfigDict(extra="forbid")

    validator_version: str
    checking_mode: str
    examples_checked: int | None = None
    execution_calls: int | None = None
    max_examples: int | None = None
    max_execution_calls: int | None = None
    timeout_seconds: float | None = None
    combinations_checked: int | None = None
    max_combinations: int | None = None
    seed: int


class AuthoringReviewResponse(BaseModel):
    """Safe, complete-enough review envelope for one owned job."""

    job_id: UUID
    state: Literal["queued", "running", "completed", "failed"]
    artifact_id: str | None = None
    artifact_sha256: Sha256 | None = None
    library_revision: str
    request: AuthoringRequestView
    template: ReviewTemplate | None = None
    checks: list[ReviewCheck] = Field(default_factory=list)
    attempts: list[ReviewAttempt] = Field(default_factory=list)
    fixed_plan: list[str] = Field(default_factory=list)
    validation_limits: ValidationLimits | None = None
    candidate: dict[str, Any] | None = None
    candidate_is_historical_attempt: bool = False
    difficulty: DifficultyOutcome | None = None
    relevance: RelevanceOutcome | None = None
    warnings: list[str] = Field(default_factory=list)
    approval: StoredReviewerApproval | None = None
    error: PublicAuthoringError | None = None


class PreviewRequest(BaseModel):
    """Deterministic local preview seed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    seed: StrictSeed


class PreviewResponse(BaseModel):
    """One deterministic question generated from the exact artifact."""

    job_id: UUID
    domain: str
    artifact_id: str
    artifact_sha256: Sha256
    seed: StrictSeed
    question: dict[str, Any]


class ApprovalRequest(BaseModel):
    """Human decision bound to the exact artifact digest shown during review."""

    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_sha256: Sha256
    decision: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("note")
    @classmethod
    def reject_blank_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ApprovalResponse(StoredReviewerApproval):
    """Persisted decision returned to the reviewer."""

    job_id: UUID


class GenerateRequest(BaseModel):
    """Synchronous deterministic batch request for an approved artifact."""

    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_sha256: Sha256
    seed: StrictSeed
    quantity: StrictQuantity

    @model_validator(mode="after")
    def validate_seed_range(self) -> "GenerateRequest":
        if self.seed + self.quantity - 1 > MAX_SEED:
            raise ValueError("seed plus quantity exceeds the supported seed range")
        return self


class GeneratedQuestionsBatch(BaseModel):
    """Downloadable JSON batch with canonical question instances."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal["1"] = "1"
    job_id: UUID
    domain: str
    artifact_id: str
    artifact_sha256: Sha256
    base_seed: StrictSeed
    requested_quantity: StrictQuantity
    actual_quantity: int
    distinct_count: int
    duplicates: int
    questions: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ApprovalRequest",
    "ApprovalResponse",
    "AuthoringCapabilitiesResponse",
    "AuthoringJobStatusResponse",
    "AuthoringJobSubmittedResponse",
    "AuthoringJobSummary",
    "AuthoringJudgeSelections",
    "AuthoringModelProfile",
    "AuthoringModelSelection",
    "AuthoringReviewResponse",
    "ErrorResponse",
    "GenerateRequest",
    "GeneratedQuestionsBatch",
    "GenerationMetadata",
    "JudgeDisplay",
    "PreviewRequest",
    "PreviewResponse",
    "ReviewCheck",
    "ReviewFinding",
    "StoredReviewerApproval",
    "SubmitAuthoringJobRequest",
    "ValidationLimits",
]
