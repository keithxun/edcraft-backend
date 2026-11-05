"""Owner-scoped, file-backed persistence for AI authoring jobs.

The store is intentionally small: one JSON record per job, plus the exact
validated artifact and the full server-only authoring report. All writes use a
same-directory temporary file, fsync and :func:`os.replace`. Paths are derived
only from canonical UUIDs.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from edcraft_validator.application.authoring_api import (
    AuthoringSummary,
    PublicAuthoringError,
    summarize_authoring,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringProgress,
    AuthoringResult,
)
from edcraft_validator.domains.code.code_types import DEFAULT_DISTRACTORS
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.schemas.ai_authoring import (
    AuthoringJudgeSelections,
    DistractorCount,
    GenerationMetadata,
)

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "completed", "failed"]
SERVER_RESTARTED_MESSAGE = (
    "The server restarted before this job finished. Start a new authoring run."
)


class AuthoringStoreError(EdCraftBaseException):
    """Base class for sanitized authoring persistence failures."""


class AuthoringNotFound(AuthoringStoreError):
    """The job is unknown or does not belong to the requesting owner."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 404)


class AuthoringConflict(AuthoringStoreError):
    """The stored state cannot satisfy the requested operation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 409)


class AuthoringStorageUnavailable(AuthoringStoreError):
    """The configured durable volume cannot be written."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 503)


class AuthoringRequestRecord(BaseModel):
    """Server-side original request plus the selected configured profile."""

    model_config = ConfigDict(extra="forbid")

    domain: Literal["code"] = "code"
    prompt: str
    difficulty: Literal["easy", "medium", "hard"]
    num_distractors: DistractorCount = DEFAULT_DISTRACTORS
    model_profile: str | None = None
    connection_id: UUID | None = None
    model: str | None = None
    judges: AuthoringJudgeSelections | None = None


class ApprovalRecord(BaseModel):
    """Current human decision and its exact artifact binding."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: UUID
    reviewer: str
    note: str | None = None
    decided_at: datetime


class StoredJob(BaseModel):
    """One complete owner-scoped job record (server-only fields included)."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    job_id: UUID
    owner_id: UUID
    request: AuthoringRequestRecord
    provider: str | None = None
    model: str | None = None
    connection_id: UUID | None = None
    library_revision: str
    state: JobState = "queued"
    progress: AuthoringProgress | None = None
    result: AuthoringSummary | None = None
    artifact_id: str | None = None
    artifact_sha256: str | None = None
    approval: ApprovalRecord | None = None
    last_generation: GenerationMetadata | None = None
    error: PublicAuthoringError | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Optional Nomad worker channel. Older records load with all fields None.
    nomad_job_id: str | None = None
    worker_token_hash: str | None = None
    worker_token_expires_at: datetime | None = None
    worker_context_issued_at: datetime | None = None
    worker_revision: str | None = None


def require_bound_report(
    job: StoredJob,
    *,
    expected_artifact_sha256: str,
    report: AuthoringResult | None,
) -> AuthoringResult:
    """Require saved report evidence bound to the job's exact artifact."""
    if (
        job.artifact_id is None
        or job.artifact_sha256 is None
        or job.artifact_sha256 != expected_artifact_sha256
        or job.result is None
        or job.result.artifact_id != job.artifact_id
    ):
        raise AuthoringConflict("Authoring review evidence does not match the job artifact.")
    if report is None or report.status != "checked" or report.artifact is None:
        raise AuthoringConflict("Authoring review evidence is missing or invalid; start a new run.")
    if report.artifact.artifact_id != job.artifact_id:
        raise AuthoringConflict("Authoring review evidence does not match the job artifact.")
    report_digest = hashlib.sha256(report.artifact.model_dump_json().encode()).hexdigest()
    if report_digest != expected_artifact_sha256:
        raise AuthoringConflict("Authoring review evidence no longer matches the job artifact.")
    return report


def require_reusable_evidence(
    job: StoredJob,
    *,
    expected_artifact_sha256: str,
    report: AuthoringResult | None,
) -> AuthoringResult:
    """Require reusable evidence without re-running library checks.

    The library summary remains the authority for technical status. This
    function additionally verifies that the persisted report, exact artifact
    identity and digest still match the job at the point of reuse.
    """
    technical = (
        [check for check in job.result.checks if not check.advisory]
        if job.result is not None
        else []
    )
    if (
        job.state != "completed"
        or job.result is None
        or job.result.status != "checked"
        or job.artifact_id is None
        or job.artifact_sha256 is None
        or not technical
        or any(check.status != "passed" for check in technical)
    ):
        raise AuthoringConflict(
            "Only a technically checked artifact with saved review evidence "
            "can be previewed, approved or generated."
        )
    return require_bound_report(
        job,
        expected_artifact_sha256=expected_artifact_sha256,
        report=report,
    )


class AuthoringStore:
    """Thread-safe durable store using atomic, UUID-derived paths."""

    def __init__(self, settings: AuthoringSettings) -> None:
        self.root = Path(settings.storage_dir)
        self._lock = threading.RLock()

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def ensure_ready(self) -> None:
        """Create the volume layout and fail clearly when it is not writable."""
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.artifacts_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            probe = self.root / f".probe-{uuid.uuid4().hex}"
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, b"ok")
                os.fsync(fd)
            finally:
                os.close(fd)
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise AuthoringStorageUnavailable(
                "AI authoring storage is not writable; check the durable volume."
            ) from exc

    def create_job(self, job: StoredJob) -> StoredJob:
        """Persist a new queued job; UUID collisions are refused."""
        with self._lock:
            path = self._job_path(job.job_id)
            if path.exists():
                raise AuthoringConflict("Authoring job already exists.")
            self._write_bytes(path, job.model_dump_json().encode(), overwrite=False)
            return job

    def get_job(self, job_id: UUID) -> StoredJob:
        """Load one job without applying owner filtering."""
        with self._lock:
            return self._load_job(job_id)

    def get_owned_job(self, job_id: UUID, owner_id: UUID) -> StoredJob:
        """Load a job only when the authenticated owner matches."""
        with self._lock:
            job = self._load_job(job_id)
            if job.owner_id != owner_id:
                raise AuthoringNotFound("Authoring job not found.")
            return job

    def list_jobs(self, owner_id: UUID, *, limit: int = 50) -> list[StoredJob]:
        """Return recent owned jobs, ignoring unreadable records."""
        with self._lock:
            jobs: list[StoredJob] = []
            for path in self.jobs_dir.glob("*.json"):
                try:
                    job = StoredJob.model_validate_json(path.read_bytes())
                except (OSError, ValidationError, ValueError):
                    logger.warning("Ignoring unreadable authoring job record")
                    continue
                if job.owner_id == owner_id:
                    jobs.append(job)
            jobs.sort(key=lambda item: (item.created_at, item.job_id), reverse=True)
            return jobs[:limit]

    def mark_started(self, job_id: UUID) -> StoredJob:
        """Move a queued job to running exactly once."""

        def mutate(job: StoredJob) -> StoredJob:
            if job.state != "queued":
                raise AuthoringConflict("Authoring job is not queued.")
            now = _now()
            return job.model_copy(update={"state": "running", "started_at": now, "updated_at": now})

        return self._mutate(job_id, mutate)

    def set_worker_channel(
        self,
        job_id: UUID,
        token_hash: str,
        expires_at: datetime,
        nomad_job_id: str,
    ) -> StoredJob:
        """Persist only the callback token hash and its Nomad job identity."""

        def mutate(job: StoredJob) -> StoredJob:
            return job.model_copy(
                update={
                    "nomad_job_id": nomad_job_id,
                    "worker_token_hash": token_hash,
                    "worker_token_expires_at": expires_at,
                    "updated_at": _now(),
                }
            )

        return self._mutate(job_id, mutate)

    def get_job_by_worker_token(self, token_hash: str) -> StoredJob:
        """Resolve one job by callback token hash with a constant-time compare."""
        with self._lock:
            job = self._find_by_token_unlocked(token_hash)
            if job is None:
                raise AuthoringNotFound("Authoring worker token is invalid.")
            return job

    def mark_worker_context_issued(self, job_id: UUID) -> StoredJob:
        """Record that the worker fetched its credential context."""

        def mutate(job: StoredJob) -> StoredJob:
            now = _now()
            return job.model_copy(
                update={
                    "worker_context_issued_at": now,
                    "worker_revision": job.library_revision,
                    "updated_at": now,
                }
            )

        return self._mutate(job_id, mutate)

    def update_progress_by_token(
        self, token_hash: str, progress: AuthoringProgress
    ) -> StoredJob:
        """Persist progress for a running job identified by its token hash."""
        with self._lock:
            job = self._find_by_token_unlocked(token_hash)
            if job is None:
                raise AuthoringNotFound("Authoring worker token is invalid.")
            if job.state != "running":
                return job
            updated = job.model_copy(
                update={"progress": progress, "updated_at": _now()}
            )
            self._write_job_unlocked(updated)
            return updated

    def fail_by_token(
        self, token_hash: str, error: PublicAuthoringError
    ) -> StoredJob:
        """Fail an active job identified by its token hash."""
        with self._lock:
            job = self._find_by_token_unlocked(token_hash)
            if job is None:
                raise AuthoringNotFound("Authoring worker token is invalid.")
            if job.state not in {"queued", "running"}:
                raise AuthoringConflict("Authoring job is already finished.")
            now = _now()
            updated = job.model_copy(
                update={
                    "state": "failed",
                    "progress": None,
                    "error": error,
                    "updated_at": now,
                    "finished_at": now,
                }
            )
            self._write_job_unlocked(updated)
            return updated

    def update_progress(self, job_id: UUID, progress: AuthoringProgress) -> None:
        """Persist the latest phase/attempt while a job is running."""

        def mutate(job: StoredJob) -> StoredJob:
            if job.state != "running":
                return job
            return job.model_copy(
                update={
                    "progress": progress,
                    "updated_at": _now(),
                }
            )

        try:
            self._mutate(job_id, mutate)
        except AuthoringStoreError:
            logger.warning("Could not persist authoring progress", exc_info=True)

    def complete_job(self, job_id: UUID, report: AuthoringResult) -> StoredJob:
        """Persist the exact artifact, full report and safe job outcome."""
        with self._lock:
            job = self._load_job(job_id)
            if job.state not in {"queued", "running"}:
                raise AuthoringConflict("Authoring job is already finished.")

            summary = summarize_authoring(report)
            artifact = report.artifact
            artifact_sha256: str | None = None
            if report.status == "checked":
                if artifact is None or artifact.artifact_id is None:
                    raise AuthoringConflict("Checked authoring report did not contain an artifact.")
                artifact_bytes = artifact.model_dump_json().encode()
                artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
                self._write_artifact(job_id, artifact_bytes)

            try:
                report_bytes = report.model_dump_json().encode()
                self._write_bytes(self._report_path(job_id), report_bytes, overwrite=True)
            except Exception:
                if report.status == "checked":
                    self._artifact_path(job_id).unlink(missing_ok=True)
                raise

            now = _now()
            state: JobState = "failed" if report.status == "error" else "completed"
            error = summary.error
            if state == "failed" and error is None:
                error = PublicAuthoringError(
                    code="AUTHORING_FAILED",
                    message="Authoring stopped without a usable artifact.",
                )
            updated = job.model_copy(
                update={
                    "state": state,
                    "progress": None,
                    "result": summary,
                    "artifact_id": (artifact.artifact_id if report.status == "checked" else None),
                    "artifact_sha256": artifact_sha256,
                    "error": error if state == "failed" else None,
                    "approval": None,
                    "last_generation": None,
                    "updated_at": now,
                    "finished_at": now,
                }
            )
            self._write_job_unlocked(updated)
            return updated

    def fail_job(self, job_id: UUID, error: PublicAuthoringError) -> StoredJob:
        """Fail a queued/running job without replacing a completed outcome."""

        def mutate(job: StoredJob) -> StoredJob:
            if job.state not in {"queued", "running"}:
                return job
            now = _now()
            return job.model_copy(
                update={
                    "state": "failed",
                    "progress": None,
                    "error": error,
                    "updated_at": now,
                    "finished_at": now,
                }
            )

        return self._mutate(job_id, mutate)

    def save_approval(self, job_id: UUID, approval: ApprovalRecord) -> StoredJob:
        """Replace the current decision when the exact checked evidence still matches."""

        def mutate(job: StoredJob) -> StoredJob:
            if approval.artifact_sha256 != job.artifact_sha256:
                raise AuthoringConflict("The artifact changed; refresh the review before deciding.")
            self._require_reusable_unlocked(job, approval.artifact_sha256)
            return job.model_copy(update={"approval": approval, "updated_at": _now()})

        return self._mutate(job_id, mutate)

    def save_generation(self, job_id: UUID, metadata: GenerationMetadata) -> StoredJob:
        """Record batch metadata only while the exact approval remains current."""

        def mutate(job: StoredJob) -> StoredJob:
            if metadata.artifact_sha256 != job.artifact_sha256:
                raise AuthoringConflict(
                    "The artifact changed while questions were being generated."
                )
            self._require_reusable_unlocked(job, metadata.artifact_sha256)
            approval = job.approval
            if (
                approval is None
                or approval.decision != "approved"
                or approval.artifact_sha256 != metadata.artifact_sha256
            ):
                raise AuthoringConflict(
                    "The exact artifact must remain approved while generating questions."
                )
            return job.model_copy(update={"last_generation": metadata, "updated_at": _now()})

        return self._mutate(job_id, mutate)

    def read_artifact(self, job_id: UUID) -> bytes:
        """Read the exact stored artifact bytes."""
        with self._lock:
            path = self._artifact_path(job_id)
            try:
                return path.read_bytes()
            except OSError as exc:
                raise AuthoringConflict("The validated artifact is unavailable.") from exc

    def read_report(self, job_id: UUID) -> AuthoringResult:
        """Read the full server-only report for review projection."""
        with self._lock:
            path = self._report_path(job_id)
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise AuthoringConflict("Authoring review evidence is unavailable.") from exc
            try:
                return AuthoringResult.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                raise AuthoringConflict("Authoring review evidence is invalid.") from exc

    def _require_reusable_unlocked(
        self, job: StoredJob, expected_artifact_sha256: str
    ) -> AuthoringResult:
        """Validate persisted evidence while the store lock is already held."""
        artifact_bytes = self.read_artifact(job.job_id)
        artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_digest != expected_artifact_sha256:
            raise AuthoringConflict(
                "The stored artifact failed its integrity check; refresh the review."
            )
        report = self.read_report(job.job_id)
        return require_reusable_evidence(
            job,
            expected_artifact_sha256=expected_artifact_sha256,
            report=report,
        )

    def reconcile_startup(self) -> int:
        """Mark prior queued/running jobs failed; never replay model work."""
        interrupted = 0
        now = _now()
        error = PublicAuthoringError(code="SERVER_RESTARTED", message=SERVER_RESTARTED_MESSAGE)
        with self._lock:
            for path in self.jobs_dir.glob("*.json"):
                try:
                    job = StoredJob.model_validate_json(path.read_bytes())
                except (OSError, ValidationError, ValueError):
                    logger.warning("Ignoring unreadable authoring job record")
                    continue
                if job.state not in {"queued", "running"}:
                    continue
                updated = job.model_copy(
                    update={
                        "state": "failed",
                        "progress": None,
                        "error": error,
                        "updated_at": now,
                        "finished_at": now,
                    }
                )
                self._write_job_unlocked(updated)
                interrupted += 1
        if interrupted:
            logger.warning("Marked %s interrupted AI authoring jobs failed", interrupted)
        return interrupted

    def _mutate(self, job_id: UUID, mutator: Callable[[StoredJob], StoredJob]) -> StoredJob:
        with self._lock:
            job = self._load_job(job_id)
            updated = mutator(job)
            if updated == job:
                return job
            self._write_job_unlocked(updated)
            return updated

    def _find_by_token_unlocked(self, token_hash: str) -> StoredJob | None:
        """Scan job records for a matching hash while the lock is already held."""
        if not token_hash:
            return None
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = StoredJob.model_validate_json(path.read_bytes())
            except (OSError, ValidationError, ValueError):
                logger.warning("Ignoring unreadable authoring job record")
                continue
            if job.worker_token_hash and hmac.compare_digest(
                job.worker_token_hash, token_hash
            ):
                return job
        return None

    def _load_job(self, job_id: UUID) -> StoredJob:
        path = self._job_path(job_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise AuthoringNotFound("Authoring job not found.") from exc
        except OSError as exc:
            raise AuthoringStorageUnavailable("Authoring job record is unavailable.") from exc
        try:
            return StoredJob.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            raise AuthoringConflict("Authoring job record is invalid.") from exc

    def _write_job_unlocked(self, job: StoredJob) -> None:
        self._write_bytes(
            self._job_path(job.job_id), job.model_dump_json().encode(), overwrite=True
        )

    def _write_artifact(self, job_id: UUID, data: bytes) -> None:
        path = self._artifact_path(job_id)
        if path.exists():
            raise AuthoringConflict("The validated artifact already exists.")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_bytes(path, data, overwrite=False)

    def _job_path(self, job_id: UUID) -> Path:
        return self.jobs_dir / f"{_canonical_uuid(job_id)}.json"

    def _artifact_path(self, job_id: UUID) -> Path:
        return self.artifacts_dir / _canonical_uuid(job_id) / "artifact.json"

    def _report_path(self, job_id: UUID) -> Path:
        return self.artifacts_dir / _canonical_uuid(job_id) / "report.json"

    def _write_bytes(self, path: Path, data: bytes, *, overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not overwrite and path.exists():
            raise AuthoringConflict("Refusing to overwrite an existing record.")
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if not overwrite and path.exists():
                raise AuthoringConflict("Refusing to overwrite an existing record.")
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise


def _canonical_uuid(value: UUID) -> str:
    """Accept only UUID objects, never client path text."""
    if not isinstance(value, UUID):
        raise AuthoringNotFound("Authoring job not found.")
    return str(value)


def _now() -> datetime:
    return datetime.now(UTC)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)
