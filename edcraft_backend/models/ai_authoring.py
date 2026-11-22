"""Authoring metadata and exact evidence associated with an async job."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from edcraft_backend.models.base import Base
from edcraft_backend.models.job import Job


class AuthoringJobData(Base):
    """Private authoring data over the shared job lifecycle and owner."""

    __tablename__ = "authoring_job_data"
    __table_args__ = (
        CheckConstraint(
            "(artifact_bytes IS NULL) = (artifact_sha256 IS NULL)",
            name="ck_authoring_job_artifact_digest",
        ),
        CheckConstraint(
            "(report_bytes IS NULL) = (report_sha256 IS NULL)",
            name="ck_authoring_job_report_digest",
        ),
        CheckConstraint(
            "execution_mode IN ('inline', 'nomad')", name="ck_authoring_job_execution_mode"
        ),
        Index(
            "ix_authoring_active_attempts",
            "execution_mode",
            "job_id",
            postgresql_where=text("attempt_id IS NOT NULL AND attempt_ended_at IS NULL"),
        ),
        Index(
            "ix_authoring_queued_attempts",
            "execution_mode",
            "job_id",
            postgresql_where=text("attempt_id IS NULL AND execution_payload IS NOT NULL"),
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    job: Mapped[Job] = relationship()
    # JSON preserves escaped NUL characters in pasted prompts and material.
    record: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    artifact_bytes: Mapped[bytes | None] = mapped_column(LargeBinary)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    report_bytes: Mapped[bytes | None] = mapped_column(LargeBinary)
    report_sha256: Mapped[str | None] = mapped_column(String(64))
    execution_mode: Mapped[str] = mapped_column(String(10), server_default=text("'inline'"))
    execution_payload: Mapped[str | None] = mapped_column(Text)
    attempt_id: Mapped[UUID | None] = mapped_column()
    attempt_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_claim_id: Mapped[str | None] = mapped_column(String(255))
    worker_context_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    nomad_registered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
