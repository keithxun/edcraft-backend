"""Job and JobToken models for async Nomad job queue."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from edcraft_backend.models.base import Base


class JobStatus(StrEnum):
    """Lifecycle states of an async job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(StrEnum):
    """Which endpoint submitted this job."""

    ANALYSE_CODE = "analyse_code"
    GENERATE_QUESTION = "generate_question"
    GENERATE_TEMPLATE = "generate_template"
    QUESTION_FROM_TEMPLATE = "question_from_template"
    ASSESSMENT_FROM_TEMPLATE = "assessment_from_template"
    GENERATE_INPUTS = "generate_inputs"
    AI_AUTHORING = "ai_authoring"


class Job(Base):
    """Tracks the status and result of a Nomad batch job."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "ix_jobs_authoring_owner_created",
            "user_id",
            "created_at",
            "id",
            postgresql_where=text("type = 'ai_authoring'"),
        ),
        Index(
            "ix_jobs_authoring_status_created",
            "status",
            "created_at",
            "id",
            postgresql_where=text("type = 'ai_authoring'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=JobStatus.QUEUED.value)
    nomad_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobToken(Base):
    """One-time token used by a Nomad worker to POST results back via callback."""

    __tablename__ = "job_tokens"

    token: Mapped[str] = mapped_column(String(255), primary_key=True)
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
