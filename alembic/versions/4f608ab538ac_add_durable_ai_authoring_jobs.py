"""Add durable AI authoring data to the shared jobs table.

Revision ID: 4f608ab538ac
Revises: d85dbdecf221
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "4f608ab538ac"
down_revision: str | Sequence[str] | None = "d85dbdecf221"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create private authoring data and queue/list indexes."""
    op.create_table(
        "authoring_job_data",
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
        sa.Column("artifact_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("report_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("report_sha256", sa.String(length=64), nullable=True),
        sa.Column("execution_mode", sa.String(length=10), server_default="inline", nullable=False),
        sa.Column("execution_payload", sa.Text(), nullable=True),
        sa.Column("attempt_id", sa.Uuid(), nullable=True),
        sa.Column("attempt_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_claim_id", sa.String(length=255), nullable=True),
        sa.Column("worker_context_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("nomad_registered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(artifact_bytes IS NULL) = (artifact_sha256 IS NULL)",
            name="ck_authoring_job_artifact_digest",
        ),
        sa.CheckConstraint(
            "(report_bytes IS NULL) = (report_sha256 IS NULL)",
            name="ck_authoring_job_report_digest",
        ),
        sa.CheckConstraint(
            "execution_mode IN ('inline', 'nomad')", name="ck_authoring_job_execution_mode"
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(
        "ix_jobs_authoring_owner_created",
        "jobs",
        ["user_id", "created_at", "id"],
        postgresql_where=sa.text("type = 'ai_authoring'"),
    )
    op.create_index(
        "ix_jobs_authoring_status_created",
        "jobs",
        ["status", "created_at", "id"],
        postgresql_where=sa.text("type = 'ai_authoring'"),
    )
    op.create_index(
        "ix_authoring_active_attempts",
        "authoring_job_data",
        ["execution_mode", "job_id"],
        postgresql_where=sa.text("attempt_id IS NOT NULL AND attempt_ended_at IS NULL"),
    )
    op.create_index(
        "ix_authoring_queued_attempts",
        "authoring_job_data",
        ["execution_mode", "job_id"],
        postgresql_where=sa.text("attempt_id IS NULL AND execution_payload IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove authoring rows and schema while preserving other job types."""
    op.execute(sa.text("DELETE FROM jobs WHERE type = 'ai_authoring'"))
    op.drop_index("ix_jobs_authoring_status_created", table_name="jobs")
    op.drop_index("ix_jobs_authoring_owner_created", table_name="jobs")
    op.drop_table("authoring_job_data")
