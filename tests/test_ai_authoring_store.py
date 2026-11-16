"""Durable file store, ownership and restart reconciliation tests."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from edcraft_validator.application.authoring_contracts import AuthoringError

from edcraft_backend.config.settings import AuthoringSettings
from edcraft_backend.schemas.ai_authoring import GenerationMetadata
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
from tests.ai_authoring_helpers import (
    make_checked_report,
    make_command,
    make_needs_review_report,
)


def _store(tmp_path) -> AuthoringStore:
    store = AuthoringStore(AuthoringSettings(storage_dir=tmp_path))
    store.ensure_ready()
    return store


def _job(owner_id, job_id, prompt="Create a tracing question") -> StoredJob:
    now = datetime.now(UTC)
    return StoredJob(
        job_id=job_id,
        owner_id=owner_id,
        request=AuthoringRequestRecord(
            prompt=prompt,
            difficulty="easy",
            num_distractors=3,
            domain="code",
            model_profile="test-profile",
        ),
        provider="stub",
        model="stub-model",
        library_revision="test-revision",
        created_at=now,
        updated_at=now,
    )


def test_store_is_owner_scoped_and_uses_clean_atomic_paths(tmp_path) -> None:
    store = _store(tmp_path)
    owner = uuid4()
    other = uuid4()
    job = _job(owner, uuid4())
    store.create_job(job)
    store.mark_started(job.job_id)

    assert store.get_owned_job(job.job_id, owner).job_id == job.job_id
    with pytest.raises(AuthoringNotFound):
        store.get_owned_job(job.job_id, other)
    with pytest.raises(AuthoringNotFound):
        store.get_job(uuid4())

    assert [item.job_id for item in store.list_jobs(owner)] == [job.job_id]
    assert store.list_jobs(other) == []
    assert not list(store.jobs_dir.glob(".tmp-*"))
    assert (store.jobs_dir / f"{job.job_id}.json").is_file()


def test_needs_review_failure_keeps_job_completed_with_structured_code(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    command = make_command()
    job = _job(uuid4(), command.job_id)
    store.create_job(job)
    store.mark_started(job.job_id)
    report = make_needs_review_report(
        command,
        failure=AuthoringError(stage="checking", code="TEMPLATE_CHECKS_FAILED"),
    )

    completed = store.complete_job(job.job_id, report)

    assert completed.state == "completed"
    assert completed.result is not None
    assert completed.result.error is not None
    assert completed.result.error.code == "TEMPLATE_CHECKS_FAILED"
    assert completed.error is None


def test_checked_report_stores_exact_artifact_and_report(tmp_path) -> None:
    store = _store(tmp_path)
    command = make_command()
    job = _job(uuid4(), command.job_id)
    store.create_job(job)
    store.mark_started(job.job_id)
    report = make_checked_report(command)

    completed = store.complete_job(job.job_id, report)

    assert completed.state == "completed"
    assert completed.result is not None
    assert completed.result.status == "checked"
    assert completed.artifact_sha256 is not None
    assert completed.artifact_id == report.artifact.artifact_id
    artifact_bytes = store.read_artifact(job.job_id)
    assert artifact_bytes == report.artifact.model_dump_json().encode()
    assert store.read_report(job.job_id).provider_settings == {"api_key": "never-return-this"}


def test_reconcile_marks_running_jobs_failed_and_preserves_approval(tmp_path) -> None:
    store = _store(tmp_path)
    owner = uuid4()

    completed_command = make_command()
    completed_job = _job(owner, completed_command.job_id)
    store.create_job(completed_job)
    store.mark_started(completed_job.job_id)
    completed = store.complete_job(completed_job.job_id, make_checked_report(completed_command))
    approval = ApprovalRecord(
        decision="approved",
        artifact_sha256=completed.artifact_sha256 or "",
        reviewer_id=owner,
        reviewer=str(owner),
        decided_at=datetime.now(UTC),
    )
    store.save_approval(completed_job.job_id, approval)

    running_job = _job(owner, uuid4(), prompt="Interrupted work")
    store.create_job(running_job)
    store.mark_started(running_job.job_id)

    assert store.reconcile_startup() == 1

    reloaded = store.get_owned_job(completed_job.job_id, owner)
    assert reloaded.state == "completed"
    assert reloaded.approval is not None
    assert reloaded.approval.decision == "approved"
    assert reloaded.artifact_sha256 == completed.artifact_sha256

    interrupted = store.get_owned_job(running_job.job_id, owner)
    assert interrupted.state == "failed"
    assert interrupted.progress is None
    assert interrupted.error is not None
    assert interrupted.error.code == "SERVER_RESTARTED"
    assert "restart" in interrupted.error.message.lower()


def test_refuses_duplicate_job_and_non_writable_volume(tmp_path) -> None:
    store = _store(tmp_path)
    job = _job(uuid4(), uuid4())
    store.create_job(job)
    with pytest.raises(AuthoringConflict):
        store.create_job(job)

    blocking_file = tmp_path / "plain-file"
    blocking_file.write_text("not a directory")
    bad_store = AuthoringStore(AuthoringSettings(storage_dir=blocking_file / "child"))
    with pytest.raises(AuthoringStorageUnavailable):
        bad_store.ensure_ready()


def _approval(
    digest: str,
    owner: UUID,
    decision: Literal["approved", "rejected"] = "approved",
) -> ApprovalRecord:
    return ApprovalRecord(
        decision=decision,
        artifact_sha256=digest,
        reviewer_id=owner,
        reviewer=str(owner),
        decided_at=datetime.now(UTC),
    )


def _generation(digest: str) -> GenerationMetadata:
    return GenerationMetadata(
        artifact_sha256=digest,
        base_seed=0,
        requested_quantity=1,
        actual_quantity=1,
        distinct_count=1,
        duplicates=0,
        warnings=[],
        created_at=datetime.now(UTC),
    )


def test_save_approval_and_generation_require_saved_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    owner = uuid4()
    command = make_command()
    job = _job(owner, command.job_id)
    store.create_job(job)
    store.mark_started(job.job_id)
    completed = store.complete_job(job.job_id, make_checked_report(command))
    digest = completed.artifact_sha256 or ""

    report_path = store.root / "artifacts" / str(job.job_id) / "report.json"
    report_path.unlink()

    with pytest.raises(AuthoringConflict):
        store.save_approval(job.job_id, _approval(digest, owner))
    assert store.get_job(job.job_id).approval is None

    with pytest.raises(AuthoringConflict):
        store.save_generation(job.job_id, _generation(digest))
    assert store.get_job(job.job_id).last_generation is None


def test_save_approval_and_generation_reject_changed_artifact(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    owner = uuid4()
    command = make_command()
    job = _job(owner, command.job_id)
    store.create_job(job)
    store.mark_started(job.job_id)
    completed = store.complete_job(job.job_id, make_checked_report(command))
    digest = completed.artifact_sha256 or ""
    store.save_approval(job.job_id, _approval(digest, owner))

    artifact_path = store.root / "artifacts" / str(job.job_id) / "artifact.json"
    artifact_path.write_bytes(b'{"tampered":true}')

    with pytest.raises(AuthoringConflict, match="integrity"):
        store.save_approval(job.job_id, _approval(digest, owner, "rejected"))
    with pytest.raises(AuthoringConflict, match="integrity"):
        store.save_generation(job.job_id, _generation(digest))

    stored = store.get_job(job.job_id)
    assert stored.approval is not None
    assert stored.approval.decision == "approved"
    assert stored.last_generation is None


def _completed_store(tmp_path: Path) -> tuple[AuthoringStore, UUID, StoredJob]:
    store = _store(tmp_path)
    owner = uuid4()
    command = make_command()
    job = _job(owner, command.job_id)
    store.create_job(job)
    store.mark_started(job.job_id)
    completed = store.complete_job(job.job_id, make_checked_report(command))
    return store, owner, completed


def _artifact_path(store: AuthoringStore, job: StoredJob) -> Path:
    return store.root / "artifacts" / str(job.job_id) / "artifact.json"


def _report_path(store: AuthoringStore, job: StoredJob) -> Path:
    return store.root / "artifacts" / str(job.job_id) / "report.json"


def test_read_evidence_returns_verified_artifact_and_report(tmp_path: Path) -> None:
    store, owner, job = _completed_store(tmp_path)

    evidence = store.read_evidence(job.job_id, owner_id=owner)

    assert evidence.job.job_id == job.job_id
    assert evidence.artifact.artifact_id == job.artifact_id
    assert evidence.report.status == "checked"
    assert evidence.artifact_bytes == store.read_artifact(job.job_id)


def test_read_evidence_rejects_another_owner(tmp_path: Path) -> None:
    store, _, job = _completed_store(tmp_path)

    with pytest.raises(AuthoringNotFound):
        store.read_evidence(job.job_id, owner_id=uuid4())


def test_read_evidence_rejects_missing_artifact_file(tmp_path: Path) -> None:
    store, owner, job = _completed_store(tmp_path)
    _artifact_path(store, job).unlink()

    with pytest.raises(AuthoringConflict, match="artifact is unavailable"):
        store.read_evidence(job.job_id, owner_id=owner)


def test_read_evidence_rejects_stale_report(tmp_path: Path) -> None:
    store, owner, job = _completed_store(tmp_path)
    report = store.read_report(job.job_id)
    _report_path(store, job).write_bytes(
        report.model_copy(update={"artifact": None}).model_dump_json().encode()
    )

    with pytest.raises(AuthoringConflict, match="does not match the job artifact"):
        store.read_evidence(job.job_id, owner_id=owner)


def test_read_evidence_rejects_tampered_bytes(tmp_path: Path) -> None:
    store, owner, job = _completed_store(tmp_path)
    _artifact_path(store, job).write_bytes(b'{"tampered":true}')

    with pytest.raises(AuthoringConflict, match="integrity"):
        store.read_evidence(job.job_id, owner_id=owner)


def test_read_evidence_requires_a_current_approval_for_generation(
    tmp_path: Path,
) -> None:
    store, owner, job = _completed_store(tmp_path)
    digest = job.artifact_sha256 or ""
    store.save_approval(job.job_id, _approval(digest, owner, "rejected"))

    with pytest.raises(AuthoringConflict, match="must remain approved"):
        store.read_evidence(
            job.job_id,
            owner_id=owner,
            requirement=EvidenceRequirement.APPROVED,
        )

    # A rejected decision is still durable review evidence for the review view.
    assert store.read_evidence(job.job_id, owner_id=owner).approval.decision == "rejected"


def test_read_evidence_accepts_an_equivalent_reserialized_artifact(
    tmp_path: Path,
) -> None:
    store, owner, job = _completed_store(tmp_path)
    artifact_path = _artifact_path(store, job)
    reserialized = json.dumps(json.loads(artifact_path.read_bytes()), indent=2).encode()
    assert reserialized != artifact_path.read_bytes()
    artifact_path.write_bytes(reserialized)
    digest = hashlib.sha256(reserialized).hexdigest()
    store._mutate(
        job.job_id,
        lambda current: current.model_copy(update={"artifact_sha256": digest}),
    )

    evidence = store.read_evidence(job.job_id, owner_id=owner)

    # The stored raw bytes, not a report re-serialisation, define the artifact.
    assert evidence.artifact_bytes == reserialized
    assert evidence.artifact.artifact_id == job.artifact_id
