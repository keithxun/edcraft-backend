"""Smoke the real pinned AI workflow with a scripted provider and real MCP checks."""

import tomllib
from pathlib import Path

from edcraft_validator.domains.code.code_schemas import ValidatedCodeTemplate

from edcraft_backend.services.ai_authoring_service import LIBRARY_REVISION
from tests.ai_authoring_helpers import (
    generate,
    make_command,
    run_scripted_pipeline,
)


def test_library_revision_matches_the_pyproject_pin() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    pinned = config["tool"]["uv"]["sources"]["edcraft-validator"]["rev"]
    assert LIBRARY_REVISION == pinned


def test_scripted_library_pipeline_checks_and_reloads_a_real_artifact() -> None:
    command = make_command()
    progress = []
    record = run_scripted_pipeline(command, on_progress=progress.append)

    assert record.response.job_id == command.job_id
    assert record.report.status == "checked"
    assert record.report.artifact is not None
    assert record.report.artifact.validation.examples_checked > 0
    assert record.report.artifact.validation.execution_calls > 0
    assert record.report.attempts[-1].passed is True
    # Progress is reported per check, so collapse repeats to compare the phases.
    phases = [item.phase for item in progress]
    assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == [
        "generating",
        "checking",
        "finalizing",
    ]
    checking = [item for item in progress if item.phase == "checking"]
    assert checking[-1].checks_done == checking[-1].checks_total

    reloaded = ValidatedCodeTemplate.model_validate_json(record.report.artifact.model_dump_json())
    first = generate(reloaded, 11)
    second = generate(reloaded, 11)
    assert first == second
    assert first.question.proposed_answer is not None
    assert first.question.distractors
    assert first.question.distractor_reasons
