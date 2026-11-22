"""Synchronous executor boundary for the pinned AI authoring library."""

from __future__ import annotations

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    run_authoring_job_sync,
)
from edcraft_validator.application.template_workflow import TemplateApplication

from edcraft_backend.services.ai_authoring_judge import build_model_tool_client
from edcraft_backend.services.ai_authoring_runtime import (
    JobExecutor,
    JobRuntime,
    ProgressCallback,
)


def run_library_authoring_job(
    command: AuthoringJobCommand,
    runtime: JobRuntime,
    on_progress: ProgressCallback | None,
) -> AuthoringJobRecord:
    """Run the validator with separately selected author and advisory models."""
    tool_client = build_model_tool_client(
        difficulty=runtime.difficulty,
        relevance=runtime.relevance,
        decision_fallback=runtime.decision_fallback,
    )

    def application_factory() -> TemplateApplication:
        return TemplateApplication(tool_client=tool_client)

    if not runtime.author.is_byok:
        return run_authoring_job_sync(
            command,
            application_factory=application_factory,
            on_progress=on_progress,
        )
    author = runtime.author
    return run_authoring_job_sync(
        command,
        application_factory=application_factory,
        provider_factory=lambda _selection: author.open_provider(),
        on_progress=on_progress,
    )


__all__ = [
    "JobExecutor",
    "ProgressCallback",
    "run_library_authoring_job",
]
