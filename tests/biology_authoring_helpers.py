"""Biology deterministic authoring test fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    run_authoring_job_sync,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringProgress,
    AuthoringResult,
    CheckExecution,
    GenerationAttempt,
)
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.domains.biology.biology_domain import BiologyDomain
from edcraft_validator.domains.biology.biology_schemas import (
    BiologyTemplateContent,
    BiologyTemplateRequest,
)
from edcraft_validator.domains.biology.candidate_builder import build_biology_candidate
from edcraft_validator.llm.llm_contracts import PlannedGenerationResponse
from edcraft_validator.mcp.biology_tools import (
    BiologyFidelityRequest,
    validate_options,
    verify_cross,
    verify_text_fidelity_tool,
)
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server
from edcraft_validator.validation.check_names import (
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

from tests.ai_authoring_helpers import ScriptedDecisions, ScriptedJudge, StaticExecutor

BIOLOGY_ITEM: dict[str, Any] = {
    "schema_version": "bio-mono/0.1",
    "topic": "genetics.monohybrid.conditional_count",
    "organism": "glimmerwort",
    "trait": "petal colour",
    "gene": {"symbol": "P", "dominant_phenotype": "purple", "recessive_phenotype": "white"},
    "assumptions": {
        "inheritance": "autosomal",
        "dominance": "complete",
        "segregation": "mendelian",
        "allele_count": 2,
    },
    "parameters": [
        {"name": "parent1", "kind": "genotype", "values": ["Pp"]},
        {"name": "parent2", "kind": "genotype", "values": ["Pp"]},
        {"name": "target", "kind": "genotype", "values": ["Pp"]},
        {"name": "n", "kind": "count", "values": ["30", "60", "90", "120"]},
    ],
    "cross": {"parent1": "parent1", "parent2": "parent2"},
    "question": {
        "quantity": "expected_count",
        "target": {"kind": "genotype", "param": "target"},
        "given": {"phenotype": "dominant"},
        "offspring_count": "n",
    },
    "answer_recipe": {
        "rule": "exact",
        "claimed_examples": [
            {
                "parameters": {"parent1": "Pp", "parent2": "Pp", "target": "Pp", "n": "90"},
                "answer": "60",
            }
        ],
    },
    "distractors": [
        {"rule": "fixed_value", "value": "60", "misconception": "assumes a fixed count"},
        {"rule": "unconditioned", "misconception": "ignores the condition"},
        {"rule": "complement", "misconception": "counts the homozygotes instead"},
        {"rule": "reverse_condition", "misconception": "reverses the condition"},
        {"rule": "forget_heterozygote", "misconception": "treats heterozygotes as white"},
        {"rule": "uniform_genotype_classes", "misconception": "uses equally likely classes"},
    ],
}

BIOLOGY_CHECKS = (
    "biology_verify_cross",
    "biology_validate_options",
    "biology_verify_text_fidelity",
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

FIT_ITEM: dict[str, Any] = {
    "schema_version": "bio-fit/0.1",
    "topic": "statistics.plant_goodness_of_fit",
    "categories": [
        {"label": "smooth leaves", "observed": [21, 31], "expected_weight": 1},
        {"label": "ruffled leaves", "observed": [13], "expected_weight": 1},
    ],
    "critical_value": "3.84",
    "claimed_examples": [{"observed": [21, 13], "statistic": "1.88", "decision": "fail_to_reject"}],
    "distractors": [
        {"rule": "observed_denominator", "misconception": "uses the observed divisor"},
        {"rule": "unweighted_squared_error", "misconception": "omits the divisors"},
        {"rule": "omit_category", "category": 1, "misconception": "omits the first row"},
        {"rule": "divide_by_total", "misconception": "normalizes again"},
        {"rule": "largest_contribution", "misconception": "uses only the largest row"},
        {"rule": "reverse_decision", "misconception": "reverses the inequality"},
    ],
}


class ScriptedBiologyProvider:
    """Return one known biology proposal and the fixed biology check plan."""

    provider = "stub"
    model = "stub-model"

    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        return request.response_model.model_validate(
            {
                "proposal": {"item": self.item},
                "checks": [{"name": name, "arguments": {}} for name in BIOLOGY_CHECKS],
            }
        )


def _unexpected_fidelity_model() -> Any:
    raise AssertionError("A rendered or historical Biology item needs no blind parser")


def run_scripted_biology_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
    *,
    item: dict[str, Any] = BIOLOGY_ITEM,
) -> AuthoringJobRecord:
    """Run real Biology authoring with a scripted model and real MCP checks."""
    server = create_validation_server(
        difficulty_provider_factory=ScriptedJudge,
        decision_provider_factory=ScriptedDecisions,
        fidelity_provider_factory=_unexpected_fidelity_model,
    )
    tool_client = FastMcpToolClient(server)

    def application_factory() -> TemplateApplication:
        return TemplateApplication(tool_client=tool_client)

    return run_authoring_job_sync(
        command,
        application_factory=application_factory,
        provider_factory=lambda selection: ScriptedBiologyProvider(item),
        on_progress=on_progress,
    )


def historical_biology_executor(item: dict[str, Any]) -> StaticExecutor:
    """Supply an already-checked historical record to the HTTP persistence layer."""

    def report(command: AuthoringJobCommand) -> AuthoringResult:
        domain = BiologyDomain()
        request = BiologyTemplateRequest.model_validate(command.request)
        proposal = BiologyTemplateContent.model_validate({"item": item})
        candidate = build_biology_candidate(request, proposal)
        evidence = [
            verify_cross(candidate.item),
            validate_options(candidate.item),
            asyncio.run(
                verify_text_fidelity_tool(
                    BiologyFidelityRequest(template=candidate.item),
                    provider_factory=_unexpected_fidelity_model,
                    timeout_seconds=1,
                )
            ),
        ]
        artifact = domain.finalize_checked_template(
            candidate, evidence, advisory_checks=frozenset()
        ).model_copy(update={"artifact_id": command.job_id.hex})
        executions = [
            CheckExecution(
                call_id=f"attempt-1:{check.tool}",
                tool=check.tool,
                requested_arguments={},
                evidence=check,
            )
            for check in evidence
        ]
        return AuthoringResult(
            domain=domain.name,
            status="checked",
            provider="stub",
            model="historical-fixture",
            request=command.request,
            prompt_version="historical-h2-biology",
            fixed_plan=[check.tool for check in evidence],
            tool_catalogue=[],
            artifact=artifact,
            attempts=[
                GenerationAttempt(
                    number=1,
                    candidate=candidate.model_dump(mode="json"),
                    executions=executions,
                    pending=[],
                    complete=True,
                    passed=True,
                )
            ],
        )

    return StaticExecutor(report)
