"""Biology deterministic authoring test fixtures."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    run_authoring_job_sync,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.llm.llm_contracts import PlannedGenerationResponse
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server
from edcraft_validator.validation.check_names import (
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

from tests.ai_authoring_helpers import ScriptedDecisions, ScriptedJudge

BIOLOGY_ITEM: dict[str, Any] = {
    "schema_version": "bio-mono/0.1",
    "topic": "genetics.monohybrid.conditional_count",
    "organism": "a hypothetical flowering plant called the glimmerwort",
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
    "text": {
        "template": (
            "In {organism}, {dominant_phenotype} {trait} is completely dominant over "
            "{recessive_phenotype}. A {parent1} plant is crossed with a {parent2} "
            "plant. Of {n} offspring with {given_phenotype} petals, how many are "
            "expected to have genotype {target}?"
        )
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


class ScriptedBlindParser:
    """Return the correct blind reading of each rendered biology variant."""

    provider = "stub"
    model = "stub-blind-parser"

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> Any:
        # Parse-back reads several variants, which differ only in the count.
        match = re.search(r"Of (\d+) offspring", request.messages[-1]["content"])
        assert match is not None
        return request.response_model.model_validate(
            {
                "parent1_genotype": "Pp",
                "parent2_genotype": "Pp",
                "quantity": "expected_count",
                "target_kind": "genotype",
                "target_value": "Pp",
                "given_phenotype": "purple",
                "offspring_count": int(match.group(1)),
                "ratio_kind": None,
                "dominant_phenotype": "purple",
                "recessive_phenotype": "white",
                "dominance": "complete",
                "inheritance": "autosomal",
                "gene_count": 1,
                "organism_kind": "hypothetical",
                "segregation": "mendelian",
                "additional_rules": [],
                "sampled_population": "offspring",
            }
        )


def run_scripted_biology_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
    *,
    item: dict[str, Any] = BIOLOGY_ITEM,
) -> AuthoringJobRecord:
    """Run the real biology workflow with scripted models and real MCP checks.

    The server binds its blind parse-back judge when it is created, so the judge
    is replaced only for that call.
    """
    with patch(
        "edcraft_validator.mcp.server.configured_judge_provider",
        lambda settings: ScriptedBlindParser(),
    ):
        server = create_validation_server(
            difficulty_provider_factory=lambda: ScriptedJudge(),
            decision_provider_factory=lambda: ScriptedDecisions(),
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
