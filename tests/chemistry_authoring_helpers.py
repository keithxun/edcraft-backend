"""Chemistry deterministic authoring test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    run_authoring_job_sync,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.domains.chemistry.chemistry_schemas import (
    BalancedEquationTarget,
    ChemistryItemSpec,
    ChemistryTemplateContent,
    NormalisedCoefficientsRecipe,
    PerturbCoefficient,
    ReactionParameter,
    ScaleBalanced,
    SkeletonCoefficients,
    SwapSpecies,
)
from edcraft_validator.domains.chemistry.chemistry_types import (
    CHEMISTRY_BALANCE_TOOL,
    CHEMISTRY_OPTIONS_TOOL,
)
from edcraft_validator.llm.llm_contracts import PlannedGenerationResponse
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server
from edcraft_validator.validation.check_names import (
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

from tests.ai_authoring_helpers import ScriptedDecisions, ScriptedJudge

CHEMISTRY_CHECKS = (
    CHEMISTRY_BALANCE_TOOL,
    CHEMISTRY_OPTIONS_TOOL,
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

CHEMISTRY_REACTION_IDS = ["comb.methane", "comb.ethane", "comb.propane"]


def _chemistry_item_spec() -> ChemistryItemSpec:
    """A small finite alkane-combustion family over three catalogue reactions."""
    return ChemistryItemSpec(
        schema_version="chem-balance/0.1",
        family="alkane_combustion",
        parameters=[ReactionParameter(name="reaction", values=CHEMISTRY_REACTION_IDS)],
        target=BalancedEquationTarget(kind="balanced_equation"),
        answer_recipe=NormalisedCoefficientsRecipe(kind="normalised_coefficients"),
        distractors=[
            PerturbCoefficient(
                kind="perturb_coefficient",
                slot="product:2",
                delta=1,
                misconception="unbalanced_element",
            ),
            ScaleBalanced(
                kind="scale_balanced",
                factor=2,
                misconception="non_reduced_coefficients",
            ),
            SwapSpecies(
                kind="swap_species",
                variant="O2_as_O",
                misconception="wrong_species",
            ),
            ScaleBalanced(
                kind="scale_balanced",
                factor=3,
                misconception="non_reduced_coefficients",
            ),
            SkeletonCoefficients(
                kind="skeleton_coefficients",
                misconception="unbalanced_skeleton",
            ),
            PerturbCoefficient(
                kind="perturb_coefficient",
                slot="reactant:2",
                delta=1,
                misconception="unbalanced_element",
            ),
        ],
    )


def chemistry_proposal() -> ChemistryTemplateContent:
    return ChemistryTemplateContent(item=_chemistry_item_spec())


def chemistry_concept_proposal() -> ChemistryTemplateContent:
    return ChemistryTemplateContent.model_validate(
        {
            "item": {
                "schema_version": "chem-concept/0.1",
                "family": "atomic_structure",
                "parameters": [{"name": "context", "values": ["skill-1-d-1-a", "skill-1-a-1-a"]}],
                "target": {"kind": "concept"},
                "answer_recipe": {"kind": "concept_key"},
                "distractors": [{"kind": "concept_alternative", "index": i} for i in range(1, 7)],
            }
        }
    )


def chemistry_yield_proposal() -> ChemistryTemplateContent:
    return ChemistryTemplateContent.model_validate(
        {
            "item": {
                "schema_version": "chem-yield/0.1",
                "family": "neutralisation",
                "parameters": [
                    {"name": "reaction", "values": ["neutral.h3po4_koh"]},
                    {"name": "reactant_1_mmol", "values": [5]},
                    {"name": "reactant_2_mmol", "values": [2]},
                ],
                "target": {"kind": "theoretical_yield", "product": "product:1"},
                "answer_recipe": {"kind": "limiting_extent"},
                "distractors": [
                    {"kind": "yield_alternative", "method": method}
                    for method in (
                        "sum_extents",
                        "multiply_by_1000",
                        "divide_by_1000",
                        "first_extent",
                        "second_extent",
                        "larger_extent",
                    )
                ],
            }
        }
    )


class ScriptedChemistryProvider:
    """Return one known chemistry proposal and the fixed chemistry check plan."""

    provider = "stub"
    model = "stub-model"

    def __init__(self, proposal_factory: Callable[[], ChemistryTemplateContent]) -> None:
        self._proposal_factory = proposal_factory

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        assert request.response_model
        content = self._proposal_factory()
        checks = (
            CHEMISTRY_CHECKS
            if content.item.schema_version == "chem-balance/0.1"
            else (TEMPLATE_DIFFICULTY_CHECK, TEMPLATE_RELEVANCE_CHECK)
        )
        return PlannedGenerationResponse(
            proposal=content,
            checks=[{"name": name, "arguments": {}} for name in checks],
        )


def run_scripted_chemistry_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
    *,
    proposal_factory: Callable[[], ChemistryTemplateContent] = chemistry_proposal,
) -> AuthoringJobRecord:
    """Run the real chemistry workflow with a scripted provider and real MCP checks."""
    tool_client = FastMcpToolClient(
        create_validation_server(
            difficulty_provider_factory=lambda: ScriptedJudge(),
            decision_provider_factory=lambda: ScriptedDecisions(),
        )
    )

    def application_factory() -> TemplateApplication:
        return TemplateApplication(tool_client=tool_client)

    return run_authoring_job_sync(
        command,
        application_factory=application_factory,
        provider_factory=lambda selection: ScriptedChemistryProvider(proposal_factory),
        on_progress=on_progress,
    )


def make_chemistry_command(job_id: UUID | None = None) -> AuthoringJobCommand:
    return AuthoringJobCommand(
        job_id=job_id or uuid4(),
        domain="chemistry",
        author={"provider": "stub", "model": "stub-model"},
        request={
            "prompt": "Create a balanced alkane combustion equation question",
            "difficulty": "medium",
            "num_distractors": 3,
        },
    )
