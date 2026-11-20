"""Model-free physics authoring helpers for integration tests.

The scripted provider returns one known physics proposal, and the pipeline runs
the real physics MCP checks, so the test exercises the validator end to end
without a model call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    run_authoring_job_sync,
)
from edcraft_validator.application.authoring_contracts import AuthoringProgress
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.domains.physics.fidelity import (
    PhysicsFidelitySubject,
    identity_reconstruction,
    representative_variants,
)
from edcraft_validator.domains.physics.h2_authoring import H2CompactProposal, H2CompactUnsupported
from edcraft_validator.domains.physics.h2_statement_data import (
    H2StatementProposal,
    requested_skill,
)
from edcraft_validator.domains.physics.physics_schemas import (
    PhysicsTemplateContent,
    PhysicsTemplateSpec,
)
from edcraft_validator.domains.physics.physics_types import (
    PHYSICS_FIDELITY_TOOL,
    PHYSICS_ITEM_TOOL,
    PHYSICS_OPTIONS_TOOL,
    PHYSICS_TEXT_TOOL,
)
from edcraft_validator.domains.physics.text import render_question
from edcraft_validator.llm.llm_contracts import PlannedGenerationResponse
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server
from edcraft_validator.validation.check_names import (
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)

from tests.ai_authoring_helpers import ScriptedDecisions, ScriptedJudge

# Two opposing forces on a cart. Every answer and option is exact at two
# decimals for all 12 combinations, and the sign-error option never equals the key.
PHYSICS_ITEM: dict[str, Any] = {
    "schema_version": "physics-1d/0.1",
    "topic": "school.newton_second_law",
    "axis": {"orientation": "horizontal", "positive": "right"},
    "parameters": [
        {"name": "m", "quantity": "mass", "unit": "kg", "values": ["2", "4", "5"]},
        {"name": "F1", "quantity": "force", "unit": "N", "values": ["40", "60"]},
        {"name": "F2", "quantity": "force", "unit": "N", "values": ["10", "20"]},
    ],
    "body": {
        "target": "acceleration",
        "mass": "m",
        "forces": [
            {"param": "F1", "direction": "positive", "label": "push"},
            {"param": "F2", "direction": "negative", "label": "friction"},
        ],
        "text": "A cart of mass {m} is pushed with {F1}. Friction of {F2} also acts on it.",
        "answer_form": "magnitude_and_direction",
        "display": {"decimals": 2},
        "answer_recipe": {
            "op": "div",
            "args": [
                {"op": "sub", "args": [{"param": "F1"}, {"param": "F2"}]},
                {"param": "m"},
            ],
        },
        "distractors": [
            {
                "recipe": {
                    "op": "div",
                    "args": [
                        {"op": "sub", "args": [{"param": "F1"}, {"param": "F2"}]},
                        {"param": "m"},
                    ],
                },
                "misconception": "other",
                "explanation": "Repeats the correct answer.",
            },
            {
                "recipe": {
                    "op": "div",
                    "args": [
                        {"op": "add", "args": [{"param": "F1"}, {"param": "F2"}]},
                        {"param": "m"},
                    ],
                },
                "misconception": "added_opposing_forces",
                "explanation": "Adds the friction force instead of subtracting it.",
            },
            {
                "recipe": {"op": "div", "args": [{"param": "F1"}, {"param": "m"}]},
                "misconception": "ignored_opposing_force",
                "explanation": "Uses the push alone as the net force.",
            },
            {
                "recipe": {
                    "op": "neg",
                    "args": [
                        {
                            "op": "div",
                            "args": [
                                {"op": "sub", "args": [{"param": "F1"}, {"param": "F2"}]},
                                {"param": "m"},
                            ],
                        }
                    ],
                },
                "misconception": "direction_sign_error",
                "explanation": "Reverses the direction of the acceleration.",
            },
            {
                "recipe": {"op": "mul", "args": [{"param": "F1"}, {"param": "m"}]},
                "misconception": "multiplied_instead_of_divided",
                "explanation": "Multiplies the push by the mass.",
            },
            {
                "recipe": {"const": "0"},
                "misconception": "other",
                "explanation": "Assumes there is no acceleration.",
            },
        ],
    },
}

PHYSICS_CHECKS = (
    PHYSICS_ITEM_TOOL,
    PHYSICS_OPTIONS_TOOL,
    PHYSICS_TEXT_TOOL,
    PHYSICS_FIDELITY_TOOL,
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)


def physics_proposal() -> PhysicsTemplateContent:
    return PhysicsTemplateContent.model_validate({"item": PHYSICS_ITEM})


class ScriptedPhysicsProvider:
    """Return one known physics proposal and the physics check plan."""

    provider = "stub"
    model = "stub-model"

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        assert request.response_model
        return PlannedGenerationResponse(
            proposal=physics_proposal(),
            checks=[{"name": name, "arguments": {}} for name in PHYSICS_CHECKS],
        )


class ScriptedH2PhysicsProvider(ScriptedPhysicsProvider):
    def __init__(self, unsupported: H2CompactUnsupported | None = None) -> None:
        self.unsupported = unsupported

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        if self.unsupported is not None:
            return PlannedGenerationResponse(
                proposal=self.unsupported,
                checks=[{"name": PHYSICS_ITEM_TOOL, "arguments": {}}],
            )
        skill = requested_skill(request.messages[-1]["content"])
        if skill is not None:
            return PlannedGenerationResponse(
                proposal=H2StatementProposal(skill_id=skill, key=0, distractors=[1, 2, 4, 8, 3, 5]),
                checks=[{"name": PHYSICS_ITEM_TOOL, "arguments": {}}],
            )
        proposal = H2CompactProposal.model_validate(
            {
                "family_id": "wave_speed",
                "answer": "f*w",
                "distractors": [
                    {"expression": expression, "explanation": "Test recipe."}
                    for expression in ("f*w", "f+w", "f-w", "f/w", "w/f", "f")
                ],
            }
        )
        return PlannedGenerationResponse(
            proposal=proposal,
            checks=[{"name": PHYSICS_ITEM_TOOL, "arguments": {}}],
        )


class ScriptedPhysicsBlindParser:
    """Return the correct blind reading of each rendered physics variant."""

    provider = "stub"
    model = "stub-blind-parser"

    def __init__(self) -> None:
        spec = PhysicsTemplateSpec.model_validate(PHYSICS_ITEM)
        self.readings = {
            render_question(spec, values): identity_reconstruction(
                PhysicsFidelitySubject(template=spec, parameters=values)
            )
            for values in representative_variants(spec)
        }

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> Any:
        return self.readings[request.messages[-1]["content"]]


def run_scripted_physics_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
) -> AuthoringJobRecord:
    """Run the real physics workflow with a scripted provider and real MCP checks."""
    tool_client = FastMcpToolClient(
        create_validation_server(
            difficulty_provider_factory=lambda: ScriptedJudge(),
            decision_provider_factory=lambda: ScriptedDecisions(),
            fidelity_provider_factory=lambda: ScriptedPhysicsBlindParser(),
        )
    )

    def application_factory() -> TemplateApplication:
        return TemplateApplication(tool_client=tool_client)

    return run_authoring_job_sync(
        command,
        application_factory=application_factory,
        provider_factory=lambda selection: ScriptedPhysicsProvider(),
        on_progress=on_progress,
    )


def run_scripted_h2_physics_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
    *,
    unsupported: H2CompactUnsupported | None = None,
) -> AuthoringJobRecord:
    def no_fidelity_provider() -> Any:
        raise AssertionError("Constrained H2 questions need no fidelity judge")

    client = FastMcpToolClient(
        create_validation_server(fidelity_provider_factory=no_fidelity_provider)
    )
    return run_authoring_job_sync(
        command,
        application_factory=lambda: TemplateApplication(tool_client=client),
        provider_factory=lambda selection: ScriptedH2PhysicsProvider(unsupported),
        on_progress=on_progress,
    )
