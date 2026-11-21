"""Deterministic, model-free fixtures for AI authoring tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from edcraft_validator.application.authoring_api import (
    AuthoringJobCommand,
    AuthoringJobRecord,
    AuthoringJobResponse,
    run_authoring_job_sync,
    summarize_authoring,
)
from edcraft_validator.application.authoring_contracts import (
    AuthoringError,
    AuthoringProgress,
    AuthoringResult,
    CheckExecution,
    GenerationAttempt,
)
from edcraft_validator.application.template_workflow import TemplateApplication
from edcraft_validator.artifact_contracts import ValidatedTemplateArtifact
from edcraft_validator.domains.code.code_schemas import (
    CodeQuestionInstance,
    CodeTemplateCandidate,
    CodeTemplateProposal,
    TemplateValidationSummary,
    ValidatedCodeTemplate,
)
from edcraft_validator.domains.domain_registry import create_domain
from edcraft_validator.domains.domain_support import build_review_view
from edcraft_validator.domains.math.candidate_builder import template_sha256
from edcraft_validator.domains.math.ir.schema import (
    AnswerRecipe,
    Distractor,
    Expr,
    Given,
    Goal,
    MathItemSpec,
    MathTemplateSpec,
    ParameterSpec,
    Relation,
    SymbolSpec,
    TextSpec,
)
from edcraft_validator.domains.math.math_schemas import (
    MathQuestionInstance,
    MathTemplateCandidate,
    MathTemplateContent,
    MathValidationSummary,
    ValidatedMathTemplate,
)
from edcraft_validator.domains.math.math_types import (
    MATH_ITEM_SPEC_TOOL,
    MATH_OPTIONS_TOOL,
)
from edcraft_validator.llm.forced_decision import ClassDecision
from edcraft_validator.llm.llm_contracts import PlannedGenerationResponse
from edcraft_validator.mcp.client import FastMcpToolClient
from edcraft_validator.mcp.server import create_validation_server
from edcraft_validator.validation.check_contracts import ToolEvidence, ToolFinding
from edcraft_validator.validation.check_names import (
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)
from edcraft_validator.validation.validation_contracts import ValidationEvidence
from pydantic import BaseModel, ConfigDict

from edcraft_backend.services.ai_authoring_runtime import JobRuntime
from edcraft_backend.services.provider_endpoint import ResolvedConnection

CANDIDATE: dict[str, Any] = {
    "template_id": "arithmetic.linear_sum",
    "topic": "arithmetic",
    "difficulty": "easy",
    "code": "def calculate(a, b, c):\n    return a + b - c",
    "entry_function": "calculate",
    "parameters": [
        {"name": "a", "kind": "integer", "values": [2, 4]},
        {"name": "b", "kind": "integer", "values": [5, 8]},
        {"name": "c", "kind": "integer", "values": [1, 3]},
    ],
    "question_template": "What does calculate({a}, {b}, {c}) return?",
    "answer_target": "return_value",
    "answer_expression": "a + b - c",
    "distractors": [
        {
            "expression": "a + b + c",
            "reason_template": "Adds c instead of subtracting it.",
        },
        {
            "expression": "a - b - c",
            "reason_template": "Subtracts both b and c from a.",
        },
        {"expression": "a + b", "reason_template": "Omits the final subtraction of c."},
        {
            "expression": "a * b - c",
            "reason_template": "Multiplies a and b before subtracting c.",
        },
        {
            "expression": "b - c",
            "reason_template": "Subtracts c from b without adding a.",
        },
        {
            "expression": "c - (a + b)",
            "reason_template": "Reverses the order of the final subtraction.",
        },
    ],
    "question_type": "mcq",
}

PROPOSAL_FIELDS = (
    "question_template",
    "code",
    "entry_function",
    "parameters",
    "answer_target",
    "answer_expression",
    "distractors",
)


def _sym(name: str) -> Expr:
    return Expr(sym=name)


def _param(name: str) -> Expr:
    return Expr(param=name)


def _add(*args: Expr) -> Expr:
    return Expr(op="add", args=list(args))


def _mul(*args: Expr) -> Expr:
    return Expr(op="mul", args=list(args))


def _sub(left: Expr, right: Expr) -> Expr:
    return Expr(op="sub", args=[left, right])


def _div(left: Expr, right: Expr) -> Expr:
    return Expr(op="div", args=[left, right])


def _math_item_spec() -> MathTemplateSpec:
    """A small finite linear-equation family: 3 x 2 x 2 declared variants."""
    body = MathItemSpec(
        topic="school.linear",
        symbols=[SymbolSpec(name="x", domain="real")],
        givens=[
            Given(
                id="equation",
                relation=Relation(
                    kind="eq",
                    lhs=_add(_mul(_param("a"), _sym("x")), _param("b")),
                    rhs=_param("c"),
                ),
            )
        ],
        goal=Goal(kind="solve", symbol="x", domain="real"),
        answer_recipe=AnswerRecipe(
            kind="set",
            items=[_div(_sub(_param("c"), _param("b")), _param("a"))],
        ),
        distractors=[
            Distractor(
                value=_div(_sub(_param("b"), _param("c")), _param("a")),
                misconception="SIGN_ERROR",
                reason="Negates the numerator",
            ),
            Distractor(
                value=_sub(_param("c"), _param("b")),
                misconception="NO_DIVISION",
                reason="Omits dividing by a",
            ),
            Distractor(
                value=_div(_add(_param("c"), _param("b")), _param("a")),
                misconception="ADD_NOT_SUBTRACT",
                reason="Adds b instead of subtracting",
            ),
        ],
        text=TextSpec(mode="symbolic", template="Solve {equation} for x."),
    )
    return MathTemplateSpec(
        topic="school.linear",
        parameters=[
            ParameterSpec(name="a", values=["2", "3", "5"]),
            ParameterSpec(name="b", values=["1", "4"]),
            ParameterSpec(name="c", values=["7", "10"]),
        ],
        body=body,
    )


MATH_CANDIDATE: dict[str, Any] = {
    "template_id": "math.medium.linear_equation",
    "difficulty": "medium",
    "question_type": "mcq",
    "item": _math_item_spec().model_dump(mode="json"),
}

MATH_PROPOSAL_FIELDS = ("item",)

MATH_CHECKS = (
    MATH_ITEM_SPEC_TOOL,
    MATH_OPTIONS_TOOL,
    TEMPLATE_DIFFICULTY_CHECK,
    TEMPLATE_RELEVANCE_CHECK,
)


def proposal() -> CodeTemplateProposal:
    return CodeTemplateProposal.model_validate({key: CANDIDATE[key] for key in PROPOSAL_FIELDS})


def math_proposal() -> MathTemplateContent:
    proposal = MathTemplateContent.model_validate(
        {key: MATH_CANDIDATE[key] for key in MATH_PROPOSAL_FIELDS}
    )
    assert proposal.item is not None
    proposal.item.body.distractors *= 2
    return proposal


class StubRequest(BaseModel):
    """Minimal request model for an out-of-tree test domain."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    difficulty: str
    num_distractors: int


class StubCandidate(BaseModel):
    """Minimal candidate model with the shared template_id convention."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    instructions: str
    choices: list[str]


class StubValidatedTemplate(ValidatedTemplateArtifact):
    """Validated artifact without a validation budget to test optional limits."""

    model_config = ConfigDict(extra="forbid")

    format_version: str = "1.0.0"
    template: StubCandidate


class StubQuestion(BaseModel):
    """Deterministic question model produced by the stub domain."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    artifact_id: str | None = None
    seed: int
    prompt: str
    choices: list[str]


class StubDomain:
    """In-memory domain used to prove generic service plumbing."""

    name = "stub"
    request_model = StubRequest
    candidate_model = StubCandidate
    validated_model = StubValidatedTemplate
    question_model = StubQuestion
    template_language = None
    capabilities = frozenset()

    def register_tools(self, server: Any, dependencies: Any) -> None:
        """Register the stub's single deterministic check."""

        @server.tool(name="stub_check")
        async def stub_check(candidate: dict[str, Any]) -> ToolEvidence:
            return _evidence("stub_check", "passed")

    def review_view(self, artifact: ValidatedTemplateArtifact):
        return build_review_view(artifact, language=self.template_language)

    def distractor_count(self, candidate: Any) -> int:
        return len(StubCandidate.model_validate(candidate).choices)

    def required_checks(self, candidate: Any) -> frozenset[str]:
        return frozenset({"stub_check"})

    def finalize_checked_template(
        self,
        candidate: Any,
        evidence: list[ToolEvidence],
        *,
        advisory_checks: frozenset[str],
    ) -> StubValidatedTemplate:
        template = StubCandidate.model_validate(candidate)
        return StubValidatedTemplate(artifact_id=uuid4().hex, template=template)

    def generate_question(self, validated: Any, *, seed: int) -> StubQuestion:
        if not isinstance(validated, StubValidatedTemplate):
            validated = StubValidatedTemplate.model_validate(validated.model_dump())
        template = validated.template
        return StubQuestion(
            template_id=template.template_id,
            artifact_id=validated.artifact_id,
            seed=seed,
            prompt=f"{template.instructions} Variant {seed}.",
            choices=list(template.choices),
        )


class ScriptedProvider:
    """Return one known proposal and one technical check without a model."""

    provider = "stub"
    model = "stub-model"

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        assert request.response_model
        return PlannedGenerationResponse(
            proposal=proposal(),
            checks=[{"name": "code_validate_answers_and_distractors", "arguments": {}}],
        )


def run_scripted_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
) -> AuthoringJobRecord:
    """Run the real library workflow with a scripted provider and real MCP checks."""
    return run_authoring_job_sync(
        command,
        provider_factory=lambda selection: ScriptedProvider(),
        on_progress=on_progress,
    )


class ScriptedMathProvider:
    """Return one known math proposal and the fixed math check plan."""

    provider = "stub"
    model = "stub-model"

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def generate(self, request: Any) -> PlannedGenerationResponse[Any]:
        assert request.response_model
        return PlannedGenerationResponse(
            proposal=math_proposal(),
            checks=[{"name": name, "arguments": {}} for name in MATH_CHECKS],
        )


class ScriptedDecisions:
    """Deterministic relevance classifier for model-free integration tests."""

    provider = "stub"
    model = "stub-decisions"
    threshold = 0.5

    def score(self, request_text: str, question_text: str) -> float:
        return 1.0


class ScriptedJudge:
    """Deterministic difficulty judge for model-free integration tests."""

    provider = "stub"
    model = "stub-judge"

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted"}

    def decide(self, messages: list[dict[str, Any]], labels: tuple[str, ...]) -> ClassDecision:
        return ClassDecision(
            label="medium",
            distribution={"easy": 0.05, "medium": 0.9, "hard": 0.05},
            confidence=0.9,
            served_model=self.model,
        )

    def generate(self, request: Any) -> Any:  # pragma: no cover - decisions stop the cascade
        raise AssertionError("The scripted difficulty judge only answers decisions")


def run_scripted_math_pipeline(
    command: AuthoringJobCommand,
    credentials: object | None = None,
    on_progress: Callable[[AuthoringProgress], None] | None = None,
) -> AuthoringJobRecord:
    """Run the real math workflow with a scripted provider and real MCP checks."""
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
        provider_factory=lambda selection: ScriptedMathProvider(),
        on_progress=on_progress,
    )


def make_command(job_id: UUID | None = None) -> AuthoringJobCommand:
    return AuthoringJobCommand(
        job_id=job_id or uuid4(),
        domain="code",
        author={"provider": "stub", "model": "stub-model"},
        request={
            "prompt": "Create an arithmetic tracing question",
            "difficulty": "easy",
            "num_distractors": 3,
        },
    )


def make_math_command(job_id: UUID | None = None) -> AuthoringJobCommand:
    return AuthoringJobCommand(
        job_id=job_id or uuid4(),
        domain="math",
        author={"provider": "stub", "model": "stub-model"},
        request={
            "prompt": "Create a linear equation question for x",
            "difficulty": "medium",
            "num_distractors": 3,
        },
    )


def _evidence(tool: str, status: str, **details: Any) -> ToolEvidence:
    findings = (
        []
        if status == "passed"
        else [ToolFinding(code="TECHNICAL_FAILURE", message="Check did not pass.")]
    )
    return ToolEvidence(
        tool=tool,
        version="1.0.0",
        status=status,
        findings=findings,
        details=details,
    )


def make_stub_checked_report(command: AuthoringJobCommand) -> AuthoringResult:
    """Build a checked report for the out-of-tree stub domain."""
    candidate = StubCandidate(
        template_id="stub.greeting",
        instructions="Choose the greeting.",
        choices=["Hello", "Hi", "Hey"],
    )
    artifact = StubValidatedTemplate(artifact_id=uuid4().hex, template=candidate)
    execution = CheckExecution(
        call_id="attempt-1:stub_check",
        tool="stub_check",
        advisory=False,
        requested_arguments={},
        evidence=_evidence("stub_check", "passed"),
    )
    attempt = GenerationAttempt(
        number=1,
        candidate=candidate.model_dump(mode="json"),
        executions=[execution],
        pending=[],
        complete=True,
        passed=True,
    )
    return AuthoringResult(
        status="checked",
        provider="stub",
        model="stub-model",
        domain="stub",
        request=command.request,
        prompt_version="stub-v1",
        fixed_plan=[execution.tool],
        tool_catalogue=[],
        attempts=[attempt],
        artifact=artifact,
    )


def make_checked_report(
    command: AuthoringJobCommand,
    *,
    advisory_failure: bool = False,
    advisory_unavailable: bool = False,
    technical_failure: bool = False,
) -> AuthoringResult:
    """Build a real validated artifact with safe, inspectable review evidence."""
    technical_status = "failed" if technical_failure else "passed"
    candidate = CodeTemplateCandidate.model_validate(CANDIDATE)
    validation = TemplateValidationSummary(
        checking_mode="property_based",
        examples_checked=4,
        execution_calls=4,
        max_examples=20,
        max_execution_calls=20,
        timeout_seconds=2.0,
        seed=0,
        whole_domain=True,
        evidence=[
            ValidationEvidence(check="code_validate_answers_and_distractors", status="passed")
        ],
    )
    artifact = ValidatedCodeTemplate(
        artifact_id=uuid4().hex,
        template=candidate,
        validation=validation,
    )
    checks = [
        CheckExecution(
            call_id="attempt-1:code_validate_answers_and_distractors",
            tool="code_validate_answers_and_distractors",
            advisory=False,
            requested_arguments={},
            evidence=_evidence("code_validate_answers_and_distractors", technical_status),
        )
    ]
    if advisory_failure:
        checks.append(
            CheckExecution(
                call_id=f"attempt-1:{TEMPLATE_DIFFICULTY_CHECK}",
                tool=TEMPLATE_DIFFICULTY_CHECK,
                advisory=True,
                requested_arguments={},
                evidence=_evidence(
                    TEMPLATE_DIFFICULTY_CHECK,
                    "failed",
                    advisory=True,
                    difficulty_outcome={
                        "assessment_path": "decision",
                        "requested_difficulty": "easy",
                        "estimated_difficulty": "hard",
                        "decision": "advisory_mismatch",
                        "status": "failed",
                        "reason": "Advisory estimate differs from the target.",
                    },
                ),
            )
        )
    elif advisory_unavailable:
        checks.append(
            CheckExecution(
                call_id=f"attempt-1:{TEMPLATE_DIFFICULTY_CHECK}",
                tool=TEMPLATE_DIFFICULTY_CHECK,
                advisory=True,
                requested_arguments={},
                error="raw advisory transport failure",
            )
        )
    attempt = GenerationAttempt(
        number=1,
        candidate=candidate.model_dump(mode="json"),
        executions=checks,
        pending=[],
        complete=True,
        passed=not technical_failure,
    )
    return AuthoringResult(
        status="checked",
        provider="stub",
        model="stub-model",
        domain="code",
        provider_settings={"api_key": "never-return-this"},
        request=command.request,
        prompt_version="scripted-v1",
        fixed_plan=[check.tool for check in checks],
        tool_catalogue=[],
        attempts=[attempt],
        artifact=artifact,
        reason="raw exception text must never be returned",
    )


def make_math_checked_report(command: AuthoringJobCommand) -> AuthoringResult:
    """Build a real validated math artifact with sampled review evidence."""
    candidate = MathTemplateCandidate.model_validate(MATH_CANDIDATE)
    checked_digest = template_sha256(candidate)
    domain_size = 12
    property_coverage = {
        "checking_mode": "property_based",
        "examples_checked": domain_size,
        "execution_calls": domain_size,
        "max_examples": 100,
        "max_execution_calls": 400,
        "timeout_seconds": 15.0,
        "seed": 0,
        "whole_domain": True,
    }
    validation = MathValidationSummary(
        **property_coverage,
        declared_domain_size=domain_size,
        evidence=[
            ValidationEvidence(
                check=MATH_ITEM_SPEC_TOOL,
                status="passed",
                details={"tool_version": "1.1.0", **property_coverage},
            ),
            ValidationEvidence(
                check=MATH_OPTIONS_TOOL,
                status="passed",
                details={"tool_version": "1.1.0", **property_coverage},
            ),
            ValidationEvidence(
                check=TEMPLATE_DIFFICULTY_CHECK,
                status="passed",
                details={
                    "advisory": True,
                    "tool_version": "2.0.0",
                    "subject_sha256": checked_digest,
                },
            ),
            ValidationEvidence(
                check=TEMPLATE_RELEVANCE_CHECK,
                status="passed",
                details={
                    "advisory": True,
                    "tool_version": "1.0.0",
                    "subject_sha256": checked_digest,
                    "relevant": True,
                },
            ),
        ],
    )
    artifact = ValidatedMathTemplate(
        artifact_id=uuid4().hex,
        template=candidate,
        validation=validation,
    )
    checks = [
        CheckExecution(
            call_id=f"attempt-1:{MATH_ITEM_SPEC_TOOL}",
            tool=MATH_ITEM_SPEC_TOOL,
            advisory=False,
            requested_arguments={},
            evidence=_evidence(
                MATH_ITEM_SPEC_TOOL,
                "passed",
                declared_domain_size=domain_size,
                **property_coverage,
            ),
        ),
        CheckExecution(
            call_id=f"attempt-1:{MATH_OPTIONS_TOOL}",
            tool=MATH_OPTIONS_TOOL,
            advisory=False,
            requested_arguments={},
            evidence=_evidence(
                MATH_OPTIONS_TOOL,
                "passed",
                distractors=len(candidate.item.body.distractors),
            ),
        ),
        CheckExecution(
            call_id=f"attempt-1:{TEMPLATE_DIFFICULTY_CHECK}",
            tool=TEMPLATE_DIFFICULTY_CHECK,
            advisory=True,
            requested_arguments={},
            evidence=_evidence(
                TEMPLATE_DIFFICULTY_CHECK,
                "passed",
                advisory=True,
                subject_sha256=checked_digest,
                difficulty_outcome={
                    "assessment_path": "decision",
                    "requested_difficulty": "medium",
                    "estimated_difficulty": "medium",
                    "decision": "matches",
                    "status": "passed",
                    "reason": "Advisory estimate matches the requested difficulty.",
                },
            ),
        ),
        CheckExecution(
            call_id=f"attempt-1:{TEMPLATE_RELEVANCE_CHECK}",
            tool=TEMPLATE_RELEVANCE_CHECK,
            advisory=True,
            requested_arguments={},
            evidence=_evidence(
                TEMPLATE_RELEVANCE_CHECK,
                "passed",
                advisory=True,
                subject_sha256=checked_digest,
                relevant=True,
            ),
        ),
    ]
    attempt = GenerationAttempt(
        number=1,
        candidate=candidate.model_dump(mode="json"),
        executions=checks,
        pending=[],
        complete=True,
        passed=True,
    )
    return AuthoringResult(
        status="checked",
        provider="stub",
        model="stub-model",
        domain="math",
        request=command.request,
        prompt_version="math-scripted-v1",
        fixed_plan=[check.tool for check in checks],
        tool_catalogue=[],
        attempts=[attempt],
        artifact=artifact,
    )


def make_needs_review_report(
    command: AuthoringJobCommand, *, failure: AuthoringError | None = None
) -> AuthoringResult:
    execution = CheckExecution(
        call_id="attempt-1:code_validate_answers_and_distractors",
        tool="code_validate_answers_and_distractors",
        advisory=False,
        requested_arguments={},
        evidence=_evidence("code_validate_answers_and_distractors", "failed"),
    )
    attempt = GenerationAttempt(
        number=1,
        candidate=CodeTemplateCandidate.model_validate(CANDIDATE).model_dump(mode="json"),
        executions=[execution],
        pending=[],
        complete=True,
        passed=False,
    )
    return AuthoringResult(
        status="needs_review",
        provider="stub",
        model="stub-model",
        domain="code",
        provider_settings={"api_key": "never-return-this"},
        request=command.request,
        prompt_version="scripted-v1",
        fixed_plan=[execution.tool],
        tool_catalogue=[],
        attempts=[attempt],
        reason="raw check failure text must never be returned",
        failure=failure,
    )


class StaticExecutor:
    """Return a prebuilt report for the exact submitted command."""

    def __init__(self, report_factory: Callable[[AuthoringJobCommand], AuthoringResult]):
        self._factory = report_factory
        self.calls = 0
        self.commands: list[AuthoringJobCommand] = []
        self.runtimes: list[JobRuntime] = []

    @property
    def author_connections(self) -> list[ResolvedConnection | None]:
        """Convenience view of the author role used by each job."""
        return [runtime.author if runtime.author.is_byok else None for runtime in self.runtimes]

    def __call__(
        self,
        command: AuthoringJobCommand,
        runtime: JobRuntime,
        on_progress: Callable[[AuthoringProgress], None] | None = None,
    ) -> AuthoringJobRecord:
        self.calls += 1
        self.commands.append(command)
        self.runtimes.append(runtime)
        report = self._factory(command)
        if on_progress is not None:
            on_progress(AuthoringProgress(phase="generating", attempt=1))
            on_progress(AuthoringProgress(phase="checking", attempt=1))
        summary = summarize_authoring(report)
        return AuthoringJobRecord(
            response=AuthoringJobResponse(
                job_id=command.job_id,
                state="failed" if report.status == "error" else "completed",
                result=summary,
            ),
            report=report,
        )


def generate(artifact: ValidatedCodeTemplate, seed: int) -> CodeQuestionInstance:
    result = TemplateApplication().generate_question(
        artifact, domain=create_domain("code"), seed=seed
    )
    return CodeQuestionInstance.model_validate(result.model_dump())


def generate_math(artifact: ValidatedMathTemplate, seed: int) -> MathQuestionInstance:
    result = TemplateApplication().generate_question(
        artifact, domain=create_domain("math"), seed=seed
    )
    return MathQuestionInstance.model_validate(result.model_dump())
