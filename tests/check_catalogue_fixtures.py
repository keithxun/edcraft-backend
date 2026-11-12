"""Model-free structured proposals for catalogue integration tests.

Every payload is validated by the real AI-owned Pydantic model carried on the
provider request. The helper is deliberately thin: it mirrors what a planning
model would return, never backend mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIFFERENTIATE = "polynomial.differentiate/1"
EQUATION = "polynomial.equation-real/1"
INEQUALITY = "polynomial.inequality-real/1"

CHECK_PLAN = "check_plan"
CHECK_RECIPE = "check_recipe"
POLYNOMIAL_FAMILY = "polynomial_family"

SOLVE_OPERATION_SET = "polynomial-solve-real/1"
DERIVATIVE_OPERATION_SET = "polynomial-derivative/1"


def plan(
    contract_id: str,
    *,
    relation: str = "none",
    degree: int = 2,
    saved_check_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A supported structured plan inside one trusted contract."""
    solve = contract_id != DIFFERENTIATE
    return {
        "contract_id": contract_id,
        "goal": {
            "operation": "solve" if solve else "differentiate",
            "relation": relation,
            "min_degree": degree,
            "max_degree": degree,
            "max_abs_coefficient": 6,
            "answer_type": "solution_set" if solve else "expression",
            "requires_complete_solution_set": solve,
        },
        "reason": (
            "The selected contract checks the complete answer to the declared mathematical task."
        ),
        "saved_check_ref": saved_check_ref,
    }


def unsupported_plan(
    reason: str = "LU decomposition is not a supported contract.",
) -> dict[str, Any]:
    """A plan that explicitly reports a missing developer-owned capability."""
    return {
        "supported": False,
        "development_needed": reason,
        "contract_id": None,
        "goal": None,
        "reason": "",
        "saved_check_ref": None,
    }


def check(contract_id: str, *, operation: str | None = None) -> dict[str, Any]:
    """A complete-answer recipe proposal for one trusted contract."""
    solve = contract_id != DIFFERENTIATE
    first_operation = operation or ("relation.solve_real" if solve else "poly.differentiate")
    return {
        "title": ("Complete real solution set" if solve else "Polynomial derivative equivalence"),
        "description": (
            "Compare the complete derived answer with the candidate using "
            "supported typed operations."
        ),
        "recipe": {
            "operation_set": (SOLVE_OPERATION_SET if solve else DERIVATIVE_OPERATION_SET),
            "steps": [
                {
                    "id": "expected",
                    "operation": first_operation,
                    "version": "1",
                    "arguments": {
                        ("problem" if solve else "polynomial"): {
                            "source": "input",
                            "name": "problem",
                        }
                    },
                },
                {
                    "id": "out",
                    "operation": "set.equal" if solve else "poly.equivalent",
                    "version": "1",
                    "arguments": {
                        "expected": {"source": "step", "name": "expected"},
                        "actual": {"source": "input", "name": "candidate"},
                    },
                },
            ],
            "output": "out",
        },
    }


def template(
    contract_id: str,
    *,
    relation: str = "none",
    degree: int = 2,
) -> dict[str, Any]:
    """A bounded family proposal within the selected plan."""
    solve = contract_id != DIFFERENTIATE
    family: dict[str, Any] = {
        "variable": "x",
        "degree": degree,
        "num_distractors": 3,
        "coefficients": [
            {
                "exponent": exponent,
                "values": [1, 2] if exponent == degree else [-6, -2, 0, 2],
            }
            for exponent in range(degree + 1)
        ],
    }
    if solve:
        family["relation"] = relation
    return {
        "contract_id": contract_id,
        "reason": ("A bounded parameterized polynomial task within the selected contract."),
        ("solve_family" if solve else "family"): family,
    }


def differentiation_script() -> dict[str, list[dict[str, Any]]]:
    return {
        CHECK_PLAN: [plan(DIFFERENTIATE, degree=2)],
        CHECK_RECIPE: [check(DIFFERENTIATE)],
        POLYNOMIAL_FAMILY: [template(DIFFERENTIATE, degree=2)],
    }


def solve_script(
    contract_id: str,
    *,
    relation: str,
    degree: int = 2,
    saved_check_ref: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    script: dict[str, list[dict[str, Any]]] = {
        CHECK_PLAN: [
            plan(
                contract_id,
                relation=relation,
                degree=degree,
                saved_check_ref=saved_check_ref,
            )
        ],
        POLYNOMIAL_FAMILY: [template(contract_id, relation=relation, degree=degree)],
    }
    if saved_check_ref is None:
        script[CHECK_RECIPE] = [check(contract_id)]
    return script


def invalid_solve_script(
    plan_payload: dict[str, Any],
    *,
    check_contract: str = EQUATION,
    template_contract: str = EQUATION,
    relation: str = "eq",
    degree: int = 2,
    include_check: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    script: dict[str, list[dict[str, Any]]] = {
        CHECK_PLAN: [plan_payload],
        POLYNOMIAL_FAMILY: [template(template_contract, relation=relation, degree=degree)],
    }
    if include_check:
        script[CHECK_RECIPE] = [check(check_contract)]
    return script


@dataclass
class CatalogueProvider:
    """Serve queued structured proposals keyed by the request schema name."""

    script: dict[str, list[dict[str, Any]]]
    calls: list[str] = field(default_factory=list)
    call_details: list[dict[str, str]] = field(default_factory=list)
    closed: bool = False
    provider: str = "stub"
    model: str = "stub-catalogue"
    _used: dict[str, int] = field(default_factory=dict)

    def generation_settings(self) -> dict[str, Any]:
        return {"mode": "scripted", "reasoning_effort": "none"}

    def generate(self, request: Any) -> Any:
        name = request.schema_name
        self.calls.append(name)
        self.call_details.append(
            {
                "schema_name": name,
                "prompt_version": request.prompt_version,
                "response_model": request.response_model.__name__,
            }
        )
        if name not in self.script:
            raise AssertionError(
                f"unexpected structured role {name!r}; expected {sorted(self.script)}"
            )
        payloads = self.script[name]
        index = min(self._used.get(name, 0), len(payloads) - 1)
        self._used[name] = index + 1
        return request.response_model.model_validate(payloads[index])

    def close(self) -> None:
        self.closed = True


@dataclass
class ProviderScripts:
    """Create one provider per backend author request from ordered scripts."""

    scripts: list[dict[str, list[dict[str, Any]]]]
    providers: list[CatalogueProvider] = field(default_factory=list)

    def build(self, _: Any) -> CatalogueProvider:
        if not self.scripts:
            raise AssertionError("no provider script was queued for this request")
        provider = CatalogueProvider(self.scripts.pop(0))
        self.providers.append(provider)
        return provider


__all__ = [
    "CHECK_PLAN",
    "CHECK_RECIPE",
    "CatalogueProvider",
    "DIFFERENTIATE",
    "EQUATION",
    "INEQUALITY",
    "POLYNOMIAL_FAMILY",
    "ProviderScripts",
    "check",
    "differentiation_script",
    "invalid_solve_script",
    "plan",
    "solve_script",
    "template",
    "unsupported_plan",
]
