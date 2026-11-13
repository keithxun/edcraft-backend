"""Model-free proposals for testing new structured shapes through HTTP."""

from __future__ import annotations

from typing import Any

from edcraft_validator.application.check_pilot.kernel_registry import kernel_contract
from edcraft_validator.application.check_pilot.models import (
    CheckPlanProposal,
    KernelGoal,
)

FAMILIES: dict[str, tuple[dict[str, Any], str, list[int]]] = {
    "polynomial.antiderivative/1": (
        {"kind": "polynomial_antiderivative", "coefficients": ["1", "2", "3"], "general": False},
        "coefficients.0",
        [1, 2, 3],
    ),
    "polynomial.factor/1": (
        {
            "kind": "polynomial-factor",
            "expression": {
                "op": "add",
                "args": [
                    {"op": "pow", "args": [{"sym": "x"}, {"num": "2"}]},
                    {"num": "-1"},
                ],
            },
        },
        "expression.args.1.num",
        [-1, -4, -9],
    ),
    "linear-system.unique/1": (
        {"kind": "linear-system-unique", "matrix": [[2, 1], [1, -1]], "rhs": [5, 1]},
        "rhs.0",
        [5, 8, 11],
    ),
    "matrix.lu/1": (
        {"kind": "matrix_lu", "matrix": [["2", "1"], ["4", "3"]]},
        "matrix.1.1",
        [3, 4, 5],
    ),
}


def script(contract_id: str) -> dict[str, list[dict[str, Any]]]:
    contract = kernel_contract(contract_id)
    assert contract is not None
    plan = CheckPlanProposal(
        contract_id=contract_id,
        reason="This request is within the selected bounded correctness contract.",
        goal=KernelGoal(
            operation=contract.operation,
            answer_type=contract.answer_type,
            min_degree=contract.min_degree,
            max_degree=contract.max_degree,
            max_abs_coefficient=contract.max_abs_coefficient,
            max_dimension=contract.max_dimension,
            requires_general_antiderivative=False,
            domain_assumptions=list(contract.domain_assumptions),
        ),
    ).model_dump(mode="json")
    problem, path, values = FAMILIES[contract_id]
    return {
        "check_plan": [plan],
        "check_recipe": [
            {
                "supported": True,
                "title": "Check " + contract_id,
                "description": "Validate the complete answer with the trusted contract.",
                "recipe": {
                    "operation_set": "structured-check/1",
                    "steps": [
                        {
                            "id": "expected",
                            "operation": "structured.expected",
                            "version": "1",
                            "arguments": {"problem": {"source": "input", "name": "problem"}},
                        },
                        {
                            "id": "out",
                            "operation": "structured.accepts",
                            "version": "1",
                            "arguments": {
                                "expected": {"source": "step", "name": "expected"},
                                "problem": {"source": "input", "name": "problem"},
                                "candidate": {"source": "input", "name": "candidate"},
                            },
                        },
                    ],
                    "output": "out",
                },
            }
        ],
        "polynomial_family": [
            {
                "supported": True,
                "contract_id": contract_id,
                "reason": "The structure is fixed and numeric leaves vary within bounds.",
                "structured_family": {
                    "contract_id": contract_id,
                    "problem": problem,
                    "bindings": [{"path": path, "values": values}],
                    "num_distractors": 3,
                },
            }
        ],
    }
