"""Strict request validation for the reusable check pilot."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from edcraft_backend.schemas.check_pilot import (
    PilotApprovalRequest,
    PilotGenerateRequest,
    PilotTemplateRequest,
)

DIGEST = "a" * 64


@pytest.mark.parametrize(
    "source",
    [
        {},
        {
            "model_profile": "server-profile",
            "connection_id": str(uuid4()),
            "model": "m",
        },
        {"connection_id": str(uuid4())},
        {"model": "m"},
        {"model_profile": ""},
    ],
)
def test_template_request_requires_exactly_one_model_source(
    source: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PilotTemplateRequest.model_validate(
            {"prompt": "Differentiate a polynomial", **source}
        )


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_template_request_rejects_blank_prompts(prompt: str) -> None:
    with pytest.raises(ValidationError):
        PilotTemplateRequest.model_validate({"prompt": prompt, "model_profile": "p"})


def test_template_request_accepts_profile_or_byok() -> None:
    profile = PilotTemplateRequest.model_validate(
        {"prompt": "Differentiate a polynomial", "model_profile": "p"}
    )
    assert profile.num_distractors == 3
    assert profile.connection_id is None
    assert profile.model is None

    connection_id = uuid4()
    byok = PilotTemplateRequest.model_validate(
        {
            "prompt": "Differentiate a polynomial",
            "connection_id": str(connection_id),
            "model": "byok-model",
            "num_distractors": 2,
        }
    )
    assert byok.connection_id == connection_id
    assert byok.model == "byok-model"
    assert byok.num_distractors == 2


@pytest.mark.parametrize("num_distractors", [0, 11, True, 2.0, "3"])
def test_template_request_bounds_distractors(num_distractors: object) -> None:
    with pytest.raises(ValidationError):
        PilotTemplateRequest.model_validate(
            {
                "prompt": "Differentiate a polynomial",
                "model_profile": "p",
                "num_distractors": num_distractors,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "template_sha256": DIGEST,
            "check_sha256": "not-a-digest",
            "decision": "approved",
        },
        {"template_sha256": DIGEST, "check_sha256": DIGEST, "decision": "maybe"},
        {"template_sha256": DIGEST, "check_sha256": DIGEST},
        {"template_sha256": DIGEST, "decision": "approved"},
    ],
)
def test_approval_request_requires_exact_digests_and_decision(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PilotApprovalRequest.model_validate(payload)


def test_approval_request_accepts_approved_and_rejected() -> None:
    for decision in ("approved", "rejected"):
        request = PilotApprovalRequest.model_validate(
            {
                "template_sha256": DIGEST,
                "check_sha256": DIGEST,
                "decision": decision,
            }
        )
        assert request.decision == decision


def test_generate_request_defaults_to_one_and_bounds_values() -> None:
    request = PilotGenerateRequest.model_validate({"seed": 0})
    assert request.quantity == 1

    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": 0, "quantity": 0})
    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": 0, "quantity": 51})
    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": -1})
    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": True})
    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": 1.5})
    with pytest.raises(ValidationError):
        PilotGenerateRequest.model_validate({"seed": 2_147_483_647, "quantity": 2})
