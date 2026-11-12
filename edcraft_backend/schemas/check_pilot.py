"""Strict HTTP contracts for the reusable check authoring pilot.

Response bodies are AI-owned Pydantic records. This module only owns the
backend request shapes, so nested recipe/polynomial structures are never
duplicated here.
"""

from typing import Literal

from edcraft_validator.domains.code.code_types import DEFAULT_DISTRACTORS
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edcraft_backend.schemas.ai_authoring import (
    MAX_SEED,
    AuthoringModelSelection,
    DistractorCount,
    Sha256,
    StrictQuantity,
    StrictSeed,
)

Decision = Literal["approved", "rejected"]


class _StrictRequest(BaseModel):
    """Reject unknown fields and implicit numeric coercions."""

    model_config = ConfigDict(extra="forbid", strict=True)


class PilotTemplateRequest(AuthoringModelSelection):
    """Choose exactly one existing author model and request a pilot template."""

    prompt: str = Field(min_length=1, max_length=4000)
    num_distractors: DistractorCount = DEFAULT_DISTRACTORS

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class PilotApprovalRequest(_StrictRequest):
    """Human decision bound to the exact template and check digests shown."""

    template_sha256: Sha256
    check_sha256: Sha256
    decision: Decision


class PilotGenerateRequest(_StrictRequest):
    """Deterministic seeded generation from an approved qualified template."""

    seed: StrictSeed
    quantity: StrictQuantity = 1

    @model_validator(mode="after")
    def validate_seed_range(self) -> "PilotGenerateRequest":
        if self.seed + self.quantity - 1 > MAX_SEED:
            raise ValueError("seed plus quantity exceeds the supported seed range")
        return self


__all__ = [
    "Decision",
    "PilotApprovalRequest",
    "PilotGenerateRequest",
    "PilotTemplateRequest",
]
