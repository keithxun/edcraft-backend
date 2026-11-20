"""Request and response contracts for materials-to-practice authoring."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from edcraft_backend.schemas.ai_authoring import StrictSeed

Excerpt = Annotated[str, Field(min_length=1, max_length=60_000, pattern=r"\S")]


class MaterialPagesResponse(BaseModel):
    pages: list[str]


class MaterialExcerpts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lecture: Excerpt
    exam: Excerpt


class MaterialTaskAlternative(BaseModel):
    family: str
    task: str
    reason: str


class MaterialTaskProposal(BaseModel):
    family: str | None
    task: str
    reason: str
    alternatives: list[MaterialTaskAlternative] = Field(default_factory=list)


class MaterialPracticeRequest(MaterialExcerpts):
    family: str = Field(min_length=1, max_length=64)
    seed: StrictSeed


class MaterialPracticeQuestion(BaseModel):
    stem: str
    options: list[str]
    correct_index: int
    explanation: str


class MaterialPracticeBatch(BaseModel):
    questions: list[MaterialPracticeQuestion]
    seed: StrictSeed
    generator_version: int
