"""Request and response contracts for materials-to-practice authoring."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edcraft_backend.schemas.ai_authoring import StrictSeed

Excerpt = Annotated[str, Field(min_length=1, max_length=60_000, pattern=r"\S")]


class MaterialPagesResponse(BaseModel):
    pages: list[str]


class MaterialPageRef(BaseModel):
    document: str = Field(min_length=1, max_length=300, pattern=r"\S")
    page: int = Field(ge=1, le=300)


class MaterialSourceRef(MaterialPageRef):
    """A reviewer-supplied physical page reference, retained for exact request replay."""

    role: Literal["lecture", "exam"]
    edited: bool = False


class MaterialPage(MaterialPageRef):
    text: str = Field(max_length=60_000)


class MaterialCollection(BaseModel):
    pages: list[MaterialPage] = Field(min_length=1, max_length=750)

    @model_validator(mode="after")
    def check_collection(self) -> "MaterialCollection":
        refs = {(page.document, page.page) for page in self.pages}
        if len(refs) != len(self.pages):
            raise ValueError("Each document and page number must identify one page.")
        if sum(len(page.text) for page in self.pages) > 4_000_000:
            raise ValueError("Select a collection of at most four million characters.")
        return self


class MaterialRetrievalRequest(MaterialCollection):
    strategy: Literal["selected", "keyword", "semantic"]
    topic: str = Field(default="", max_length=500)
    selected: list[MaterialPageRef] = Field(default_factory=list, max_length=750)
    limit: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def unique_selection(self) -> "MaterialRetrievalRequest":
        if len({(ref.document, ref.page) for ref in self.selected}) != len(self.selected):
            raise ValueError("Select each source page once.")
        return self


class MaterialTopicList(BaseModel):
    topics: list[str]


class MaterialEvidence(MaterialPage):
    score: float | None


class MaterialRetrievalResult(MaterialTopicList):
    evidence: list[MaterialEvidence]


class MaterialExcerpts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sources: list[MaterialSourceRef] = Field(default_factory=list, max_length=750)
    lecture: Excerpt
    exam: str = Field(default="", max_length=24_000)
    topic: str = Field(default="", max_length=500)


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
    sources: list[MaterialSourceRef]
    input_sha256: str
