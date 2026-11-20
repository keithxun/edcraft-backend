from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from tokenizers import Tokenizer, models

from edcraft_backend.services import material_embeddings
from edcraft_backend.services.material_retrieval import (
    Page,
    SemanticRetrievalUnavailable,
    retrieve,
    topics,
)

PAGES: list[Page] = [
    {
        "document": "Automata.pdf",
        "page": 4,
        "text": "Deterministic finite automata\n\nA transition function maps a state and symbol.",
    },
    {
        "document": "Probability.pdf",
        "page": 3,
        "text": "Cumulative distribution functions\n\nEach probability is obtained by integration.",
    },
    {
        "document": "Automata.pdf",
        "page": 5,
        "text": "Minimising automata\n\nTwo states are equivalent if they accept the same words.",
    },
    {"document": "Automata.pdf", "page": 6, "text": "   "},
]


def test_selected_pages_preserve_text_and_selection_order() -> None:
    result = retrieve(PAGES, "unrelated", "selected", [("Automata.pdf", 5), ("Automata.pdf", 6)], 1)

    assert result == [{**PAGES[2], "score": None}, {**PAGES[3], "score": None}]


def test_unknown_selected_page_is_reported() -> None:
    with pytest.raises(ValueError, match="Unknown selected page"):
        retrieve(PAGES, "automata", "selected", [("Probability.pdf", 4)], 3)


def test_keyword_retrieval_improves_rank_for_more_matching_terms() -> None:
    result = retrieve(PAGES, "Deterministic automata", "keyword", [], 5)

    assert [(page["document"], page["page"]) for page in result] == [
        ("Automata.pdf", 4),
        ("Automata.pdf", 5),
    ]
    assert result[0]["score"] is not None and result[1]["score"] is not None
    assert result[0]["score"] > result[1]["score"] > 0


def test_keyword_retrieval_respects_limit_and_has_no_arbitrary_matches() -> None:
    assert len(retrieve(PAGES, "automata", "keyword", [], 1)) == 1
    assert retrieve(PAGES, "water lilies", "keyword", [], 3) == []


def test_topics_include_source_headings_once_and_skip_uninformative_pages() -> None:
    repeated: Page = {"document": "Automata.pdf", "page": 7, "text": PAGES[0]["text"]}
    assert topics([*PAGES, repeated]) == [
        "Deterministic finite automata",
        "Cumulative distribution functions",
        "Minimising automata",
    ]


def test_topics_prefer_toc_headings_and_remove_toc_page_numbers() -> None:
    pages: list[Page] = [
        {"document": "Notes.pdf", "page": 1, "text": "ECOLOGY\n  Jordan Lee     Casey Tan\n"},
        {
            "document": "Notes.pdf",
            "page": 2,
            "text": "Contents\n1 Habitat fragmentation       7\n  1.1 Wildlife corridors . . . 12",
        },
        {
            "document": "Notes.pdf",
            "page": 3,
            "text": "2 Waterways             19\n  2.1 River connectivity......23\n  "
            "2.2 Stream restoration … 25",
        },
        {
            "document": "Notes.pdf",
            "page": 4,
            "text": "Chapter 1\nHabitat fragmentation\n\nA landscape may be divided into patches.",
        },
    ]

    assert topics(pages) == [
        "Habitat fragmentation",
        "Wildlife corridors",
        "Waterways",
        "River connectivity",
        "Stream restoration",
    ]


def test_topics_prefer_slide_titles_and_collapse_continued_titles() -> None:
    pages: list[Page] = [
        {
            "document": "Slides.pdf",
            "page": 1,
            "text": "Computing Lab Lecture #4\nLinux File Permissions (1/2)\n"
            "• File owners have rights",
        },
        {
            "document": "Slides.pdf",
            "page": 2,
            "text": "Linux File Permissions (2/2)\n• Reading and writing files",
        },
        {"document": "Slides.pdf", "page": 3, "text": "Learning Objectives\n• Read a directory"},
        {
            "document": "Slides.pdf",
            "page": 4,
            "text": "Directory Access in Unix\n\nExample:\n"
            "1.1 open the directory\n1.2 inspect the entries",
        },
    ]

    assert topics(pages) == ["Linux File Permissions", "Directory Access in Unix"]


def test_topics_use_numbered_sections_beyond_opening_lines() -> None:
    pages: list[Page] = [
        {
            "document": "Statistics.pdf",
            "page": 1,
            "text": "SAMPLE ESTIMATION\nAlex Liang\n",
        },
        {
            "document": "Statistics.pdf",
            "page": 2,
            "text": "The discussion continues\nfrom the previous page.\n\n"
            "Chapter 2\nSampling methods\n\nA random sample is one approach.\n\n"
            "2.1     Stratified sampling\nEach stratum contributes observations.\n\n"
            "2.2     Cluster sampling\nRandomly select clusters.",
        },
    ]

    assert topics(pages) == ["Sampling methods", "Stratified sampling", "Cluster sampling"]


def test_missing_semantic_model_is_reported_without_keyword_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EDCRAFT_MATERIAL_EMBEDDING_DIR", str(tmp_path))

    with pytest.raises(SemanticRetrievalUnavailable, match="model is unavailable"):
        retrieve(PAGES, "Deterministic automata", "semantic", [], 3)


def test_semantic_cache_reuses_content_without_ranking_other_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "water": 4, "soil": 5}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    encoded_words: list[int] = []

    def infer(
        _: list[str], inputs: dict[str, npt.NDArray[np.int64]]
    ) -> list[npt.NDArray[np.float32]]:
        ids = inputs["input_ids"]
        encoded_words.extend(ids[:, 1].tolist())
        return [np.eye(len(vocabulary), dtype=np.float32)[ids]]

    model = material_embeddings._MiniLM(tokenizer, SimpleNamespace(run=infer))
    monkeypatch.setattr(material_embeddings, "_load_model", lambda _: model)
    water: Page = {"document": "Water.pdf", "page": 1, "text": "water"}
    soil: Page = {"document": "Soil.pdf", "page": 1, "text": "soil"}

    assert retrieve([soil, water], "water", "semantic", [], 1)[0]["document"] == "Water.pdf"
    assert encoded_words == [4, 5, 4]

    copy: Page = {"document": "My revision notes.pdf", "page": 3, "text": "soil"}
    warm = retrieve([copy], "water", "semantic", [], 2)
    assert [(page["document"], page["page"]) for page in warm] == [(copy["document"], 3)]
    assert encoded_words == [4, 5, 4, 4]
