"""HTTP tests for the materials-to-practice pilot."""

import shutil
from collections.abc import AsyncIterator
from tempfile import SpooledTemporaryFile
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from starlette import formparsers

from edcraft_backend.dependencies import get_current_user
from edcraft_backend.main import app
from edcraft_backend.models.user import User
from edcraft_backend.services import cs1010e_practice, material_pdf
from edcraft_backend.services.ai_domains import LIBRARY_REVISION
from edcraft_backend.services.cs1010e.functions import MAX_OUTLINE

MATERIALS_URL = "/ai-authoring/materials"
EXCERPTS = {
    "lecture": "Nested loops run the inner loop for each outer iteration.",
    "exam": """What integer is printed by this code fragment?
int i, j, n = 0;
for (i=0; i<20; i+=2) {
    n++;
    for (j=1; j<20; j+=2) n--;
}
printf("%d", n);""",
}


@pytest.fixture
async def materials_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: User(id=uuid4()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


def sample_pdf(*page_streams: str, padding: int = 0) -> bytes:
    """Create a minimal PDF with a standard font and independent page streams."""
    kids = " ".join(f"{number} 0 R" for number in range(4, 4 + 2 * len(page_streams), 2))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Count {len(page_streams)} /Kids [{kids}] >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    for index, page_stream in enumerate(page_streams):
        content_id = 5 + 2 * index
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream_object = f"<< /Length {len(page_stream)} >>\nstream\n{page_stream}\nendstream"
        objects.append(stream_object.encode())
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    document.extend(b"% PDF padding\n" * padding)
    cross_reference = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        document.extend(f"{offset:010} 00000 n \n".encode())
    trailer = (
        f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{cross_reference}\n%%EOF\n"
    )
    document.extend(trailer.encode())
    return bytes(document)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="Poppler is required")
@pytest.mark.parametrize("padding", [0, 100_000], ids=["in-memory", "rolled-to-disk"])
async def test_pdf_upload_preserves_physical_pages_and_indentation(
    materials_client: AsyncClient,
    padding: int,
) -> None:
    pdf = sample_pdf(
        "BT /F1 12 Tf 72 720 Td (for i in range) Tj 28.8 -14 Td (print i) Tj ET",
        "",
        "BT /F1 12 Tf 72 720 Td (Exam question) Tj ET",
        padding=padding,
    )
    response = await materials_client.post(
        f"{MATERIALS_URL}/pages", files={"file": ("material.pdf", pdf, "application/pdf")}
    )
    assert response.status_code == 200, response.text
    assert [page.rstrip() for page in response.json()["pages"]] == [
        "for i in range\n    print i",
        "",
        "Exam question",
    ]


async def test_pdf_upload_rejects_other_file_content(materials_client: AsyncClient) -> None:
    response = await materials_client.post(
        f"{MATERIALS_URL}/pages",
        files={"file": ("material.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 415, response.text


async def test_pdf_upload_requires_authentication() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"{MATERIALS_URL}/pages",
            files={"file": ("material.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert response.status_code == 401, response.text


async def test_rejects_oversized_stream_even_without_content_length(
    materials_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(material_pdf, "MAX_PDF_BYTES", 1024)
    monkeypatch.setattr(formparsers.MultiPartParser, "spool_max_size", 32)
    created_files: list[SpooledTemporaryFile[bytes]] = []

    def track_spooled_file(*, max_size: int) -> SpooledTemporaryFile[bytes]:
        file = SpooledTemporaryFile(max_size=max_size, mode="w+b")
        created_files.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_spooled_file)

    async def upload() -> AsyncIterator[bytes]:
        yield b'--pdf-boundary\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        yield b"Content-Type: application/pdf\r\n\r\n%PDF-1.4\n" + b"% padding\n" * 16
        assert len(created_files) == 1 and created_files[0].name is not None
        yield b"% padding\n" * 2048
        yield b"\r\n--pdf-boundary--\r\n"

    response = await materials_client.post(
        f"{MATERIALS_URL}/pages",
        content=upload(),
        headers={"content-type": "multipart/form-data; boundary=pdf-boundary"},
    )
    assert response.status_code == 413, response.text
    assert created_files[0].closed


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="Poppler is required")
@pytest.mark.parametrize("pdf", [b"%PDF-1.4\n", sample_pdf("")], ids=["corrupt", "empty-page"])
async def test_pdf_without_readable_pages_returns_an_error(
    materials_client: AsyncClient,
    pdf: bytes,
) -> None:
    response = await materials_client.post(
        f"{MATERIALS_URL}/pages", files={"file": ("material.pdf", pdf, "application/pdf")}
    )
    assert response.status_code == 422, response.text


async def test_propose_and_generate_replay_one_supported_task(
    materials_client: AsyncClient,
) -> None:
    proposed = await materials_client.post(f"{MATERIALS_URL}/propose", json=EXCERPTS)
    assert proposed.status_code == 200, proposed.text
    result = proposed.json()
    assert result["family"] == "nested-loops"
    assert result["task"]
    assert result["alternatives"] == []

    payload = {**EXCERPTS, "family": result["family"], "seed": 42}
    generated = await materials_client.post(f"{MATERIALS_URL}/generate", json=payload)
    replay = await materials_client.post(f"{MATERIALS_URL}/generate", json=payload)
    assert generated.status_code == 200, generated.text
    batch = generated.json()
    assert batch == replay.json()
    assert batch["seed"] == 42
    assert batch["generator_version"] == cs1010e_practice.VERSION
    assert len(batch["questions"]) == 5
    for question in batch["questions"]:
        assert question["stem"] and question["explanation"]
        assert len(set(question["options"])) == 4
        assert 0 <= question["correct_index"] < len(question["options"])


async def test_propose_exposes_the_lecture_reduction_as_an_alternate_task(
    materials_client: AsyncClient,
) -> None:
    lecture = """Draw the trace tree for this recursion:
int study(int x, int y) {
    int mid;
    if (x == y) return x*x;
    else { mid=(x+y)/2; return study(x,mid) + study(mid+1,y); }
}"""
    exam = "What integer is printed by this code fragment?\n" + MAX_OUTLINE
    excerpts = {"lecture": lecture, "exam": exam}
    proposed = await materials_client.post(f"{MATERIALS_URL}/propose", json=excerpts)
    assert proposed.status_code == 200, proposed.text
    result = proposed.json()
    assert result["family"] == "recursive-max"
    (alternate,) = result["alternatives"]
    assert alternate["family"] == "recursive-square-sum"
    assert alternate["task"] and alternate["reason"]

    generated = await materials_client.post(
        f"{MATERIALS_URL}/generate", json={**excerpts, "family": alternate["family"], "seed": 11}
    )
    assert generated.status_code == 200, generated.text
    assert len(generated.json()["questions"]) == 5


async def test_generation_rechecks_current_excerpts(materials_client: AsyncClient) -> None:
    excerpts = {**EXCERPTS, "lecture": "Declaring a pointer with the star operator."}
    proposed = await materials_client.post(f"{MATERIALS_URL}/propose", json=excerpts)
    assert proposed.status_code == 200, proposed.text
    assert proposed.json()["family"] is None
    assert proposed.json()["reason"]
    assert proposed.json()["alternatives"] == []

    payload = {**excerpts, "family": "nested-loops", "seed": 42}
    generated = await materials_client.post(f"{MATERIALS_URL}/generate", json=payload)
    assert generated.status_code == 422, generated.text


@pytest.mark.parametrize("action", ["propose", "generate"])
async def test_json_materials_endpoints_require_authentication(action: str) -> None:
    payload: dict[str, str | int] = {**EXCERPTS}
    if action == "generate":
        payload.update(family="nested-loops", seed=42)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"{MATERIALS_URL}/{action}", json=payload)
    assert response.status_code == 401, response.text


async def test_generic_retrieval_respects_selected_pages_and_collection(
    materials_client: AsyncClient,
):
    pages = [
        {"document": "Week 1", "page": 1, "text": "Nested loops\nBody of the lecture."},
        {"document": "Week 2", "page": 1, "text": "Partial derivatives\nA second topic."},
    ]
    body = {
        "pages": pages,
        "topic": "partial derivatives",
        "strategy": "selected",
        "selected": [{"document": "Week 1", "page": 1}],
        "limit": 1,
    }
    explicit = await materials_client.post(f"{MATERIALS_URL}/retrieve", json=body)
    assert explicit.status_code == 200
    assert explicit.json()["evidence"][0]["text"] == pages[0]["text"]
    ranked = await materials_client.post(
        f"{MATERIALS_URL}/retrieve", json={**body, "strategy": "keyword"}
    )
    assert ranked.json()["evidence"][0]["document"] == "Week 2"
    missing = await materials_client.post(
        f"{MATERIALS_URL}/retrieve", json={**body, "pages": pages[1:]}
    )
    assert missing.status_code == 422


async def test_lecture_only_can_generate_but_rechecks_topic(materials_client: AsyncClient):
    pair = {
        "lecture": (
            "Nested loops: for (int i=0; i<3; i++) the outer loop restarts "
            "the inner loop. printf prints the result."
        ),
        "topic": "nested loops",
    }
    result = await materials_client.post(f"{MATERIALS_URL}/propose", json=pair)
    assert result.status_code == 200
    assert result.json()["family"] == "nested-loops"
    request = {**pair, "family": "nested-loops", "seed": 23}
    batch = await materials_client.post(f"{MATERIALS_URL}/generate", json=request)
    assert len(batch.json()["questions"]) == 5
    mismatch = await materials_client.post(
        f"{MATERIALS_URL}/generate", json={**request, "topic": "gene expression"}
    )
    assert mismatch.status_code == 422


async def test_unrelated_module_can_use_its_own_explicit_table(materials_client: AsyncClient):
    pair = {
        "lecture": (
            "Moon | Planet\n--- | ---\nIo | Jupiter\nTitan | Saturn\n"
            "Triton | Neptune\nPhobos | Mars\nMiranda | Uranus"
        ),
        "topic": "Moon and planet",
    }
    result = await materials_client.post(f"{MATERIALS_URL}/propose", json=pair)
    assert result.json()["family"] == "table-lookup"
    request = {**pair, "family": "table-lookup", "seed": 987}
    result = await materials_client.post(f"{MATERIALS_URL}/generate", json=request)
    assert len({q["stem"] for q in result.json()["questions"]}) == 5
    for q in result.json()["questions"]:
        expected = q["options"][q["correct_index"]]
        assert expected in pair["lecture"]
    # An exam asking for an explanation cannot be answered by looking up a named cell.
    other = await materials_client.post(
        f"{MATERIALS_URL}/propose", json={**pair, "exam": "Explain how orbital resonances develop."}
    )
    assert other.json()["family"] is None


async def test_evidence_receipt_tracks_edits_and_replay(materials_client: AsyncClient) -> None:
    refs = [
        {"document": "lecture.pdf", "page": 2, "role": "lecture", "edited": False},
        {"document": "paper.pdf", "page": 5, "role": "exam", "edited": False},
    ]
    body = {**EXCERPTS, "seed": 33, "family": "nested-loops", "sources": refs}
    first = await materials_client.post(f"{MATERIALS_URL}/generate", json=body)
    assert first.status_code == 200 and first.json()["sources"] == refs
    assert len(first.json()["input_sha256"]) == 64
    assert first.json()["validator_revision"] == LIBRARY_REVISION
    corrected = await materials_client.post(
        f"{MATERIALS_URL}/generate",
        json={
            **body,
            "lecture": EXCERPTS["lecture"] + " One correction.",
            "sources": [{**refs[0], "edited": True}, refs[1]],
        },
    )
    assert corrected.json()["input_sha256"] != first.json()["input_sha256"]
    assert (
        first.json() == (await materials_client.post(f"{MATERIALS_URL}/generate", json=body)).json()
    )


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="Poppler is required")
async def test_uploaded_glossary_uses_generic_workflow(materials_client: AsyncClient) -> None:
    lines = [
        "Thermodynamics glossary",
        "Isobaric: A process that takes place at constant pressure.",
        "Isochoric: A process that takes place at constant volume.",
        "Isothermal: A process that takes place at constant temperature.",
        "Adiabatic: A process in which no heat is exchanged.",
        "Reversible: An ideal process that can be reversed without net change.",
    ]
    stream = "BT /F1 9 Tf 60 760 Td " + " 0 -13 Td ".join(f"({s}) Tj" for s in lines) + " ET"
    response = await materials_client.post(
        f"{MATERIALS_URL}/pages",
        files={"file": ("course.pdf", sample_pdf(stream), "application/pdf")},
    )
    assert response.status_code == 200
    page = {"document": "course.pdf", "page": 1, "text": response.json()["pages"][0]}
    retrieved = await materials_client.post(
        f"{MATERIALS_URL}/retrieve",
        json={"pages": [page], "topic": "thermodynamics", "strategy": "keyword"},
    )
    lecture = retrieved.json()["evidence"][0]["text"]
    inputs = {"lecture": lecture, "topic": "thermodynamics"}
    proposed = await materials_client.post(f"{MATERIALS_URL}/propose", json=inputs)
    assert proposed.json()["family"] == "glossary-term"
    batch = await materials_client.post(
        f"{MATERIALS_URL}/generate", json={**inputs, "family": "glossary-term", "seed": 3}
    )
    assert batch.status_code == 200
    for q in batch.json()["questions"]:
        term = q["options"][q["correct_index"]]
        assert (
            next(line.split(": ", 1)[1] for line in lines if line.startswith(term + ":"))
            in q["stem"]
        )


async def test_selected_domain_scopes_proposal_generation_and_receipt(
    materials_client: AsyncClient,
):
    proposal = await materials_client.post(
        f"{MATERIALS_URL}/propose", json={**EXCERPTS, "domain": "math"}
    )
    assert proposal.status_code == 200 and proposal.json()["family"] is None
    request = {**EXCERPTS, "seed": 5, "family": "nested-loops"}
    automatic = await materials_client.post(f"{MATERIALS_URL}/generate", json=request)
    explicit = await materials_client.post(
        f"{MATERIALS_URL}/generate", json={**request, "domain": "code"}
    )
    assert automatic.status_code == explicit.status_code == 200
    assert automatic.json()["questions"] == explicit.json()["questions"]
    assert explicit.json()["input_sha256"] != automatic.json()["input_sha256"]
    assert (
        await materials_client.post(f"{MATERIALS_URL}/generate", json={**request, "domain": "math"})
    ).status_code == 422


async def test_disabled_material_domain_is_rejected_even_in_auto_mode(
    materials_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AI_AUTHORING_ENABLED_DOMAINS", "math")
    explicit = await materials_client.post(
        f"{MATERIALS_URL}/propose", json={**EXCERPTS, "domain": "code"}
    )
    assert explicit.status_code == 422
    auto = await materials_client.post(f"{MATERIALS_URL}/propose", json=EXCERPTS)
    assert auto.status_code == 200 and auto.json()["family"] is None
