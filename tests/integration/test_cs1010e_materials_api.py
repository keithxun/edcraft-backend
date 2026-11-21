"""HTTP tests for the materials-to-practice pilot."""

import hashlib
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import SpooledTemporaryFile
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from starlette import formparsers

from edcraft_backend.dependencies import get_current_user
from edcraft_backend.main import app
from edcraft_backend.models.user import User
from edcraft_backend.services import material_pdf

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


async def test_pdf_upload_reports_unavailable_extractor(
    materials_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    pdf = sample_pdf("BT /F1 12 Tf 72 720 Td (Lecture note) Tj ET")

    response = await materials_client.post(
        f"{MATERIALS_URL}/pages", files={"file": ("material.pdf", pdf, "application/pdf")}
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "PDF text extraction is unavailable."


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


def legacy_receipt(body):
    """Reproduce the original saved batch receipt for a historical request fixture."""
    concrete = {"sources": [], "topic": "", "exam": "", **body}
    concrete.pop("seed")
    snapshot = json.dumps(concrete, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {**body, "input_sha256": hashlib.sha256(snapshot.encode()).hexdigest()}


async def test_new_uploads_use_checked_authoring_jobs(materials_client: AsyncClient):
    old = await materials_client.post(f"{MATERIALS_URL}/propose", json=EXCERPTS)
    assert old.status_code == 410
    body = {**EXCERPTS, "family": "nested-loops", "seed": 42}
    old = await materials_client.post(f"{MATERIALS_URL}/generate", json=body)
    assert old.status_code == 410
    assert "Authoring job" in old.json()["detail"]


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


async def test_public_digest_cannot_enable_retired_generator(materials_client: AsyncClient):
    payload = legacy_receipt({**EXCERPTS, "family": "nested-loops", "seed": 42})
    response = await materials_client.post(f"{MATERIALS_URL}/generate", json=payload)
    assert response.status_code == 410
