"""Authenticated materials-to-practice pilot endpoints."""

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from edcraft_backend.dependencies import get_current_user
from edcraft_backend.schemas.cs1010e_materials import (
    MaterialExcerpts,
    MaterialPagesResponse,
    MaterialPracticeBatch,
    MaterialPracticeRequest,
    MaterialTaskProposal,
)
from edcraft_backend.services import cs1010e_practice, material_pdf

router = APIRouter(prefix="/materials", dependencies=[Depends(get_current_user)])


@router.post(
    "/pages",
    response_model=MaterialPagesResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    },
                },
            },
        },
    },
)
async def upload_pdf_pages(request: Request) -> MaterialPagesResponse:
    """Extract selectable, layout-preserving text pages from one PDF."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
        "multipart/form-data"
    ):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PDF file.")

    async def limited_body() -> AsyncGenerator[bytes, None]:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            # Leave room for the multipart boundary and headers.
            if size > material_pdf.MAX_PDF_BYTES + 16 * 1024:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Upload a PDF up to 10 MiB.")
            yield chunk

    # Parse after sign-in and cap bytes even when the client omits Content-Length.
    try:
        form = await MultiPartParser(
            request.headers, limited_body(), max_files=1, max_fields=0
        ).parse()
    except MultiPartException as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload one PDF as 'file'.") from exc

    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload one PDF as 'file'.")
        if file.size is not None and file.size > material_pdf.MAX_PDF_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Upload a PDF up to 10 MiB.")
        if await file.read(5) != b"%PDF-":
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PDF file.")
        return MaterialPagesResponse(pages=await material_pdf.extract_pages(file.file))
    finally:
        await form.close()


@router.post("/propose", response_model=MaterialTaskProposal)
def propose_material_task(payload: MaterialExcerpts) -> MaterialTaskProposal:
    """Identify the supported practice task evidenced by both excerpts."""
    return MaterialTaskProposal.model_validate(
        cs1010e_practice.propose(payload.lecture, payload.exam)
    )


@router.post("/generate", response_model=MaterialPracticeBatch)
def generate_material_practice(payload: MaterialPracticeRequest) -> MaterialPracticeBatch:
    """Recheck the excerpts and generate a reproducible batch for the chosen task."""
    try:
        questions = cs1010e_practice.generate(
            payload.family, payload.seed, payload.lecture, payload.exam
        )
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Choose a task supported by the current lecture and exam excerpts.",
        ) from exc
    return MaterialPracticeBatch.model_validate(
        {
            "questions": questions,
            "seed": payload.seed,
            "generator_version": cs1010e_practice.VERSION,
        }
    )
