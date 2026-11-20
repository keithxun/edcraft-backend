"""Extract the text of a PDF while keeping its page boundaries and spacing."""

import asyncio
import os
from contextlib import suppress
from typing import BinaryIO, cast

from fastapi import status

from edcraft_backend.exceptions import EdCraftBaseException

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 300
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024
EXTRACTION_TIMEOUT_SECONDS = 10


class MaterialPdfError(EdCraftBaseException):
    """A PDF extraction failure that is safe to show to the caller."""


async def extract_pages(pdf: BinaryIO) -> list[str]:
    """Run Poppler in a child process with a limited page range, time and output."""
    try:
        # The child reads the descriptor directly, bypassing Python's buffered file offset.
        descriptor = pdf.fileno()
        os.lseek(descriptor, 0, os.SEEK_SET)
        process = await asyncio.create_subprocess_exec(
            "pdftotext",
            "-layout",
            "-enc",
            "UTF-8",
            "-eol",
            "unix",
            "-l",
            str(MAX_PDF_PAGES + 1),
            "-",
            "-",
            stdin=descriptor,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise MaterialPdfError(
            "PDF text extraction is unavailable.", status.HTTP_503_SERVICE_UNAVAILABLE
        ) from exc

    stdout = cast(asyncio.StreamReader, process.stdout)
    try:
        async with asyncio.timeout(EXTRACTION_TIMEOUT_SECONDS):
            try:
                await stdout.readexactly(MAX_EXTRACTED_BYTES + 1)
            except asyncio.IncompleteReadError as exc:
                output = exc.partial
            else:
                raise MaterialPdfError(
                    "The PDF contains too much text to extract.",
                    status.HTTP_413_CONTENT_TOO_LARGE,
                )
            if await process.wait():
                raise MaterialPdfError(
                    "The PDF could not be read. Check that it opens without a password.",
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                )
    except TimeoutError as exc:
        raise MaterialPdfError(
            "PDF extraction took too long. Try a smaller PDF.",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ) from exc
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.communicate()

    pages = output.decode("utf-8").removesuffix("\f").split("\f")
    if len(pages) > MAX_PDF_PAGES:
        raise MaterialPdfError(
            f"Upload a PDF with at most {MAX_PDF_PAGES} pages.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )
    if not any(page.strip() for page in pages):
        raise MaterialPdfError(
            "The PDF has no extractable text. Upload a text-based PDF.",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    return pages
