"""Dispatch document extraction to the correct file-type parser."""

from __future__ import annotations

from pathlib import Path

from careerfit.models import ParsedDocument


class DocumentParseError(RuntimeError):
    """Raised when CareerFit cannot safely extract a document."""


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_document(path: str | Path) -> ParsedDocument:
    """Extract text from a supported PDF, DOCX, or plain-text document."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise DocumentParseError(f"File does not exist: {source}")
    if not source.is_file():
        raise DocumentParseError(f"Expected a file, received: {source}")

    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentParseError(
            f"Unsupported file type '{extension or '(none)'}'. Supported: {supported}"
        )

    if extension == ".pdf":
        from .pdf_parser import extract_pdf

        return extract_pdf(source)
    if extension == ".docx":
        from .docx_parser import extract_docx

        return extract_docx(source)

    from .text_parser import extract_text_file

    return extract_text_file(source)

