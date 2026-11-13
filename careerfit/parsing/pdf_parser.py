"""PDF extraction using pypdf."""

from __future__ import annotations

from pathlib import Path
import re

from careerfit.models import ParsedDocument

from .document_parser import DocumentParseError


def extract_pdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise DocumentParseError(
            "PDF support requires pypdf. Run: pip install -r requirements.txt"
        ) from exc

    warnings: list[str] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Could not open PDF '{path.name}': {exc}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise DocumentParseError(
                f"PDF '{path.name}' is encrypted and could not be opened."
            ) from exc
        if not unlocked:
            raise DocumentParseError(
                f"PDF '{path.name}' is password-protected. Provide an unlocked copy."
            )
        warnings.append("The PDF was encrypted but allowed empty-password access.")

    pages: list[str] = []
    links: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            if page_number == 1:
                page_text = repair_name_header(page, page_text)
        except Exception as exc:
            warnings.append(f"Page {page_number} could not be extracted: {exc}")
            page_text = ""
        pages.append(page_text)
        try:
            for annotation_ref in page.get("/Annots", []):
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                uri = action.get("/URI") if action else None
                if uri and str(uri) not in links:
                    links.append(str(uri))
        except Exception as exc:
            warnings.append(f"Links on page {page_number} could not be inspected: {exc}")

    raw_text = "\n\n".join(pages)
    if len(raw_text.strip()) < 80:
        warnings.append(
            "Very little text was extracted. This may be a scanned PDF and may require OCR."
        )

    return ParsedDocument(
        source_path=str(path),
        document_type="pdf",
        raw_text=raw_text,
        page_count=len(reader.pages),
        warnings=warnings,
        links=links,
    )


def repair_name_header(page, text: str) -> str:
    """Repair small-cap header spacing using layout evidence, never guessed names."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if index is None:
        return text
    original = lines[index].strip()
    if not re.fullmatch(r'[A-Z ]{3,60}', original):
        return text
    if not any(len(word) == 1 for word in original.split()):
        return text
    try:
        layout = page.extract_text(extraction_mode='layout') or ''
    except Exception:
        return text
    header = next((line.strip() for line in layout.splitlines() if line.strip()), '')
    candidate = re.sub(r'\s+', ' ', header)
    if (not re.fullmatch(r'[A-Z]+(?: [A-Z]+){1,3}', candidate)
            or candidate.replace(' ', '') != original.replace(' ', '')):
        return text
    ending = '\n' if lines[index].endswith('\n') else ''
    lines[index] = candidate + ending
    return ''.join(lines)
