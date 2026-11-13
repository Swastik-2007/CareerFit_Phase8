"""Plain-text extraction, useful for development and reproducible demos."""

from __future__ import annotations

from pathlib import Path

from careerfit.models import ParsedDocument

from .document_parser import DocumentParseError


def extract_text_file(path: Path) -> ParsedDocument:
    warnings: list[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = path.read_text(encoding="cp1252")
        except Exception as exc:
            raise DocumentParseError(f"Could not decode text file '{path.name}': {exc}") from exc
        warnings.append("The text file used CP-1252 fallback decoding instead of UTF-8.")
    except Exception as exc:
        raise DocumentParseError(f"Could not read text file '{path.name}': {exc}") from exc

    return ParsedDocument(
        source_path=str(path),
        document_type="txt",
        raw_text=raw_text,
        page_count=None,
        warnings=warnings,
    )

