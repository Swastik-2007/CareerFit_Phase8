"""DOCX extraction using python-docx."""

from __future__ import annotations

from pathlib import Path

from careerfit.models import ParsedDocument

from .document_parser import DocumentParseError


def extract_docx(path: Path) -> ParsedDocument:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise DocumentParseError(
            "DOCX support requires python-docx. Run: pip install -r requirements.txt"
        ) from exc

    try:
        document = Document(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Could not open DOCX '{path.name}': {exc}") from exc

    blocks: list[str] = []
    blocks.extend(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())

    # Resume skills and education are frequently placed inside tables.
    for table in document.tables:
        for row in table.rows:
            cell_values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cell_values:
                blocks.append(" | ".join(cell_values))

    links: list[str] = []
    for relationship in document.part.rels.values():
        if relationship.is_external and str(relationship.target_ref).startswith(("http://", "https://")):
            url = str(relationship.target_ref)
            if url not in links:
                links.append(url)

    raw_text = "\n".join(blocks)
    warnings: list[str] = []
    if len(raw_text.strip()) < 80:
        warnings.append(
            "Very little text was extracted from the DOCX; verify that it contains editable text."
        )

    return ParsedDocument(
        source_path=str(path),
        document_type="docx",
        raw_text=raw_text,
        page_count=None,
        warnings=warnings,
        links=links,
    )
