"""Resume-safe text normalization and section detection."""

from __future__ import annotations

import re
import unicodedata


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "objective"),
    "skills": (
        "skills",
        "technical skills",
        "core skills",
        "technologies",
        "technical proficiency",
    ),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "internships",
    ),
    "education": ("education", "academic background", "academic qualifications"),
    "projects": ("projects", "academic projects", "personal projects"),
    "certifications": ("certifications", "certificates", "courses"),
    "achievements": ("achievements", "awards", "honors"),
    "requirements": (
        "requirements",
        "minimum qualifications",
        "required qualifications",
        "mandatory requirements",
        "mandatory skills",
        "must have",
        "qualifications",
        "eligibility criteria",
    ),
    "preferred": (
        "preferred qualifications",
        "preferred skills",
        "desired qualifications",
        "nice to have",
        "good to have",
        "bonus skills",
    ),
    "responsibilities": (
        "responsibilities",
        "key responsibilities",
        "roles and responsibilities",
        "what you will do",
    ),
    "about_role": ("about the role", "role overview", "job overview"),
}


def clean_text(text: str) -> str:
    """Normalize extracted text without damaging tokens such as C++, C#, or .NET."""

    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")

    # Rejoin words split by PDF line wrapping: "develop-\nment" -> "development".
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
    normalized = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]", "-", normalized)
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        is_blank = not line
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def _match_section_header(line: str) -> tuple[str | None, str]:
    candidate = line.strip().strip("|_- ")
    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            exact = re.compile(rf"{re.escape(alias)}\s*:?", flags=re.IGNORECASE)
            if exact.fullmatch(candidate):
                return canonical, ""
            prefix = re.match(
                rf"^{re.escape(alias)}\s*:\s*(.+)$",
                candidate,
                flags=re.IGNORECASE,
            )
            if prefix:
                return canonical, prefix.group(1).strip()
    return None, ""


def split_sections(text: str) -> dict[str, str]:
    """Split cleaned resume text into common logical sections.

    Text before the first recognized heading is stored under ``preamble``.
    """

    sections: dict[str, list[str]] = {"preamble": []}
    current_section = "preamble"

    for line in text.splitlines():
        section_name, inline_content = _match_section_header(line)
        if section_name:
            current_section = section_name
            sections.setdefault(current_section, [])
            if inline_content:
                sections[current_section].append(inline_content)
            continue
        sections.setdefault(current_section, []).append(line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
        if "\n".join(lines).strip()
    }
