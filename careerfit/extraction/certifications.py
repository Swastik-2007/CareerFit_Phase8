"""Small, explicit certification vocabulary used by resumes and job descriptions."""

from __future__ import annotations

import re


CERTIFICATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "aws_cloud_practitioner",
        r"\b(?:aws certified cloud practitioner|cloud practitioner|clf-c0[12])\b",
    ),
    (
        "aws_solutions_architect",
        r"\b(?:aws certified solutions architect|solutions architect associate|saa-c0[23])\b",
    ),
    (
        "azure_fundamentals",
        r"\b(?:microsoft certified: azure fundamentals|azure fundamentals|az-900)\b",
    ),
    (
        "google_cloud_professional",
        r"\b(?:google cloud professional|professional cloud architect)\b",
    ),
    (
        "project_management_professional",
        r"\b(?:project management professional|pmp certification|pmp certified)\b",
    ),
    (
        "cissp",
        r"\b(?:certified information systems security professional|cissp)\b",
    ),
    (
        "comptia_security_plus",
        r"\b(?:comptia security\+|security plus certification)\b",
    ),
    (
        "tensorflow_developer",
        r"\b(?:tensorflow developer certificate|tensorflow developer certified)\b",
    ),
    (
        "oracle_java_programmer",
        r"\b(?:oracle certified professional java|oracle certified java programmer|ocjp)\b",
    ),
)


def extract_certifications(text: str) -> list[str]:
    found: list[str] = []
    for canonical, pattern in CERTIFICATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(canonical)
    return found

