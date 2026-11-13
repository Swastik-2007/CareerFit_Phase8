"""Create ML text that excludes direct identity and contact information."""

from __future__ import annotations

import re
from typing import Iterable

from careerfit.models import CandidateProfile


EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d() .-]{7,}\d)(?!\w)")


def contains_contact_information(text: str) -> bool:
    return bool(
        EMAIL_PATTERN.search(text) or URL_PATTERN.search(text) or PHONE_PATTERN.search(text)
    )


def sanitize_model_text(text: str, *, identity_values: Iterable[str] = ()) -> str:
    """Remove known identity values and broadly formatted contact details."""

    cleaned = text
    for value in identity_values:
        if value and value.strip():
            cleaned = re.sub(re.escape(value.strip()), " ", cleaned, flags=re.IGNORECASE)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = PHONE_PATTERN.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def candidate_model_text(candidate: CandidateProfile) -> str:
    identity = candidate.identity
    values = (
        identity.name or "",
        identity.email or "",
        identity.phone or "",
        identity.linkedin_url or "",
        identity.github_url or "",
        identity.portfolio_url or "",
    )
    return sanitize_model_text(candidate.cleaned_text, identity_values=values)
