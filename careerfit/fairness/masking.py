"""Conservative text masking that preserves structured qualifications."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from careerfit.ml.features import EMAIL_PATTERN, PHONE_PATTERN, URL_PATTERN
from careerfit.models import CandidateProfile, IdentityInfo

from .models import MaskingSummary


PRONOUN_PATTERN = re.compile(
    r"\b(?:he|him|his|himself|she|her|hers|herself|mr|mrs|ms|miss)\b\.?",
    re.IGNORECASE,
)

# The negative lookahead avoids absorbing common degree/field words when an
# institution shares a line with a qualification. Acronyms are not guessed:
# ambiguous tokens such as "MIT" can also be ordinary words or technologies.
INSTITUTION_PATTERN = re.compile(
    r"\b(?:(?!(?:Bachelor|Master|Doctor|B\.?\s?Tech|M\.?\s?Tech|"
    r"Computer|Data|Machine|Science|Engineering)\b)"
    r"[A-Z][A-Za-z&.'-]*[ \t]+){0,6}"
    r"(?i:University|College|Institute|School|Academy)\b"
    r"(?:[ \t]+(?i:of|for)[ \t]+(?:[A-Z][A-Za-z&.'-]*(?:[ \t]+|$)){1,6})?"
)
INSTITUTION_ACRONYM_PATTERN = re.compile(
    r"(?m)^\s*(?:IIT|NIT|IIIT|BITS|LPU|MIT|UCLA|NYU)(?:\s|[|,;-]|$)"
)


@dataclass(slots=True)
class MaskedCandidate:
    profile: CandidateProfile
    summary: MaskingSummary


def _remove_literal(text: str, value: str | None) -> tuple[str, int]:
    if not value or not value.strip():
        return text, 0
    return re.subn(re.escape(value.strip()), " ", text, flags=re.IGNORECASE)


def _normalize_masked_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    compact: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = is_blank
    return "\n".join(compact).strip()


def mask_candidate(candidate: CandidateProfile) -> MaskedCandidate:
    """Remove selected identity proxies while retaining structured qualifications.

    The returned profile keeps the same skills, experience, degrees, fields of
    study, and certifications. Only free text and identity fields are changed.
    """

    text = candidate.cleaned_text
    identity = candidate.identity

    text, name_count = _remove_literal(text, identity.name)

    contact_count = 0
    for value in (
        identity.email,
        identity.phone,
        identity.linkedin_url,
        identity.github_url,
        identity.portfolio_url,
    ):
        text, count = _remove_literal(text, value)
        contact_count += count
    for pattern in (EMAIL_PATTERN, URL_PATTERN, PHONE_PATTERN):
        text, count = pattern.subn(" ", text)
        contact_count += count

    text, pronoun_count = PRONOUN_PATTERN.subn(" ", text)
    acronym_warning = bool(INSTITUTION_ACRONYM_PATTERN.search(text))
    text, institution_count = INSTITUTION_PATTERN.subn(" ", text)

    summary = MaskingSummary(
        name_occurrences=name_count,
        contact_occurrences=contact_count,
        pronoun_occurrences=pronoun_count,
        institution_occurrences=institution_count,
        institution_acronym_warning=acronym_warning,
    )
    masked_profile = replace(
        candidate,
        identity=IdentityInfo(),
        cleaned_text=_normalize_masked_text(text),
    )
    return MaskedCandidate(profile=masked_profile, summary=summary)
