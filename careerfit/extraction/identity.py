"""Conservative identity extraction kept separate from scoring features."""

from __future__ import annotations

import re

from careerfit.models import IdentityInfo


EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])", re.I)
PHONE_CANDIDATE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){1,3}$")
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'\]\[(){}]+|"
    r"(?:linkedin\.com/in|github\.com)/[^\s<>\"'\]\[(){}]+",
    re.I,
)

NAME_BLOCKLIST = {
    "resume",
    "curriculum vitae",
    "professional summary",
    "career objective",
    "technical skills",
    "work experience",
    "academic qualifications",
}


def extract_email(text: str) -> str | None:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    for match in PHONE_CANDIDATE_PATTERN.finditer(text):
        raw = match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15:
            if raw.startswith("+"):
                return f"+{digits}"
            return digits
    return None


def extract_name(text: str) -> str | None:
    """Return a likely name from the resume header, or None when uncertain."""

    lines = [line.strip() for line in text.splitlines() if line.strip()][:10]
    for line in lines:
        candidate = re.sub(r"\s+", " ", line).strip("|- ")
        lowered = candidate.casefold()
        if lowered in NAME_BLOCKLIST:
            continue
        if EMAIL_PATTERN.search(candidate) or PHONE_CANDIDATE_PATTERN.search(candidate):
            continue
        if "http" in lowered or "linkedin" in lowered or "github" in lowered:
            continue
        if len(candidate) > 60 or not NAME_PATTERN.fullmatch(candidate):
            continue
        words = candidate.split()
        if all(word.islower() for word in words):
            continue
        return candidate.title() if candidate.isupper() else candidate
    return None


def extract_urls(text: str) -> tuple[str | None, str | None, str | None]:
    urls: list[str] = []
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,;:!?")
        if url.casefold().startswith("www."):
            url = f"https://{url}"
        elif not url.casefold().startswith(("http://", "https://")):
            url = f"https://{url}"
        if url not in urls:
            urls.append(url)

    linkedin = next((url for url in urls if "linkedin.com/" in url.casefold()), None)
    github = next((url for url in urls if "github.com/" in url.casefold()), None)
    excluded_hosts = ("linkedin.com/", "github.com/")
    portfolio = next(
        (url for url in urls if not any(host in url.casefold() for host in excluded_hosts)),
        None,
    )
    return linkedin, github, portfolio


def extract_identity(text: str) -> IdentityInfo:
    linkedin, github, portfolio = extract_urls(text)
    return IdentityInfo(
        name=extract_name(text),
        email=extract_email(text),
        phone=extract_phone(text),
        linkedin_url=linkedin,
        github_url=github,
        portfolio_url=portfolio,
    )
