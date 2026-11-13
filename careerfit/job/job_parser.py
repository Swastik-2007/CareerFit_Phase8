"""Convert an unstructured job description into explicit requirements."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from careerfit.extraction.certifications import extract_certifications
from careerfit.extraction.education import extract_education
from careerfit.extraction.skills import DEFAULT_TAXONOMY_PATH, SkillExtractor
from careerfit.models import JobRequirements
from careerfit.parsing import clean_text, extract_document, split_sections


ROLE_WORDS = {
    "analyst",
    "architect",
    "consultant",
    "developer",
    "engineer",
    "intern",
    "manager",
    "researcher",
    "scientist",
    "specialist",
}

REQUIRED_CUES = re.compile(
    r"\b(?:required|requirement|must|mandatory|minimum|at least|essential)\b",
    re.I,
)
PREFERRED_CUES = re.compile(
    r"\b(?:preferred|desirable|nice to have|good to have|bonus|advantage)\b",
    re.I,
)

CERTIFICATION_SKILL_MAP = {
    "aws_cloud_practitioner": "aws",
    "aws_solutions_architect": "aws",
    "azure_fundamentals": "azure",
    "google_cloud_professional": "gcp",
    "tensorflow_developer": "tensorflow",
    "oracle_java_programmer": "java",
}

MINIMUM_EXPERIENCE_PATTERNS = (
    re.compile(
        r"\b(?:minimum(?:\s+of)?|at least)\s*(?P<years>\d+(?:\.\d+)?)\s*"
        r"(?:\+\s*)?(?:years?|yrs?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<years>\d+(?:\.\d+)?)\s*(?:\+|[-–—]\s*\d+(?:\.\d+)?)?\s*"
        r"(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|professional\s+|work\s+)?experience\b",
        re.I,
    ),
)


def _segments(text: str) -> list[str]:
    """Create small evidence units from bullet lines and normal sentences."""

    result: list[str] = []
    for line in text.splitlines():
        cleaned_line = line.strip().lstrip("-–—* ")
        if not cleaned_line:
            continue
        parts = re.split(r"(?<=[.!?;])\s+", cleaned_line)
        result.extend(part.strip() for part in parts if part.strip())
    return result


def _cue_text(text: str, cue: re.Pattern[str]) -> str:
    return "\n".join(segment for segment in _segments(text) if cue.search(segment))


def extract_job_title(text: str) -> str:
    lines = [line.strip().strip("|- ") for line in text.splitlines() if line.strip()][:15]
    for line in lines:
        match = re.match(r"^(?:job\s+title|position|role)\s*:\s*(.+)$", line, flags=re.I)
        if match:
            return match.group(1).strip()

    blocked = {"job description", "about us", "careers", "vacancy"}
    for line in lines:
        words = re.findall(r"[A-Za-z]+", line.casefold())
        if (
            line.casefold() not in blocked
            and 1 < len(words) <= 10
            and any(word in ROLE_WORDS for word in words)
            and len(line) <= 100
        ):
            return line
    return "Untitled Role"


def extract_minimum_experience(text: str) -> float:
    values: list[float] = []
    for pattern in MINIMUM_EXPERIENCE_PATTERNS:
        values.extend(float(match.group("years")) for match in pattern.finditer(text))
    return round(max(values), 2) if values else 0.0


def _job_id(source_file: str, cleaned_text: str) -> str:
    payload = f"{Path(source_file).name}\0{cleaned_text}".encode("utf-8")
    return f"job_{hashlib.sha256(payload).hexdigest()[:12]}"


def build_job_requirements_from_text(
    text: str,
    *,
    source_file: str = "in_memory_job.txt",
    parser_warnings: list[str] | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> JobRequirements:
    """Build job requirements while retaining how each skill was classified."""

    cleaned = clean_text(text)
    sections = split_sections(cleaned)
    extractor = SkillExtractor(taxonomy_path)

    required_evidence = "\n".join(
        part
        for part in (sections.get("requirements", ""), _cue_text(cleaned, REQUIRED_CUES))
        if part
    )
    preferred_evidence = "\n".join(
        part
        for part in (sections.get("preferred", ""), _cue_text(cleaned, PREFERRED_CUES))
        if part
    )

    mandatory_certifications = set(extract_certifications(required_evidence))
    preferred_certifications = (
        set(extract_certifications(preferred_evidence)) - mandatory_certifications
    )
    mandatory_skills = set(extractor.extract(required_evidence))
    preferred_skills = set(extractor.extract(preferred_evidence)) - mandatory_skills

    # A certification such as AWS Cloud Practitioner is scored as a certification,
    # not a second time as a cloud-platform keyword from the same phrase.
    for certification in mandatory_certifications:
        mapped_skill = CERTIFICATION_SKILL_MAP.get(certification)
        if mapped_skill:
            mandatory_skills.discard(mapped_skill)
    for certification in preferred_certifications:
        mapped_skill = CERTIFICATION_SKILL_MAP.get(certification)
        if mapped_skill:
            preferred_skills.discard(mapped_skill)

    all_skills = set(extractor.extract(cleaned))
    relevant_skills = all_skills - mandatory_skills - preferred_skills
    for certification in mandatory_certifications | preferred_certifications:
        mapped_skill = CERTIFICATION_SKILL_MAP.get(certification)
        if mapped_skill:
            relevant_skills.discard(mapped_skill)

    required_degrees, required_fields = extract_education(required_evidence)

    experience_evidence = required_evidence or cleaned
    minimum_experience = extract_minimum_experience(experience_evidence)

    warnings = list(parser_warnings or [])
    title = extract_job_title(cleaned)
    if len(cleaned) < 80:
        warnings.append("The job description contains too little text for reliable extraction.")
    if title == "Untitled Role":
        warnings.append("A job title could not be confidently identified.")
    if not all_skills:
        warnings.append("No skills matched the current taxonomy.")
    if relevant_skills:
        warnings.append(
            "Some skills lacked mandatory/preferred wording and were classified as relevant."
        )
    if not mandatory_skills and not required_degrees and minimum_experience == 0:
        warnings.append("No explicit mandatory qualification was identified.")

    return JobRequirements(
        job_id=_job_id(source_file, cleaned),
        title=title,
        description=cleaned,
        source_file=Path(source_file).name,
        mandatory_skills=sorted(mandatory_skills),
        preferred_skills=sorted(preferred_skills),
        relevant_skills=sorted(relevant_skills),
        minimum_experience_years=minimum_experience,
        required_degrees=sorted(required_degrees),
        required_fields_of_study=sorted(required_fields),
        mandatory_certifications=sorted(mandatory_certifications),
        preferred_certifications=sorted(preferred_certifications),
        detected_sections=sorted(sections),
        warnings=list(dict.fromkeys(warnings)),
    )


def build_job_requirements(
    path: str | Path,
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> JobRequirements:
    document = extract_document(path)
    text_with_links = "\n".join((document.raw_text, *document.links))
    return build_job_requirements_from_text(
        text_with_links,
        source_file=document.source_name,
        parser_warnings=document.warnings,
        taxonomy_path=taxonomy_path,
    )
