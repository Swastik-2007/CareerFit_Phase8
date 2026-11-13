"""Build a structured candidate profile from extracted resume text."""

from __future__ import annotations

import hashlib
from pathlib import Path

from careerfit.models import CandidateProfile
from careerfit.parsing import clean_text, extract_document, split_sections

from .education import extract_education
from .certifications import extract_certifications
from .experience import extract_experience_years
from .identity import extract_identity
from .skills import DEFAULT_TAXONOMY_PATH, SkillExtractor


def _candidate_id(source_file: str, cleaned_text: str) -> str:
    digest = hashlib.sha256(f"{Path(source_file).name}\0{cleaned_text}".encode("utf-8")).hexdigest()
    return f"candidate_{digest[:12]}"


def build_candidate_profile_from_text(
    text: str,
    *,
    source_file: str = "in_memory.txt",
    parser_warnings: list[str] | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> CandidateProfile:
    """Convert unstructured resume text into the Phase 1 data contract."""

    cleaned = clean_text(text)
    sections = split_sections(cleaned)
    skill_extractor = SkillExtractor(taxonomy_path)

    # Restrict dated interval analysis to work experience when that section exists.
    experience_text = sections.get("experience", "")
    scoped_experience_text = "\n".join(
        part for part in (sections.get("summary", ""), experience_text) if part
    )
    if experience_text:
        experience_years = extract_experience_years(scoped_experience_text)
    else:
        # Without a recognized experience section, only accept explicit statements
        # such as "3 years of experience". This prevents education dates from being
        # mistaken for employment duration.
        experience_years = extract_experience_years(cleaned, allow_date_ranges=False)

    education_text = sections.get("education", cleaned)
    degrees, fields = extract_education(education_text)
    skill_categories = skill_extractor.extract_by_category(cleaned)
    skills = sorted({skill for values in skill_categories.values() for skill in values})
    certifications = extract_certifications(cleaned)

    warnings = list(parser_warnings or [])
    if len(cleaned) < 80:
        warnings.append("The resume contains too little usable text for reliable extraction.")
    if not skills:
        warnings.append("No skills matched the current taxonomy; review or extend skills.json.")
    if not degrees:
        warnings.append("No supported degree was confidently detected.")
    if not experience_text and experience_years == 0:
        warnings.append("No work-experience section or explicit experience duration was detected.")

    return CandidateProfile(
        candidate_id=_candidate_id(source_file, cleaned),
        source_file=Path(source_file).name,
        identity=extract_identity(cleaned),
        skills=skills,
        skill_categories=skill_categories,
        experience_years=experience_years,
        degrees=degrees,
        fields_of_study=fields,
        certifications=certifications,
        detected_sections=sorted(sections),
        warnings=list(dict.fromkeys(warnings)),
        cleaned_text=cleaned,
    )


def build_candidate_profile(
    path: str | Path,
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> CandidateProfile:
    document = extract_document(path)
    text_with_links = "\n".join((document.raw_text, *document.links))
    return build_candidate_profile_from_text(
        text_with_links,
        source_file=document.source_name,
        parser_warnings=document.warnings,
        taxonomy_path=taxonomy_path,
    )
