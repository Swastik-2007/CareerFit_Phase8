"""Transparent baseline matching without a trained ML model."""

from __future__ import annotations

from careerfit.models import BaselineMatchResult, CandidateProfile, JobRequirements


COMPONENT_WEIGHTS = {
    "mandatory_skills": 0.45,
    "additional_skills": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "certifications": 0.10,
}


def _coverage(available: set[str], expected: set[str]) -> float | None:
    if not expected:
        return None
    return round(len(available & expected) / len(expected), 4)


def _degree_requirement_met(candidate_degrees: set[str], required_degree: str) -> bool:
    if required_degree in candidate_degrees:
        return True
    if required_degree == "bachelor_degree":
        return any(
            degree.startswith(("bachelor_", "master_", "doctor_"))
            for degree in candidate_degrees
        )
    if required_degree == "master_degree":
        return any(degree.startswith(("master_", "doctor_")) for degree in candidate_degrees)
    return False


def _weighted_score(components: dict[str, float | None]) -> float:
    active = {name: value for name, value in components.items() if value is not None}
    if not active:
        return 0.0
    total_weight = sum(COMPONENT_WEIGHTS[name] for name in active)
    weighted = sum(COMPONENT_WEIGHTS[name] * value for name, value in active.items())
    return round(weighted / total_weight, 4)


def compare_candidate(
    candidate: CandidateProfile,
    job: JobRequirements,
) -> BaselineMatchResult:
    """Compare only job-relevant profile fields against explicit requirements."""

    candidate_skills = set(candidate.skills)
    mandatory = set(job.mandatory_skills)
    additional = set(job.preferred_skills) | set(job.relevant_skills)
    candidate_certifications = set(candidate.certifications)
    all_certifications = set(job.mandatory_certifications) | set(job.preferred_certifications)

    matched_mandatory = sorted(candidate_skills & mandatory)
    missing_mandatory = sorted(mandatory - candidate_skills)
    matched_additional = sorted(candidate_skills & additional)
    missing_additional = sorted(additional - candidate_skills)
    matched_certifications = sorted(candidate_certifications & all_certifications)
    missing_mandatory_certifications = sorted(
        set(job.mandatory_certifications) - candidate_certifications
    )

    mandatory_coverage = _coverage(candidate_skills, mandatory)
    additional_coverage = _coverage(candidate_skills, additional)
    certification_coverage = _coverage(candidate_certifications, all_certifications)

    if job.minimum_experience_years > 0:
        experience_coverage = min(
            candidate.experience_years / job.minimum_experience_years,
            1.0,
        )
    else:
        experience_coverage = None

    required_degrees = set(job.required_degrees)
    required_fields = set(job.required_fields_of_study)
    if required_degrees or required_fields:
        education_checks: list[bool] = []
        if required_degrees:
            education_checks.append(
                any(
                    _degree_requirement_met(set(candidate.degrees), degree)
                    for degree in required_degrees
                )
            )
        if required_fields:
            education_checks.append(bool(required_fields & set(candidate.fields_of_study)))
        education_coverage = sum(education_checks) / len(education_checks)
    else:
        education_coverage = None

    components = {
        "mandatory_skills": mandatory_coverage,
        "additional_skills": additional_coverage,
        "experience": experience_coverage,
        "education": education_coverage,
        "certifications": certification_coverage,
    }
    requirement_score = _weighted_score(components)

    failed: list[str] = []
    if missing_mandatory:
        failed.append(f"Missing mandatory skills: {', '.join(missing_mandatory)}")
    if (
        job.minimum_experience_years > 0
        and candidate.experience_years < job.minimum_experience_years
    ):
        minimum_label = "year" if job.minimum_experience_years == 1 else "years"
        failed.append(
            "Experience below minimum: "
            f"{candidate.experience_years:g} < {job.minimum_experience_years:g} {minimum_label}"
        )
    if required_degrees and education_coverage == 0.0:
        failed.append(f"Required degree not found: one of {', '.join(sorted(required_degrees))}")
    elif required_degrees and not any(
        _degree_requirement_met(set(candidate.degrees), degree) for degree in required_degrees
    ):
        failed.append(f"Required degree not found: one of {', '.join(sorted(required_degrees))}")
    if required_fields and not required_fields & set(candidate.fields_of_study):
        failed.append(
            f"Required field of study not found: one of {', '.join(sorted(required_fields))}"
        )
    if missing_mandatory_certifications:
        failed.append(
            "Missing mandatory certifications: "
            f"{', '.join(missing_mandatory_certifications)}"
        )

    explanation: list[str] = []
    if mandatory_coverage is not None:
        explanation.append(
            f"Mandatory skills: {len(matched_mandatory)}/{len(mandatory)} matched "
            f"({mandatory_coverage * 100:.1f}%)."
        )
    else:
        explanation.append("No mandatory skills were explicitly identified.")
    if additional_coverage is not None:
        explanation.append(
            f"Preferred/relevant skills: {len(matched_additional)}/{len(additional)} matched "
            f"({additional_coverage * 100:.1f}%)."
        )
    if experience_coverage is not None:
        candidate_year_label = "year" if candidate.experience_years == 1 else "years"
        minimum_year_label = "year" if job.minimum_experience_years == 1 else "years"
        explanation.append(
            f"Experience: {candidate.experience_years:g} {candidate_year_label} against a "
            f"{job.minimum_experience_years:g}-{minimum_year_label} minimum."
        )
    if education_coverage is not None:
        explanation.append(
            "Education requirement matched."
            if education_coverage == 1.0
            else "Education requirement did not match."
        )
    if certification_coverage is not None:
        explanation.append(
            f"Certifications: {len(matched_certifications)}/{len(all_certifications)} matched."
        )
    explanation.append(
        "Eligibility passed: no mandatory requirement failed."
        if not failed
        else "Eligibility failed independently of the weighted match score."
    )
    explanation.append(
        f"Weighted requirement score: {requirement_score * 100:.1f}% using only active components."
    )

    return BaselineMatchResult(
        candidate_id=candidate.candidate_id,
        source_file=candidate.source_file,
        job_id=job.job_id,
        rank=None,
        eligible=not failed,
        requirement_score=requirement_score,
        candidate_name=candidate.identity.name,
        mandatory_skill_coverage=mandatory_coverage,
        additional_skill_coverage=additional_coverage,
        experience_coverage=experience_coverage,
        education_coverage=education_coverage,
        certification_coverage=certification_coverage,
        matched_mandatory_skills=matched_mandatory,
        missing_mandatory_skills=missing_mandatory,
        matched_additional_skills=matched_additional,
        missing_additional_skills=missing_additional,
        matched_certifications=matched_certifications,
        missing_mandatory_certifications=missing_mandatory_certifications,
        failed_requirements=failed,
        explanation=explanation,
    )


def rank_candidates(
    candidates: list[CandidateProfile],
    job: JobRequirements,
) -> list[BaselineMatchResult]:
    """Rank eligible candidates first, then use requirement score as the tiebreaker."""

    results = [compare_candidate(candidate, job) for candidate in candidates]
    results.sort(
        key=lambda result: (
            -int(result.eligible),
            -result.requirement_score,
            result.source_file.casefold(),
        )
    )
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results
