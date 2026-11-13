"""Paired original-versus-masked score sensitivity analysis."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

from careerfit.models import CandidateProfile, JobRequirements, ScoreResult
from careerfit.scoring import HybridScoringContext, score_candidates_with_context

from .masking import mask_candidate
from .models import CandidateFairnessAudit, FairnessAuditReport


MIN_ROBUST_COHORT_SIZE = 5


def _bounded_number(value: float, label: str, *, allow_zero: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    lower_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not lower_ok or number > 1:
        qualifier = "between 0 and 1" if allow_zero else "greater than 0 and at most 1"
        raise ValueError(f"{label} must be {qualifier}.")
    return number


def select_review_threshold(
    absolute_score_changes: Iterable[float],
    *,
    minimum_threshold: float = 0.01,
    threshold_override: float | None = None,
    mad_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Select a transparent review threshold from observed paired differences."""

    values = [abs(float(value)) for value in absolute_score_changes]
    if not values:
        raise ValueError("At least one score change is required for threshold selection.")
    if not all(math.isfinite(value) and value <= 1 for value in values):
        raise ValueError("Score changes must be finite values between 0 and 1.")
    minimum = _bounded_number(minimum_threshold, "minimum_threshold")
    if not math.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be finite and greater than zero.")

    center = median(values)
    mad = median(abs(value - center) for value in values)
    robust_sigma = 1.4826 * mad
    estimated_upper = min(1.0, center + mad_multiplier * robust_sigma)
    reliable = len(values) >= MIN_ROBUST_COHORT_SIZE

    if threshold_override is not None:
        applied = _bounded_number(threshold_override, "threshold_override")
        method = "user_override"
    elif reliable:
        applied = max(minimum, estimated_upper)
        method = "robust_median_plus_mad"
    else:
        applied = minimum
        method = "small_sample_minimum"

    return {
        "method": method,
        "candidate_count": len(values),
        "minimum_reliable_cohort_size": MIN_ROBUST_COHORT_SIZE,
        "reliable_cohort_estimate": reliable,
        "median_absolute_score_change": round(center, 6),
        "median_absolute_deviation": round(mad, 6),
        "robust_sigma_estimate": round(robust_sigma, 6),
        "estimated_upper_normal_change": round(estimated_upper, 6),
        "minimum_review_threshold": round(minimum, 6),
        "applied_review_threshold": round(applied, 6),
        "mad_multiplier": float(mad_multiplier),
        "warning": (
            None
            if reliable
            else "Fewer than five candidates were audited; the minimum threshold is used instead of treating this cohort as a reliable normal-variation baseline."
        ),
    }


def _result_map(results: list[ScoreResult]) -> dict[str, ScoreResult]:
    return {result.candidate_id: result for result in results}


def audit_fairness(
    candidates: list[CandidateProfile],
    job: JobRequirements,
    bundle: dict[str, Any],
    context: HybridScoringContext,
    *,
    minimum_threshold: float = 0.01,
    threshold_override: float | None = None,
    mad_multiplier: float = 3.0,
) -> FairnessAuditReport:
    """Compare standard and masked scores without changing hiring decisions."""

    if not candidates:
        raise ValueError("At least one candidate is required for a fairness audit.")

    masked = [mask_candidate(candidate) for candidate in candidates]
    original_scores = score_candidates_with_context(candidates, job, bundle, context)
    masked_scores = score_candidates_with_context(
        [item.profile for item in masked], job, bundle, context
    )
    masked_by_id = _result_map(masked_scores)
    masking_by_id = {
        item.profile.candidate_id: item.summary
        for item in masked
    }
    candidate_by_id = {item.candidate_id: item for item in candidates}

    absolute_changes = [
        abs(masked_by_id[item.candidate_id].fit_score - item.fit_score)
        for item in original_scores
    ]
    threshold_analysis = select_review_threshold(
        absolute_changes,
        minimum_threshold=minimum_threshold,
        threshold_override=threshold_override,
        mad_multiplier=mad_multiplier,
    )
    threshold = float(threshold_analysis["applied_review_threshold"])

    audits: list[CandidateFairnessAudit] = []
    for original in original_scores:
        diagnostic = masked_by_id[original.candidate_id]
        candidate = candidate_by_id[original.candidate_id]
        summary = masking_by_id[original.candidate_id]
        signed_change = round(diagnostic.fit_score - original.fit_score, 6)
        absolute_change = round(abs(signed_change), 6)
        rank_changed = diagnostic.rank != original.rank
        eligibility_changed = diagnostic.eligible != original.eligible
        score_sensitive = absolute_change >= threshold and absolute_change > 0
        human_review = score_sensitive or rank_changed or eligibility_changed

        component_changes = {
            name: round(
                getattr(diagnostic, name) - getattr(original, name),
                6,
            )
            for name in (
                "text_similarity",
                "role_probability",
                "requirement_score",
                "fit_score",
            )
        }
        if component_changes["requirement_score"] != 0 or eligibility_changed:
            raise RuntimeError(
                "Masking changed structured requirements or eligibility; the audit cannot proceed safely."
            )

        explanation = [
            f"Standard score {original.fit_score:.6f}; masked diagnostic score {diagnostic.fit_score:.6f}.",
            f"Absolute change {absolute_change:.6f}; review threshold {threshold:.6f}.",
            "The masked result is diagnostic only and does not replace the original ranking or eligibility decision.",
        ]
        if rank_changed:
            explanation.append("The diagnostic masked ranking changed, so human review is required.")
        if summary.institution_acronym_warning:
            explanation.append(
                "A possible institution acronym was retained because automatic acronym removal is ambiguous."
            )

        audits.append(
            CandidateFairnessAudit(
                candidate_id=original.candidate_id,
                candidate_name=original.candidate_name,
                source_file=original.source_file,
                original_rank=original.rank,
                masked_diagnostic_rank=diagnostic.rank,
                rank_changed=rank_changed,
                original_fit_score=original.fit_score,
                masked_fit_score=diagnostic.fit_score,
                signed_score_change=signed_change,
                absolute_score_change=absolute_change,
                review_threshold=threshold,
                score_sensitive=score_sensitive,
                human_review_required=human_review,
                audit_disposition=(
                    "human_review" if human_review else "no_additional_review_flag"
                ),
                automatic_decision_changed=False,
                original_eligible=original.eligible,
                masked_eligible=diagnostic.eligible,
                eligibility_changed=eligibility_changed,
                component_changes=component_changes,
                masking_summary=summary,
                preserved_qualifications={
                    "skills": candidate.skills,
                    "experience_years": candidate.experience_years,
                    "degrees": candidate.degrees,
                    "fields_of_study": candidate.fields_of_study,
                    "certifications": candidate.certifications,
                },
                explanation=explanation,
            )
        )

    audits.sort(key=lambda item: item.original_rank or 10**9)
    limitations = [
        "This paired sensitivity audit can flag score changes; it cannot prove that a model or hiring process is free from discrimination.",
        "No protected demographic attributes are inferred, and group fairness metrics require a lawful, representative, consented evaluation dataset.",
        "Institution-name detection is heuristic; ambiguous acronyms are retained and reported for human review.",
        "A masked score is diagnostic only and never triggers automatic rejection or changes mandatory eligibility.",
        "Small cohorts cannot establish a stable normal-variation baseline; review the threshold warning before operational use.",
    ]
    methodology = {
        "comparison": "paired_standard_vs_masked_scoring",
        "masked_categories": [
            "candidate_name",
            "contact_details",
            "gendered_pronouns_and_honorifics",
            "recognizable_institution_names",
        ],
        "preserved_categories": [
            "skills",
            "experience_years",
            "degrees",
            "fields_of_study",
            "certifications",
        ],
        "decision_policy": "flag_for_human_review_only",
        "protected_attribute_inference": False,
    }
    return FairnessAuditReport(
        threshold_analysis=threshold_analysis,
        results=audits,
        limitations=limitations,
        methodology=methodology,
    )
