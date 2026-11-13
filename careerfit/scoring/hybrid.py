"""Combine text similarity, role confidence, and explicit requirements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from careerfit.matching import compare_candidate
from careerfit.ml.features import candidate_model_text
from careerfit.ml.inference import predict_text
from careerfit.ml.similarity import text_similarity
from careerfit.models import CandidateProfile, JobRequirements, ScoreResult


HYBRID_WEIGHTS = {
    "text_similarity": 0.40,
    "role_probability": 0.40,
    "requirement_score": 0.20,
}


@dataclass(slots=True)
class HybridScoringContext:
    """Run-level facts shared by every candidate score."""

    target_role: str
    target_role_source: str
    job_role_confidence: float | None
    weights: dict[str, float]
    probability_method: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Validate a complete weight set and normalize it to a one-unit total."""

    supplied = dict(HYBRID_WEIGHTS if weights is None else weights)
    required = set(HYBRID_WEIGHTS)
    missing = required - supplied.keys()
    unexpected = supplied.keys() - required
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected: " + ", ".join(sorted(unexpected)))
        raise ValueError("Invalid composite-score weights (" + "; ".join(details) + ").")

    numeric: dict[str, float] = {}
    for name in HYBRID_WEIGHTS:
        try:
            value = float(supplied[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight '{name}' must be numeric.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Weight '{name}' must be finite and non-negative.")
        numeric[name] = value
    total = sum(numeric.values())
    if total <= 0:
        raise ValueError("At least one composite-score weight must be greater than zero.")
    return {name: round(value / total, 12) for name, value in numeric.items()}


def build_scoring_context(
    job: JobRequirements,
    bundle: dict[str, Any],
    *,
    target_role_override: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> HybridScoringContext:
    """Resolve the job role explicitly and record uncertainty."""

    normalized_weights = normalize_weights(weights)
    probability_method = str(bundle.get("probability_method", "legacy_or_unknown"))
    classes = [str(label) for label in bundle["classes"]]
    warnings: list[str] = []

    if target_role_override:
        role_lookup = {label.casefold(): label for label in classes}
        target_role = role_lookup.get(target_role_override.strip().casefold())
        if target_role is None:
            raise ValueError(
                f"Unknown target role '{target_role_override}'. Available roles: "
                + ", ".join(classes)
            )
        return HybridScoringContext(
            target_role=target_role,
            target_role_source="user_override",
            job_role_confidence=None,
            weights=normalized_weights,
            probability_method=probability_method,
            warnings=warnings,
        )

    prediction = predict_text(bundle, f"{job.title}\n{job.description}")
    confidence = float(prediction["confidence"])
    if confidence < 0.50:
        warnings.append(
            "The inferred job role has confidence below 0.50; review it or use --target-role."
        )
    return HybridScoringContext(
        target_role=prediction["predicted_role"],
        target_role_source="model_inference",
        job_role_confidence=round(confidence, 6),
        weights=normalized_weights,
        probability_method=prediction["probability_method"],
        warnings=warnings,
    )


def score_candidates_with_context(
    candidates: list[CandidateProfile],
    job: JobRequirements,
    bundle: dict[str, Any],
    context: HybridScoringContext,
) -> list[ScoreResult]:
    """Score candidates and expose the exact arithmetic for every result."""

    results: list[ScoreResult] = []
    weights = context.weights

    for candidate in candidates:
        baseline = compare_candidate(candidate, job)
        feature_text = candidate_model_text(candidate)
        prediction = predict_text(bundle, feature_text)
        similarity = text_similarity(feature_text, job.description)
        role_probability = float(
            prediction["role_probabilities"].get(context.target_role, 0.0)
        )
        contributions = {
            "text_similarity": round(weights["text_similarity"] * similarity, 6),
            "role_probability": round(weights["role_probability"] * role_probability, 6),
            "requirement_score": round(
                weights["requirement_score"] * baseline.requirement_score, 6
            ),
        }
        fit_score = round(sum(contributions.values()), 6)
        explanation = [
            f"Text similarity: {similarity:.6f} × {weights['text_similarity']:.3f} "
            f"= {contributions['text_similarity']:.6f}.",
            f"Probability for '{context.target_role}': {role_probability:.6f} × "
            f"{weights['role_probability']:.3f} = {contributions['role_probability']:.6f}.",
            f"Requirement coverage: {baseline.requirement_score:.6f} × "
            f"{weights['requirement_score']:.3f} = "
            f"{contributions['requirement_score']:.6f}.",
            f"Final fit score: {fit_score:.6f}.",
            "Mandatory eligibility is inherited from Phase 2 and is never overridden by fit score.",
        ]
        results.append(
            ScoreResult(
                candidate_id=candidate.candidate_id,
                job_id=job.job_id,
                text_similarity=similarity,
                role_probability=round(role_probability, 6),
                requirement_score=baseline.requirement_score,
                fit_score=fit_score,
                eligible=baseline.eligible,
                source_file=candidate.source_file,
                candidate_name=candidate.identity.name,
                target_role=context.target_role,
                predicted_role=prediction["predicted_role"],
                model_name=prediction["model_name"],
                component_weights=weights.copy(),
                component_contributions=contributions,
                matched_skills=sorted(
                    set(baseline.matched_mandatory_skills)
                    | set(baseline.matched_additional_skills)
                ),
                missing_skills=sorted(
                    set(baseline.missing_mandatory_skills)
                    | set(baseline.missing_additional_skills)
                ),
                failed_requirements=baseline.failed_requirements,
                warnings=context.warnings.copy(),
                explanation=explanation,
            )
        )

    results.sort(
        key=lambda result: (
            -int(result.eligible),
            -result.fit_score,
            result.source_file.casefold(),
        )
    )
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results


def score_candidates_with_model(
    candidates: list[CandidateProfile],
    job: JobRequirements,
    bundle: dict[str, Any],
    *,
    target_role_override: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, list[ScoreResult]]:
    """Backward-compatible wrapper returning the target role and ranking."""

    context = build_scoring_context(
        job,
        bundle,
        target_role_override=target_role_override,
        weights=weights,
    )
    return context.target_role, score_candidates_with_context(candidates, job, bundle, context)
