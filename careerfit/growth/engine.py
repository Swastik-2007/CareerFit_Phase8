"""Generate conservative, graph-backed skill-growth simulations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from careerfit.matching import compare_candidate
from careerfit.models import CandidateProfile, JobRequirements, ScoreResult
from careerfit.scoring import HybridScoringContext, score_candidates_with_context

from .graph import SkillGraph
from .models import GrowthPlanResult, SkillRecommendation


PRIORITY_ORDER = {"mandatory": 0, "preferred": 1, "relevant": 2}


def _priority(skill: str, job: JobRequirements) -> str:
    if skill in job.mandatory_skills:
        return "mandatory"
    if skill in job.preferred_skills:
        return "preferred"
    return "relevant"


def _potential_fit(
    current: ScoreResult,
    requirement_score_after: float,
    context: HybridScoringContext,
) -> float:
    weights = context.weights
    return round(
        weights["text_similarity"] * current.text_similarity
        + weights["role_probability"] * current.role_probability
        + weights["requirement_score"] * requirement_score_after,
        6,
    )


def _candidate_with_skills(
    candidate: CandidateProfile,
    additional_skills: set[str],
) -> CandidateProfile:
    return replace(candidate, skills=sorted(set(candidate.skills) | additional_skills))


def analyze_growth(
    candidates: list[CandidateProfile],
    job: JobRequirements,
    bundle: dict[str, Any],
    context: HybridScoringContext,
    skill_graph: SkillGraph,
    *,
    max_recommendations: int = 5,
) -> list[GrowthPlanResult]:
    """Recommend missing job skills and calculate conservative score changes."""

    if max_recommendations <= 0:
        raise ValueError("max_recommendations must be greater than zero.")

    current_scores = score_candidates_with_context(candidates, job, bundle, context)
    score_by_candidate = {result.candidate_id: result for result in current_scores}
    job_skills = (
        set(job.mandatory_skills)
        | set(job.preferred_skills)
        | set(job.relevant_skills)
    )
    results: list[GrowthPlanResult] = []

    for candidate in candidates:
        current = score_by_candidate[candidate.candidate_id]
        missing = sorted(job_skills - set(candidate.skills))
        recommendations: list[SkillRecommendation] = []
        unmapped: list[str] = []

        for target_skill in missing:
            plan = skill_graph.learning_plan(target_skill, candidate.skills)
            if plan is None:
                unmapped.append(target_skill)
                continue
            simulated_candidate = _candidate_with_skills(
                candidate,
                set(plan.learning_path),
            )
            after = compare_candidate(simulated_candidate, job)
            potential_fit = _potential_fit(current, after.requirement_score, context)
            recommendations.append(
                SkillRecommendation(
                    target_skill=target_skill,
                    priority=_priority(target_skill, job),
                    learning_path=plan.learning_path,
                    prerequisites_to_learn=plan.prerequisites_to_learn,
                    learning_steps=plan.learning_steps,
                    estimated_hours=plan.estimated_hours,
                    difficulty=plan.difficulty,
                    rationale=plan.rationale,
                    requirement_score_after=after.requirement_score,
                    potential_fit_score=potential_fit,
                    score_gain=round(max(0.0, potential_fit - current.fit_score), 6),
                    eligible_after_learning=after.eligible,
                )
            )

        recommendations.sort(
            key=lambda item: (
                PRIORITY_ORDER[item.priority],
                -item.score_gain,
                item.estimated_hours,
                item.target_skill,
            )
        )
        selected = recommendations[:max_recommendations]
        combined_skill_set = {
            skill for recommendation in selected for skill in recommendation.learning_path
        }
        combined_path = skill_graph.order_skills(combined_skill_set)
        combined_candidate = _candidate_with_skills(candidate, set(combined_path))
        combined_baseline = compare_candidate(combined_candidate, job)
        combined_fit = _potential_fit(
            current,
            combined_baseline.requirement_score,
            context,
        )
        combined_hours = sum(
            int(skill_graph.graph.nodes[skill]["estimated_hours"])
            for skill in combined_path
        )
        limitations = [
            "The simulation changes only explicit skill coverage; text similarity and role probability remain fixed.",
            "Learning-hour estimates are planning heuristics, not guaranteed completion times.",
            "The starter prerequisite graph requires subject-matter review before high-stakes use.",
        ]
        if unmapped:
            limitations.append(
                "Some missing skills are not mapped in the current prerequisite graph."
            )

        results.append(
            GrowthPlanResult(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.identity.name,
                source_file=candidate.source_file,
                job_id=job.job_id,
                current_rank=current.rank,
                current_fit_score=current.fit_score,
                current_requirement_score=current.requirement_score,
                current_eligible=current.eligible,
                target_role=context.target_role,
                recommendations=selected,
                combined_learning_path=combined_path,
                combined_estimated_hours=combined_hours,
                combined_requirement_score=combined_baseline.requirement_score,
                combined_potential_fit_score=combined_fit,
                combined_score_gain=round(max(0.0, combined_fit - current.fit_score), 6),
                eligible_after_plan=combined_baseline.eligible,
                unmapped_missing_skills=unmapped,
                limitations=limitations,
            )
        )

    results.sort(
        key=lambda item: (
            item.current_rank if item.current_rank is not None else 10**9,
            item.source_file.casefold(),
        )
    )
    return results
