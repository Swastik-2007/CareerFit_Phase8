"""Serializable Phase 5 growth-planning results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillRecommendation:
    target_skill: str
    priority: str
    learning_path: list[str]
    prerequisites_to_learn: list[str]
    learning_steps: int
    estimated_hours: int
    difficulty: str
    rationale: str
    requirement_score_after: float
    potential_fit_score: float
    score_gain: float
    eligible_after_learning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GrowthPlanResult:
    candidate_id: str
    candidate_name: str | None
    source_file: str
    job_id: str
    current_rank: int | None
    current_fit_score: float
    current_requirement_score: float
    current_eligible: bool
    target_role: str
    recommendations: list[SkillRecommendation] = field(default_factory=list)
    combined_learning_path: list[str] = field(default_factory=list)
    combined_estimated_hours: int = 0
    combined_requirement_score: float = 0.0
    combined_potential_fit_score: float = 0.0
    combined_score_gain: float = 0.0
    eligible_after_plan: bool = False
    unmapped_missing_skills: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recommendations"] = [item.to_dict() for item in self.recommendations]
        return data
