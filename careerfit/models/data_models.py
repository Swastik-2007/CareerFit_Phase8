"""Typed data contracts for the CareerFit pipeline.

The contracts deliberately keep identity information separate from features that
are allowed to influence a hiring score. This separation will be reused by the
fairness-audit phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ParsedDocument:
    """Text and metadata returned by a document-specific parser."""

    source_path: str
    document_type: str
    raw_text: str
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    @property
    def source_name(self) -> str:
        return Path(self.source_path).name


@dataclass(slots=True)
class IdentityInfo:
    """Personally identifying information, isolated from scoring features."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None


@dataclass(slots=True)
class CandidateProfile:
    """Normalized representation of one candidate resume."""

    candidate_id: str
    source_file: str
    identity: IdentityInfo
    skills: list[str] = field(default_factory=list)
    skill_categories: dict[str, list[str]] = field(default_factory=dict)
    experience_years: float = 0.0
    degrees: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    detected_sections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_text: str = ""

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("cleaned_text", None)
        return data


@dataclass(slots=True)
class JobRequirements:
    """Normalized and explainable representation of a job description."""

    job_id: str
    title: str
    description: str
    source_file: str = ""
    mandatory_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    relevant_skills: list[str] = field(default_factory=list)
    minimum_experience_years: float = 0.0
    required_degrees: list[str] = field(default_factory=list)
    required_fields_of_study: list[str] = field(default_factory=list)
    mandatory_certifications: list[str] = field(default_factory=list)
    preferred_certifications: list[str] = field(default_factory=list)
    detected_sections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("description", None)
        return data


@dataclass(slots=True)
class BaselineMatchResult:
    """Phase 2 rule-based candidate-to-job comparison.

    Scores use the inclusive 0–1 scale. ``eligible`` is evaluated independently
    from the weighted score so a strong candidate cannot hide a mandatory failure.
    """

    candidate_id: str
    source_file: str
    job_id: str
    rank: int | None
    eligible: bool
    requirement_score: float
    candidate_name: str | None = None
    mandatory_skill_coverage: float | None = None
    additional_skill_coverage: float | None = None
    experience_coverage: float | None = None
    education_coverage: float | None = None
    certification_coverage: float | None = None
    matched_mandatory_skills: list[str] = field(default_factory=list)
    missing_mandatory_skills: list[str] = field(default_factory=list)
    matched_additional_skills: list[str] = field(default_factory=list)
    missing_additional_skills: list[str] = field(default_factory=list)
    matched_certifications: list[str] = field(default_factory=list)
    missing_mandatory_certifications: list[str] = field(default_factory=list)
    failed_requirements: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        score_fields = (
            "requirement_score",
            "mandatory_skill_coverage",
            "additional_skill_coverage",
            "experience_coverage",
            "education_coverage",
            "certification_coverage",
        )
        for field_name in score_fields:
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoreResult:
    """Explainable scoring result used by the later ranking phase.

    Every score is represented on the inclusive 0–1 scale. Presentation layers
    may multiply values by 100 to display percentages.
    """

    candidate_id: str
    job_id: str
    text_similarity: float
    role_probability: float
    requirement_score: float
    fit_score: float
    eligible: bool
    rank: int | None = None
    source_file: str = ""
    candidate_name: str | None = None
    target_role: str = ""
    predicted_role: str = ""
    model_name: str = ""
    component_weights: dict[str, float] = field(default_factory=dict)
    component_contributions: dict[str, float] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    failed_requirements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        score_fields = (
            "text_similarity",
            "role_probability",
            "requirement_score",
            "fit_score",
        )
        for field_name in score_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
