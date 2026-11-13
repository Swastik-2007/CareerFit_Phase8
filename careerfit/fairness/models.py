"""Serializable contracts for Phase 6 fairness sensitivity audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MaskingSummary:
    """Counts of removed text without exposing the removed values."""

    name_occurrences: int = 0
    contact_occurrences: int = 0
    pronoun_occurrences: int = 0
    institution_occurrences: int = 0
    institution_acronym_warning: bool = False

    @property
    def total_occurrences(self) -> int:
        return (
            self.name_occurrences
            + self.contact_occurrences
            + self.pronoun_occurrences
            + self.institution_occurrences
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_occurrences"] = self.total_occurrences
        return data


@dataclass(slots=True)
class CandidateFairnessAudit:
    candidate_id: str
    candidate_name: str | None
    source_file: str
    original_rank: int | None
    masked_diagnostic_rank: int | None
    rank_changed: bool
    original_fit_score: float
    masked_fit_score: float
    signed_score_change: float
    absolute_score_change: float
    review_threshold: float
    score_sensitive: bool
    human_review_required: bool
    audit_disposition: str
    automatic_decision_changed: bool
    original_eligible: bool
    masked_eligible: bool
    eligibility_changed: bool
    component_changes: dict[str, float]
    masking_summary: MaskingSummary
    preserved_qualifications: dict[str, Any]
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["masking_summary"] = self.masking_summary.to_dict()
        return data


@dataclass(slots=True)
class FairnessAuditReport:
    threshold_analysis: dict[str, Any]
    results: list[CandidateFairnessAudit]
    limitations: list[str]
    methodology: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_analysis": self.threshold_analysis,
            "flagged_for_human_review_count": sum(
                item.human_review_required for item in self.results
            ),
            "original_ranking_preserved": True,
            "audit_results": [item.to_dict() for item in self.results],
            "methodology": self.methodology,
            "limitations": self.limitations,
        }
