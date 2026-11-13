import unittest

from careerfit.matching import compare_candidate, rank_candidates
from careerfit.models import CandidateProfile, IdentityInfo, JobRequirements


def candidate(
    candidate_id: str,
    source_file: str,
    skills: list[str],
    experience: float,
    degrees: list[str],
) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=source_file,
        identity=IdentityInfo(name=candidate_id),
        skills=skills,
        experience_years=experience,
        degrees=degrees,
    )


JOB = JobRequirements(
    job_id="job_1",
    title="Data Analyst",
    description="",
    mandatory_skills=["python", "sql"],
    preferred_skills=["pandas", "power_bi"],
    relevant_skills=["machine_learning"],
    minimum_experience_years=1.0,
    required_degrees=["bachelor_of_technology"],
)


class BaselineMatchingTests(unittest.TestCase):
    def test_eligible_candidate_can_miss_preferred_skills(self) -> None:
        profile = candidate(
            "candidate_a",
            "a.txt",
            ["python", "sql", "pandas"],
            2.0,
            ["bachelor_of_technology"],
        )
        result = compare_candidate(profile, JOB)

        self.assertTrue(result.eligible)
        self.assertEqual(result.mandatory_skill_coverage, 1.0)
        self.assertEqual(result.additional_skill_coverage, 0.3333)
        self.assertEqual(result.failed_requirements, [])

    def test_missing_mandatory_skill_fails_eligibility(self) -> None:
        profile = candidate(
            "candidate_b",
            "b.txt",
            ["python", "pandas", "power_bi", "machine_learning"],
            3.0,
            ["bachelor_of_technology"],
        )
        result = compare_candidate(profile, JOB)

        self.assertFalse(result.eligible)
        self.assertEqual(result.missing_mandatory_skills, ["sql"])
        self.assertTrue(any("Missing mandatory skills" in item for item in result.failed_requirements))

    def test_eligible_candidates_rank_before_ineligible_candidates(self) -> None:
        eligible = candidate(
            "eligible",
            "eligible.txt",
            ["python", "sql"],
            1.0,
            ["bachelor_of_technology"],
        )
        ineligible = candidate(
            "ineligible",
            "ineligible.txt",
            ["python", "pandas", "power_bi", "machine_learning"],
            5.0,
            ["bachelor_of_technology"],
        )
        results = rank_candidates([ineligible, eligible], JOB)

        self.assertEqual(results[0].candidate_id, "eligible")
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_no_scorable_requirements_produce_zero_not_false_perfection(self) -> None:
        empty_job = JobRequirements(job_id="empty", title="Role", description="")
        profile = candidate("candidate", "candidate.txt", ["python"], 1.0, [])
        result = compare_candidate(profile, empty_job)

        self.assertTrue(result.eligible)
        self.assertEqual(result.requirement_score, 0.0)


if __name__ == "__main__":
    unittest.main()
