import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from careerfit.cli import main
from careerfit.extraction import build_candidate_profile, build_candidate_profile_from_text
from careerfit.fairness import audit_fairness, mask_candidate, select_review_threshold
from careerfit.job import build_job_requirements
from careerfit.ml import load_model_bundle, train_and_evaluate
from careerfit.scoring import build_scoring_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FairnessMaskingTests(unittest.TestCase):
    def test_masking_removes_identity_proxies_and_preserves_qualifications(self) -> None:
        candidate = build_candidate_profile_from_text(
            """Jane Doe
jane.doe@example.com | +91 98765 43210
SUMMARY
She is a Python developer and her focus is data analysis.
EDUCATION
B.Tech in Computer Science
Example Technical University
SKILLS
Python, SQL
""",
            source_file="jane.txt",
        )
        masked = mask_candidate(candidate)
        text = masked.profile.cleaned_text.casefold()

        self.assertNotIn("jane doe", text)
        self.assertNotIn("jane.doe@example.com", text)
        self.assertNotIn("example technical university", text)
        self.assertIsNone(re.search(r"\b(?:she|her)\b", text))
        self.assertIn("b.tech", text)
        self.assertIn("computer science", text)
        self.assertEqual(masked.profile.skills, candidate.skills)
        self.assertEqual(masked.profile.degrees, candidate.degrees)
        self.assertEqual(masked.profile.fields_of_study, candidate.fields_of_study)
        self.assertEqual(masked.profile.experience_years, candidate.experience_years)
        self.assertGreaterEqual(masked.summary.total_occurrences, 5)

    def test_ambiguous_institution_acronym_is_reported_not_removed(self) -> None:
        candidate = build_candidate_profile_from_text(
            """Asha Verma
SKILLS
Python and SQL
EDUCATION
B.Tech Computer Science
LPU | 2025 - 2029
""",
            source_file="asha.txt",
        )
        masked = mask_candidate(candidate)
        self.assertTrue(masked.summary.institution_acronym_warning)
        self.assertIn("LPU", masked.profile.cleaned_text)

    def test_small_cohort_uses_minimum_threshold(self) -> None:
        analysis = select_review_threshold([0.001, 0.4], minimum_threshold=0.01)
        self.assertEqual(analysis["method"], "small_sample_minimum")
        self.assertFalse(analysis["reliable_cohort_estimate"])
        self.assertEqual(analysis["applied_review_threshold"], 0.01)
        self.assertIsNotNone(analysis["warning"])

    def test_robust_threshold_does_not_follow_single_outlier(self) -> None:
        analysis = select_review_threshold(
            [0.001, 0.001, 0.002, 0.002, 0.1],
            minimum_threshold=0.001,
        )
        self.assertEqual(analysis["method"], "robust_median_plus_mad")
        self.assertTrue(analysis["reliable_cohort_estimate"])
        self.assertLess(analysis["applied_review_threshold"], 0.01)


class FairnessAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.artifacts = Path(cls.temporary.name) / "artifacts"
        cls.summary = train_and_evaluate(
            PROJECT_ROOT / "samples" / "demo_resume_dataset.csv",
            cls.artifacts,
            seed=42,
        )
        cls.bundle = load_model_bundle(cls.summary["model_path"])
        cls.job = build_job_requirements(
            PROJECT_ROOT / "samples" / "sample_job_description.txt"
        )
        cls.context = build_scoring_context(cls.job, cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_audit_preserves_requirement_score_and_automatic_decision(self) -> None:
        candidates = [
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt"),
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume_gap.txt"),
        ]
        report = audit_fairness(candidates, self.job, self.bundle, self.context)

        self.assertEqual(len(report.results), 2)
        self.assertEqual(report.threshold_analysis["method"], "small_sample_minimum")
        for result in report.results:
            self.assertEqual(result.component_changes["requirement_score"], 0)
            self.assertEqual(result.original_eligible, result.masked_eligible)
            self.assertFalse(result.automatic_decision_changed)
            self.assertIn(
                result.audit_disposition,
                {"human_review", "no_additional_review_flag"},
            )

    def test_fairness_cli_writes_phase6_output_without_masked_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "fairness.json"
            exit_code = main(
                [
                    "fairness-audit",
                    "--model",
                    self.summary["model_path"],
                    "--job",
                    str(PROJECT_ROOT / "samples" / "sample_job_description.txt"),
                    "--resumes",
                    str(PROJECT_ROOT / "samples" / "sample_resume.txt"),
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["careerfit_phase"], 6)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertTrue(payload["original_ranking_preserved"])
        self.assertNotIn("masked_text", json.dumps(payload))
        self.assertEqual(payload["errors"], [])


if __name__ == "__main__":
    unittest.main()
