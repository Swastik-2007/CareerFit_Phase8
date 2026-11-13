import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from careerfit.extraction import build_candidate_profile
from careerfit.growth import SkillGraph, analyze_growth
from careerfit.job import build_job_requirements
from careerfit.ml import (
    candidate_model_text,
    load_model_bundle,
    predict_text,
    text_similarity,
    train_and_evaluate,
)
from careerfit.scoring import (
    build_scoring_context,
    normalize_weights,
    score_candidates_with_context,
    score_candidates_with_model,
)
from careerfit.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TrainingAndScoringTests(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_training_writes_reproducible_artifacts(self) -> None:
        expected = {
            "careerfit_role_model.joblib",
            "evaluation_report.json",
            "dataset_audit.json",
            "split_manifest.json",
            "MODEL_CARD.md",
            "confusion_matrix.csv",
            "test_predictions.csv",
        }
        self.assertTrue(expected.issubset({path.name for path in self.artifacts.iterdir()}))
        report = json.loads((self.artifacts / "evaluation_report.json").read_text())
        self.assertEqual(report["careerfit_phase"], 3)
        self.assertIn(report["selected_model"], report["validation_metrics"])
        self.assertEqual(report["dataset_metadata"]["source_type"], "synthetic_demo")
        self.assertEqual(
            report["model_probability_methods"]["linear_svm"],
            "three_fold_sigmoid_calibration",
        )

    def test_prediction_returns_a_distribution(self) -> None:
        result = predict_text(
            self.bundle,
            "Python SQL pandas statistics classification and model evaluation",
        )
        self.assertIn(result["predicted_role"], self.bundle["classes"])
        self.assertAlmostEqual(sum(result["role_probabilities"].values()), 1.0, places=5)

    def test_text_similarity_is_bounded(self) -> None:
        score = text_similarity("Python SQL machine learning", "Requires Python and SQL")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_candidate_identity_is_removed_from_ml_text(self) -> None:
        candidate = build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt")
        feature_text = candidate_model_text(candidate).casefold()
        self.assertNotIn((candidate.identity.name or "").casefold(), feature_text)
        self.assertNotIn((candidate.identity.email or "").casefold(), feature_text)

    def test_hybrid_scoring_preserves_eligibility(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        candidates = [
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt"),
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume_gap.txt"),
        ]
        target_role, results = score_candidates_with_model(candidates, job, self.bundle)
        self.assertIn(target_role, self.bundle["classes"])
        self.assertTrue(results[0].eligible)
        self.assertFalse(results[1].eligible)
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertTrue(all(0.0 <= result.fit_score <= 1.0 for result in results))

    def test_component_contributions_reconstruct_fit_score(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        candidate = build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt")
        context = build_scoring_context(job, self.bundle)
        result = score_candidates_with_context([candidate], job, self.bundle, context)[0]
        self.assertAlmostEqual(
            sum(result.component_contributions.values()),
            result.fit_score,
            places=6,
        )
        self.assertIn("Final fit score", result.explanation[-2])

    def test_custom_weights_are_normalized(self) -> None:
        weights = normalize_weights(
            {
                "text_similarity": 2,
                "role_probability": 1,
                "requirement_score": 1,
            }
        )
        self.assertEqual(weights["text_similarity"], 0.5)
        self.assertEqual(weights["role_probability"], 0.25)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_invalid_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_weights(
                {
                    "text_similarity": -1,
                    "role_probability": 1,
                    "requirement_score": 1,
                }
            )

    def test_target_role_override_is_recorded(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        context = build_scoring_context(
            job,
            self.bundle,
            target_role_override="data scientist",
        )
        self.assertEqual(context.target_role, "Data Scientist")
        self.assertEqual(context.target_role_source, "user_override")
        self.assertIsNone(context.job_role_confidence)

    def test_unknown_target_role_is_rejected(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        with self.assertRaises(ValueError):
            build_scoring_context(job, self.bundle, target_role_override="Astronaut")

    def test_growth_plan_uses_missing_job_skills(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        candidates = [
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt"),
            build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume_gap.txt"),
        ]
        context = build_scoring_context(job, self.bundle)
        results = analyze_growth(candidates, job, self.bundle, context, SkillGraph())
        swastik = next(item for item in results if item.candidate_name == "Swastik Das")
        asha = next(item for item in results if item.candidate_name == "Asha Verma")
        self.assertEqual(
            {item.target_skill for item in swastik.recommendations},
            {"data_analysis", "docker"},
        )
        self.assertGreater(swastik.combined_potential_fit_score, swastik.current_fit_score)
        self.assertEqual(asha.recommendations[0].target_skill, "sql")
        self.assertEqual(asha.recommendations[0].priority, "mandatory")
        self.assertFalse(asha.recommendations[0].eligible_after_learning)

    def test_growth_gain_changes_only_requirement_component(self) -> None:
        job = build_job_requirements(PROJECT_ROOT / "samples" / "sample_job_description.txt")
        candidate = build_candidate_profile(PROJECT_ROOT / "samples" / "sample_resume.txt")
        context = build_scoring_context(job, self.bundle)
        result = analyze_growth(
            [candidate], job, self.bundle, context, SkillGraph(), max_recommendations=2
        )[0]
        expected = round(
            context.weights["requirement_score"]
            * (result.combined_requirement_score - result.current_requirement_score),
            6,
        )
        self.assertAlmostEqual(result.combined_score_gain, expected, places=6)
        self.assertLessEqual(result.combined_score_gain, context.weights["requirement_score"])

    def test_growth_cli_writes_phase5_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "growth.json"
            exit_code = main(
                [
                    "growth",
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
        self.assertEqual(payload["careerfit_phase"], 5)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertTrue(payload["skill_graph"]["acyclic"])


if __name__ == "__main__":
    unittest.main()
