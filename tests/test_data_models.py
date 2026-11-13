import unittest

from careerfit.models import ScoreResult


class ScoreResultTests(unittest.TestCase):
    def test_scores_accept_inclusive_zero_to_one_range(self) -> None:
        result = ScoreResult(
            candidate_id="candidate_1",
            job_id="job_1",
            text_similarity=0.0,
            role_probability=0.5,
            requirement_score=1.0,
            fit_score=0.6,
            eligible=True,
        )
        self.assertEqual(result.fit_score, 0.6)

    def test_scores_reject_values_above_one(self) -> None:
        with self.assertRaises(ValueError):
            ScoreResult(
                candidate_id="candidate_1",
                job_id="job_1",
                text_similarity=1.01,
                role_probability=0.5,
                requirement_score=1.0,
                fit_score=0.6,
                eligible=True,
            )


if __name__ == "__main__":
    unittest.main()

