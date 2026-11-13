import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from careerfit.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CommandLineTests(unittest.TestCase):
    def test_parse_job_command_writes_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "job.json"
            exit_code = main(
                [
                    "parse-job",
                    str(PROJECT_ROOT / "samples" / "sample_job_description.txt"),
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["job_requirements"]["title"], "Data Science Intern")
        self.assertEqual(payload["job_requirements"]["mandatory_skills"], ["python", "sql"])

    def test_match_command_writes_ranked_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "matches.json"
            exit_code = main(
                [
                    "match",
                    "--job",
                    str(PROJECT_ROOT / "samples" / "sample_job_description.txt"),
                    "--resumes",
                    str(PROJECT_ROOT / "samples" / "sample_resume.txt"),
                    str(PROJECT_ROOT / "samples" / "sample_resume_gap.txt"),
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(payload["ranking"][0]["rank"], 1)
        self.assertTrue(payload["ranking"][0]["eligible"])
        self.assertFalse(payload["ranking"][1]["eligible"])


if __name__ == "__main__":
    unittest.main()

