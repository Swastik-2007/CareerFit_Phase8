import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from careerfit.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Phase3CommandLineTests(unittest.TestCase):
    def test_validate_dataset_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "audit.json"
            exit_code = main(
                [
                    "validate-dataset",
                    str(PROJECT_ROOT / "samples" / "demo_resume_dataset.csv"),
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["dataset_audit"]["usable_rows"], 36)


if __name__ == "__main__":
    unittest.main()
