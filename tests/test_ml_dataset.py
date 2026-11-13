import csv
import tempfile
import unittest
from pathlib import Path

from careerfit.ml.dataset import (
    DatasetValidationError,
    load_and_validate_dataset,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATASET = PROJECT_ROOT / "samples" / "demo_resume_dataset.csv"


class DatasetPipelineTests(unittest.TestCase):
    def test_demo_dataset_is_valid_and_balanced(self) -> None:
        frame, audit = load_and_validate_dataset(DEMO_DATASET)
        self.assertEqual(len(frame), 36)
        self.assertEqual(audit.class_counts["Data Scientist"], 12)
        self.assertEqual(len(audit.class_counts), 3)
        self.assertEqual(len(audit.content_fingerprint), 64)

    def test_splits_are_reproducible_and_disjoint(self) -> None:
        frame, _ = load_and_validate_dataset(DEMO_DATASET)
        first = split_dataset(frame, seed=73)
        second = split_dataset(frame, seed=73)
        self.assertEqual(first.manifest(), second.manifest())
        fingerprints = [
            set(first.train["fingerprint"]),
            set(first.validation["fingerprint"]),
            set(first.test["fingerprint"]),
        ]
        self.assertFalse(fingerprints[0] & fingerprints[1])
        self.assertFalse(fingerprints[0] & fingerprints[2])
        self.assertFalse(fingerprints[1] & fingerprints[2])

    def test_conflicting_duplicate_labels_are_rejected(self) -> None:
        rows = []
        for label in ("Role A", "Role B"):
            for index in range(8):
                rows.append(
                    {
                        "record_id": f"{label}-{index}",
                        "resume_text": f"unique resume {label} example {index}",
                        "target_role": label,
                    }
                )
        rows.append(
            {
                "record_id": "conflict",
                "resume_text": "unique resume Role A example 0",
                "target_role": "Role B",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(DatasetValidationError):
                load_and_validate_dataset(path)


if __name__ == "__main__":
    unittest.main()
