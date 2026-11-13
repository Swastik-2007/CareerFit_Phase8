import unittest
from datetime import date

from careerfit.extraction.experience import extract_experience_years


class ExperienceExtractionTests(unittest.TestCase):
    def test_explicit_total_is_preferred(self) -> None:
        text = "Total professional experience: 3.5 years"
        self.assertEqual(extract_experience_years(text), 3.5)

    def test_overlapping_date_ranges_are_not_double_counted(self) -> None:
        text = "January 2023 - January 2025\nJune 2024 - January 2026"
        result = extract_experience_years(text, reference_date=date(2026, 1, 1))
        self.assertEqual(result, 3.0)

    def test_present_uses_reference_date(self) -> None:
        text = "June 2024 - Present"
        result = extract_experience_years(text, reference_date=date(2025, 6, 1))
        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()

