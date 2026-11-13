import unittest

from careerfit.extraction import build_candidate_profile_from_text


RESUME = """SWASTIK DAS
swastik@example.com | +91 98765 43210

SUMMARY
Machine learning practitioner with 2 years of experience.

SKILLS
Python, SQL, Pandas, C++, C#, .NET, sklearn, Machine Learning

EDUCATION
B.Tech in Computer Science and Engineering with Data Science

EXPERIENCE
Data Analyst | January 2023 - January 2025
"""


class ProfileBuilderTests(unittest.TestCase):
    def test_builds_structured_profile(self) -> None:
        profile = build_candidate_profile_from_text(RESUME, source_file="resume.txt")

        self.assertEqual(profile.identity.name, "Swastik Das")
        self.assertEqual(profile.identity.email, "swastik@example.com")
        self.assertEqual(profile.identity.phone, "+919876543210")
        self.assertEqual(profile.experience_years, 2.0)
        self.assertIn("python", profile.skills)
        self.assertIn("cpp", profile.skills)
        self.assertIn("csharp", profile.skills)
        self.assertIn("machine_learning", profile.skills)
        self.assertIn("scikit_learn", profile.skills)
        self.assertIn("bachelor_of_technology", profile.degrees)
        self.assertIn("computer_science", profile.fields_of_study)
        self.assertIn("skills", profile.detected_sections)

    def test_identity_is_separate_and_text_is_private_by_default(self) -> None:
        profile = build_candidate_profile_from_text(RESUME, source_file="resume.txt")
        payload = profile.to_dict()

        self.assertIn("identity", payload)
        self.assertNotIn("cleaned_text", payload)
        self.assertIn("cleaned_text", profile.to_dict(include_text=True))

    def test_candidate_id_is_deterministic(self) -> None:
        first = build_candidate_profile_from_text(RESUME, source_file="resume.txt")
        second = build_candidate_profile_from_text(RESUME, source_file="resume.txt")
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_education_dates_are_not_counted_as_experience(self) -> None:
        text = "JANE DOE\nEDUCATION\nB.Tech Computer Science | 2022 - 2026\nSKILLS\nPython"
        profile = build_candidate_profile_from_text(text, source_file="fresher.txt")
        self.assertEqual(profile.experience_years, 0.0)


if __name__ == "__main__":
    unittest.main()
