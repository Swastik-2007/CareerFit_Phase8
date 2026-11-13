import unittest

from careerfit.job import build_job_requirements_from_text


JOB_DESCRIPTION = """Job Title: Data Scientist

ABOUT THE ROLE
Build machine learning solutions and communicate results.

REQUIRED QUALIFICATIONS
- Python and SQL are mandatory.
- At least 2 years of professional experience.
- B.Tech in Computer Science is required.
- AWS Certified Cloud Practitioner is mandatory.

PREFERRED SKILLS
- Pandas, scikit-learn, Power BI, and Docker are preferred.
"""


class JobParserTests(unittest.TestCase):
    def test_extracts_classified_job_requirements(self) -> None:
        job = build_job_requirements_from_text(JOB_DESCRIPTION, source_file="job.txt")

        self.assertEqual(job.title, "Data Scientist")
        self.assertEqual(job.minimum_experience_years, 2.0)
        self.assertEqual(job.mandatory_skills, ["python", "sql"])
        self.assertEqual(
            job.preferred_skills,
            ["docker", "pandas", "power_bi", "scikit_learn"],
        )
        self.assertIn("machine_learning", job.relevant_skills)
        self.assertIn("bachelor_of_technology", job.required_degrees)
        self.assertIn("computer_science", job.required_fields_of_study)
        self.assertEqual(job.mandatory_certifications, ["aws_cloud_practitioner"])
        self.assertIn("requirements", job.detected_sections)
        self.assertIn("preferred", job.detected_sections)

    def test_omits_full_description_by_default(self) -> None:
        job = build_job_requirements_from_text(JOB_DESCRIPTION)
        self.assertNotIn("description", job.to_dict())
        self.assertIn("description", job.to_dict(include_text=True))

    def test_unclassified_skills_are_not_silently_mandatory(self) -> None:
        job = build_job_requirements_from_text(
            "Data Analyst\nWork with Python and Tableau to create reports."
        )
        self.assertEqual(job.mandatory_skills, [])
        self.assertEqual(job.relevant_skills, ["python", "tableau"])


if __name__ == "__main__":
    unittest.main()
