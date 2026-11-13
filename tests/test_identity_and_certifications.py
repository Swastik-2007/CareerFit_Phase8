import unittest

from careerfit.extraction.certifications import extract_certifications
from careerfit.extraction.identity import extract_identity


class IdentityAndCertificationTests(unittest.TestCase):
    def test_extracts_professional_links(self) -> None:
        text = """SWASTIK DAS
swastik@example.com
linkedin.com/in/swastik-das
https://github.com/swastik-das
https://swastik.dev
"""
        identity = extract_identity(text)
        self.assertEqual(identity.linkedin_url, "https://linkedin.com/in/swastik-das")
        self.assertEqual(identity.github_url, "https://github.com/swastik-das")
        self.assertEqual(identity.portfolio_url, "https://swastik.dev")

    def test_extracts_known_certifications(self) -> None:
        text = "AWS Certified Cloud Practitioner and TensorFlow Developer Certificate"
        self.assertEqual(
            extract_certifications(text),
            ["aws_cloud_practitioner", "tensorflow_developer"],
        )


if __name__ == "__main__":
    unittest.main()

