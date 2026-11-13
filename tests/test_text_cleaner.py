import unittest

from careerfit.parsing import clean_text, split_sections


class TextCleanerTests(unittest.TestCase):
    def test_preserves_technical_punctuation(self) -> None:
        cleaned = clean_text("C++   C#   .NET\nMachine-\nlearning")
        self.assertIn("C++", cleaned)
        self.assertIn("C#", cleaned)
        self.assertIn(".NET", cleaned)
        self.assertIn("Machinelearning", cleaned)

    def test_splits_known_sections(self) -> None:
        sections = split_sections("Jane Doe\nSKILLS: Python, SQL\nEDUCATION\nB.Tech CSE")
        self.assertEqual(sections["preamble"], "Jane Doe")
        self.assertEqual(sections["skills"], "Python, SQL")
        self.assertEqual(sections["education"], "B.Tech CSE")


if __name__ == "__main__":
    unittest.main()

