import tempfile
from pathlib import Path
import unittest
from careerfit.gui import build_command


class DesktopCommands(unittest.TestCase):
    def test_paths_with_spaces_stay_one_argument(self):
        cmd = build_command('parse', ['my resume.txt'], '', '', '', 'my output.json', '', '', '')
        self.assertIn('my resume.txt', cmd)
        self.assertEqual(cmd[-1], 'my output.json')

    def test_missing_resumes(self):
        with self.assertRaises(ValueError):
            build_command('score', [], '', '', '', 'out.json', '', '', '')

    def test_search_no_resume_needed(self):
        cmd = build_command('search', [], '', '', '', 'out.json', 'python', 'pandas', 'all')
        self.assertIn('search', cmd)
        self.assertIn('all', cmd)

    def test_match_does_not_require_model(self):
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)/'job.txt'; job.touch()
            cmd = build_command('match', ['resume.txt'], str(job), '', '', 'out.json', '', '', '')
            self.assertNotIn('--model', cmd)

    def test_score_requires_model(self):
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)/'job.txt'; job.touch()
            with self.assertRaises(ValueError):
                build_command('score', ['resume.txt'], str(job), 'missing', '', 'out.json', '', '', '')
