import json
import tempfile
import unittest
from pathlib import Path

from careerfit.growth import SkillGraph, SkillGraphError


class SkillGraphTests(unittest.TestCase):
    def test_default_graph_is_acyclic_and_substantial(self) -> None:
        graph = SkillGraph()
        self.assertGreaterEqual(graph.node_count, 50)
        self.assertGreater(graph.edge_count, 30)
        self.assertTrue(graph.metadata()["acyclic"])

    def test_learning_plan_excludes_existing_prerequisites(self) -> None:
        graph = SkillGraph()
        plan = graph.learning_plan(
            "data_analysis",
            {"python", "sql", "numpy", "pandas"},
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.learning_path, ["statistics", "data_analysis"])
        self.assertEqual(plan.prerequisites_to_learn, ["statistics"])
        self.assertEqual(plan.estimated_hours, 65)

    def test_cycle_is_rejected(self) -> None:
        payload = {
            "version": "test",
            "description": "cycle",
            "nodes": {
                "python": {
                    "prerequisites": ["sql"],
                    "estimated_hours": 1,
                    "difficulty": "test",
                    "rationale": "test"
                },
                "sql": {
                    "prerequisites": ["python"],
                    "estimated_hours": 1,
                    "difficulty": "test",
                    "rationale": "test"
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SkillGraphError):
                SkillGraph(path)

    def test_mastered_skill_stops_prerequisite_backtracking(self) -> None:
        graph = SkillGraph()
        plan = graph.learning_plan(
            "data_science",
            {"python", "sql", "data_analysis"},
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertNotIn("pandas", plan.learning_path)
        self.assertNotIn("numpy", plan.learning_path)
        self.assertEqual(plan.learning_path, ["statistics", "machine_learning", "data_science"])

    def test_combined_learning_order_is_deterministic(self) -> None:
        graph = SkillGraph()
        selected = {"sql", "statistics", "machine_learning", "data_science"}
        expected = ["sql", "statistics", "machine_learning", "data_science"]
        self.assertEqual(graph.order_skills(selected), expected)
        self.assertEqual(graph.order_skills(reversed(expected)), expected)


if __name__ == "__main__":
    unittest.main()
