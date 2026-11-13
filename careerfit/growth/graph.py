"""Validated directed skill-prerequisite graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from careerfit.extraction.skills import DEFAULT_TAXONOMY_PATH


DEFAULT_GRAPH_PATH = Path(__file__).with_name("skill_graph.json")


class SkillGraphError(ValueError):
    """Raised when a graph cannot support defensible learning paths."""


@dataclass(slots=True)
class LearningPlan:
    target_skill: str
    learning_path: list[str]
    prerequisites_to_learn: list[str]
    learning_steps: int
    estimated_hours: int
    difficulty: str
    rationale: str


def _taxonomy_skills(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        item["canonical"]
        for values in payload.get("categories", {}).values()
        for item in values
    }


class SkillGraph:
    def __init__(
        self,
        path: str | Path = DEFAULT_GRAPH_PATH,
        *,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillGraphError(f"Could not read skill graph {self.path}: {exc}") from exc

        self.version = str(payload.get("version", "")).strip()
        self.description = str(payload.get("description", "")).strip()
        self.methodology = str(payload.get("methodology", "")).strip()
        nodes = payload.get("nodes")
        if not self.version or not isinstance(nodes, dict) or not nodes:
            raise SkillGraphError("Skill graph requires a version and a non-empty nodes object.")

        taxonomy = _taxonomy_skills(taxonomy_path)
        unknown = sorted(set(nodes) - taxonomy)
        if unknown:
            raise SkillGraphError(
                "Graph contains skills outside the current taxonomy: " + ", ".join(unknown)
            )

        graph = nx.DiGraph()
        for skill, metadata in nodes.items():
            if not isinstance(metadata, dict):
                raise SkillGraphError(f"Metadata for '{skill}' must be an object.")
            prerequisites = metadata.get("prerequisites", [])
            if not isinstance(prerequisites, list) or not all(
                isinstance(item, str) for item in prerequisites
            ):
                raise SkillGraphError(f"Prerequisites for '{skill}' must be a string list.")
            try:
                hours = int(metadata.get("estimated_hours", 0))
            except (TypeError, ValueError) as exc:
                raise SkillGraphError(f"estimated_hours for '{skill}' must be an integer.") from exc
            if hours <= 0:
                raise SkillGraphError(f"estimated_hours for '{skill}' must be positive.")
            graph.add_node(
                skill,
                estimated_hours=hours,
                difficulty=str(metadata.get("difficulty", "unspecified")),
                rationale=str(metadata.get("rationale", "")).strip(),
            )

        for skill, metadata in nodes.items():
            for prerequisite in metadata.get("prerequisites", []):
                if prerequisite not in nodes:
                    raise SkillGraphError(
                        f"'{skill}' references missing prerequisite '{prerequisite}'."
                    )
                if prerequisite == skill:
                    raise SkillGraphError(f"'{skill}' cannot be its own prerequisite.")
                graph.add_edge(prerequisite, skill)

        if not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph)
            raise SkillGraphError(f"Skill graph contains a prerequisite cycle: {cycle}")
        self.graph = graph

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "methodology": self.methodology,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "acyclic": True,
        }

    def learning_plan(
        self,
        target_skill: str,
        current_skills: Iterable[str],
    ) -> LearningPlan | None:
        if target_skill not in self.graph:
            return None
        current = set(current_skills)
        required: set[str] = set()

        def collect_missing(skill: str) -> None:
            # A declared current skill is treated as mastered; do not require the
            # learner to retroactively complete every ancestor in this starter graph.
            if skill in current or skill in required:
                return
            for prerequisite in self.graph.predecessors(skill):
                collect_missing(str(prerequisite))
            required.add(skill)

        collect_missing(target_skill)
        order = list(
            nx.lexicographical_topological_sort(
                self.graph.subgraph(required),
                key=str,
            )
        )
        learning_path = list(order)
        prerequisites = [skill for skill in learning_path if skill != target_skill]
        metadata = self.graph.nodes[target_skill]
        return LearningPlan(
            target_skill=target_skill,
            learning_path=learning_path,
            prerequisites_to_learn=prerequisites,
            learning_steps=len(learning_path),
            estimated_hours=sum(
                int(self.graph.nodes[skill]["estimated_hours"]) for skill in learning_path
            ),
            difficulty=str(metadata["difficulty"]),
            rationale=str(metadata["rationale"]),
        )

    def order_skills(self, skills: Iterable[str]) -> list[str]:
        selected = {skill for skill in skills if skill in self.graph}
        return list(
            nx.lexicographical_topological_sort(
                self.graph.subgraph(selected),
                key=str,
            )
        )
