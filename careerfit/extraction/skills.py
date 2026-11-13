"""Taxonomy-backed canonical skill extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "skills.json"


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    canonical: str
    category: str
    aliases: tuple[str, ...]


class SkillExtractor:
    """Match aliases and return canonical skills grouped by category."""

    def __init__(self, taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH) -> None:
        taxonomy_file = Path(taxonomy_path)
        payload = json.loads(taxonomy_file.read_text(encoding="utf-8"))
        definitions: list[SkillDefinition] = []
        for category, skills in payload["categories"].items():
            for skill in skills:
                aliases = tuple(dict.fromkeys((skill["canonical"], *skill.get("aliases", []))))
                definitions.append(
                    SkillDefinition(
                        canonical=skill["canonical"],
                        category=category,
                        aliases=aliases,
                    )
                )

        self.definitions = definitions
        self._compiled: list[tuple[SkillDefinition, list[re.Pattern[str]]]] = []
        for definition in definitions:
            patterns = [self._alias_pattern(alias) for alias in definition.aliases]
            self._compiled.append((definition, patterns))

    @staticmethod
    def _alias_pattern(alias: str) -> re.Pattern[str]:
        escaped = re.escape(alias.casefold()).replace(r"\ ", r"[\s_-]+")
        return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)

    def extract_by_category(self, text: str) -> dict[str, list[str]]:
        normalized = text.casefold()
        result: dict[str, list[str]] = {}
        for definition, patterns in self._compiled:
            if any(pattern.search(normalized) for pattern in patterns):
                result.setdefault(definition.category, []).append(definition.canonical)
        return {category: sorted(set(skills)) for category, skills in sorted(result.items())}

    def extract(self, text: str) -> list[str]:
        grouped = self.extract_by_category(text)
        return sorted({skill for skills in grouped.values() for skill in skills})

