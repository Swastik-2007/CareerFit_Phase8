"""Degree and field-of-study extraction."""

from __future__ import annotations

import re


DEGREE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("doctor_of_philosophy", r"\b(?:ph\.?d\.?|doctor of philosophy)\b"),
    ("master_of_technology", r"\b(?:m\.?\s?tech\.?|master of technology)\b"),
    ("master_of_science", r"\b(?:m\.?\s?sc\.?|master of science)\b"),
    ("master_of_computer_applications", r"\b(?:m\.?c\.?a\.?|master of computer applications)\b"),
    ("master_of_business_administration", r"\b(?:m\.?b\.?a\.?|master of business administration)\b"),
    ("bachelor_of_technology", r"\b(?:b\.?\s?tech\.?|bachelor of technology)\b"),
    ("bachelor_of_engineering", r"\b(?:b\.?e\.?|bachelor of engineering)\b"),
    ("bachelor_of_science", r"\b(?:b\.?\s?sc\.?|bachelor of science)\b"),
    ("bachelor_of_computer_applications", r"\b(?:b\.?c\.?a\.?|bachelor of computer applications)\b"),
    ("master_degree", r"\bmaster(?:'s|s)? degree\b"),
    ("bachelor_degree", r"\bbachelor(?:'s|s)? degree\b"),
    ("diploma", r"\b(?:polytechnic diploma|diploma)\b"),
)

FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("computer_science", r"\b(?:computer science(?: and engineering)?|cse)\b"),
    ("data_science", r"\bdata science\b"),
    ("machine_learning", r"\bmachine learning\b"),
    ("information_technology", r"\binformation technology\b"),
    ("electronics_and_communication", r"\b(?:electronics and communication|ece)\b"),
    ("electrical_engineering", r"\belectrical engineering\b"),
    ("mathematics", r"\bmathematics\b"),
    ("statistics", r"\bstatistics\b"),
    ("business_administration", r"\bbusiness administration\b"),
)


def _ordered_matches(text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    found: list[str] = []
    for canonical, pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE) and canonical not in found:
            found.append(canonical)
    return found


def extract_education(text: str) -> tuple[list[str], list[str]]:
    return _ordered_matches(text, DEGREE_PATTERNS), _ordered_matches(text, FIELD_PATTERNS)
