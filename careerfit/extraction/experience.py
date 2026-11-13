"""Conservative extraction of total professional experience."""

from __future__ import annotations

import re
from datetime import date


EXPLICIT_PATTERNS = (
    re.compile(
        r"\b(?:total\s+)?(?:professional\s+|relevant\s+)?experience\s*[:=-]?\s*"
        r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?"
        r"(?:professional\s+|relevant\s+|work\s+)?experience\b",
        re.I,
    ),
)

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_TOKEN = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_RANGE_PATTERN = re.compile(
    rf"\b(?:(?P<start_month>{MONTH_TOKEN})\.?\s+)?"
    r"(?P<start_year>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    rf"(?:(?:(?P<end_month>{MONTH_TOKEN})\.?\s+)?"
    r"(?P<end_year>(?:19|20)\d{2})|(?P<present>present|current|now))\b",
    re.I,
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _merge_month_intervals(intervals: list[tuple[int, int]]) -> int:
    """Return months in the union of intervals, avoiding overlap double-counting."""

    if not intervals:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(max(0, end - start) for start, end in merged)


def extract_experience_years(
    text: str,
    *,
    reference_date: date | None = None,
    allow_date_ranges: bool = True,
) -> float:
    """Extract explicit total experience, falling back to dated work intervals."""

    for pattern in EXPLICIT_PATTERNS:
        matches = [float(match.group("years")) for match in pattern.finditer(text)]
        if matches:
            return round(max(matches), 2)

    if not allow_date_ranges:
        return 0.0

    today = reference_date or date.today()
    intervals: list[tuple[int, int]] = []
    for match in DATE_RANGE_PATTERN.finditer(text):
        start_year = int(match.group("start_year"))
        start_month = MONTHS.get((match.group("start_month") or "jan").casefold(), 1)

        if match.group("present"):
            end_year, end_month = today.year, today.month
        else:
            end_year = int(match.group("end_year"))
            end_month = MONTHS.get((match.group("end_month") or "jan").casefold(), 1)

        start_index = _month_index(start_year, start_month)
        end_index = _month_index(end_year, end_month)
        if 0 <= end_index - start_index <= 600:
            intervals.append((start_index, end_index))

    months = _merge_month_intervals(intervals)
    return round(months / 12, 2)
