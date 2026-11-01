from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.request import Request, urlopen

from .core import Draw

DEFAULT_SSQ_URL = "http://data.17500.cn/ssq_asc.txt"

RED_BALL_COUNT = 33
BLUE_BALL_COUNT = 16
REDS_PER_DRAW = 6

_DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})(?:(?P<sep>[-/])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})|(?P<compact>\d{4}))"
)
_ISSUE_PATTERN = re.compile(r"\b\d{5,8}\b")
#: Any standalone 1-2 digit token, used only for the ball columns after the date.
_NUMBER_PATTERN = re.compile(r"\b\d{1,2}\b")


@dataclass(frozen=True)
class ParseStats:
    """Audit trail for one :func:`parse_ssq_text_with_stats` call.

    ``skipped_line_count`` only counts lines that *looked* like data rows (they
    contained a date) but failed validation, so header and footer noise does not
    inflate the number. Silent row loss is exactly how a bad column offset used to
    shrink the history without anyone noticing.
    """

    total_lines: int = 0
    parsed_lines: int = 0
    skipped_line_count: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)


class _LineSkip(Exception):
    """Raised internally when a candidate data line cannot be trusted."""


def fetch_ssq_text(url: str = DEFAULT_SSQ_URL, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": "ssq-quant-research-system/0.2"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _parse_ssq_line(line: str) -> Draw:
    """Parse one line of the 17500 text dump, raising :class:`_LineSkip` on doubt."""
    date_match = _DATE_PATTERN.search(line)
    if date_match is None:
        raise _LineSkip("no_date")
    issue_matches = _ISSUE_PATTERN.findall(line[: date_match.start()])
    if not issue_matches:
        raise _LineSkip("missing_issue")
    issue = issue_matches[-1]
    try:
        draw_date = date(
            int(date_match.group("year")),
            int(date_match.group("month") or date_match.group("compact")[:2]),
            int(date_match.group("day") or date_match.group("compact")[2:]),
        ).isoformat()
    except ValueError as exc:
        raise _LineSkip("invalid_date") from exc
    numbers = [int(token) for token in _NUMBER_PATTERN.findall(line[date_match.end():])]
    if len(numbers) < REDS_PER_DRAW + 1:
        raise _LineSkip("too_few_numbers")
    reds = tuple(sorted(numbers[:REDS_PER_DRAW]))
    blue = numbers[REDS_PER_DRAW]
    # Explicit checks with a named reason: without them a mis-aligned row either
    # looks fine or disappears silently inside a bare try/except.
    if len(set(reds)) != REDS_PER_DRAW or not all(1 <= n <= RED_BALL_COUNT for n in reds):
        raise _LineSkip("invalid_reds")
    if not 1 <= blue <= BLUE_BALL_COUNT:
        raise _LineSkip("invalid_blue")
    try:
        return Draw(issue, draw_date, reds, blue)
    except ValueError as exc:  # pragma: no cover - guard if Draw rules tighten
        raise _LineSkip("invalid_draw") from exc


def parse_ssq_text_with_stats(text: str) -> tuple[list[Draw], ParseStats]:
    """Parse the downloaded text and return both the draws and an audit record."""
    draws: list[Draw] = []
    reasons: dict[str, int] = {}
    total_lines = 0
    skipped = 0
    for line in text.splitlines():
        total_lines += 1
        if not line.strip():
            continue
        try:
            draws.append(_parse_ssq_line(line))
        except _LineSkip as exc:
            reason = str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason != "no_date":
                skipped += 1
    unique = {draw.issue: draw for draw in draws}
    if not unique:
        raise ValueError("no valid SSQ draws found in downloaded text")
    stats = ParseStats(
        total_lines=total_lines,
        parsed_lines=len(unique),
        skipped_line_count=skipped,
        skipped_reasons=dict(sorted(reasons.items())),
    )
    return sorted(unique.values(), key=lambda draw: (draw.draw_date, draw.issue)), stats


def parse_ssq_text(text: str) -> list[Draw]:
    """Parse the downloaded text into validated draws (unchanged 0.1.0 contract).

    Use :func:`parse_ssq_text_with_stats` when the number of skipped lines
    matters; this wrapper keeps existing callers working.
    """
    draws, _stats = parse_ssq_text_with_stats(text)
    return draws


def fetch_ssq_draws(url: str = DEFAULT_SSQ_URL, timeout: int = 20) -> list[Draw]:
    return parse_ssq_text(fetch_ssq_text(url, timeout))


def fetch_ssq_draws_with_stats(url: str = DEFAULT_SSQ_URL, timeout: int = 20) -> tuple[list[Draw], ParseStats]:
    """Fetch and parse in one call, keeping the row-level audit trail."""
    return parse_ssq_text_with_stats(fetch_ssq_text(url, timeout))
