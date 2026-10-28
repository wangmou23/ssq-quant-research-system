from __future__ import annotations

import re
from datetime import date
from urllib.request import Request, urlopen

from .core import Draw

DEFAULT_SSQ_URL = "http://data.17500.cn/ssq_asc.txt"


def fetch_ssq_text(url: str = DEFAULT_SSQ_URL, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": "ssq-quant-research-system/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_ssq_text(text: str) -> list[Draw]:
    draws: list[Draw] = []
    for line in text.splitlines():
        date_match = re.search(
            r"(?P<year>20\d{2})(?:(?P<sep>[-/])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})|(?P<compact>\d{4}))",
            line,
        )
        if date_match is None:
            continue
        issue_matches = re.findall(r"\b\d{5,8}\b", line[:date_match.start()])
        if not issue_matches:
            continue
        issue = issue_matches[-1]
        try:
            draw_date = date(
                int(date_match.group("year")),
                int(date_match.group("month") or date_match.group("compact")[:2]),
                int(date_match.group("day") or date_match.group("compact")[2:]),
            ).isoformat()
            numbers = [int(token) for token in re.findall(r"\b\d{1,2}\b", line[date_match.end():])]
            if len(numbers) < 7:
                continue
            draws.append(Draw(issue, draw_date, tuple(sorted(numbers[:6])), numbers[6]))
        except (ValueError, IndexError):
            continue
    unique = {draw.issue: draw for draw in draws}
    if not unique:
        raise ValueError("no valid SSQ draws found in downloaded text")
    return sorted(unique.values(), key=lambda draw: (draw.draw_date, draw.issue))


def fetch_ssq_draws(url: str = DEFAULT_SSQ_URL, timeout: int = 20) -> list[Draw]:
    return parse_ssq_text(fetch_ssq_text(url, timeout))

