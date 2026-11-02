from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .core import Draw, load_csv

#: Primary source. Kept on HTTPS: the old ``http://`` form failed behind
#: Kubernetes egress proxies (Kaggle) with a plain "connection reset".
DEFAULT_SSQ_URL = "https://data.17500.cn/ssq_asc.txt"
#: Same dump over plain HTTP, kept only as a fallback for restricted networks.
LEGACY_SSQ_URL = "http://data.17500.cn/ssq_asc.txt"
#: Official China Welfare Lottery JSON endpoint (paginated, needs a Referer).
OFFICIAL_API_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
OFFICIAL_REFERER = "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/"
OFFICIAL_PAGE_SIZE = 100
OFFICIAL_MAX_PAGES = 80
#: Browsers-only UA: the official endpoint answers 403 to non-browser agents.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
#: Bundled snapshot, the last resort when every network source fails.
SNAPSHOT_PATH = Path(__file__).with_name("data") / "ssq_snapshot.csv"

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


# --------------------------------------------------------------------------- #
# Official JSON source
# --------------------------------------------------------------------------- #


def _official_request(page_no: int, page_size: int, timeout: int) -> dict:
    query = urlencode(
        {
            "name": "ssq",
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": "",
            "dayEnd": "",
            "pageNo": page_no,
            "pageSize": page_size,
            "week": "",
            "systemType": "PC",
        }
    )
    request = Request(
        f"{OFFICIAL_API_URL}?{query}",
        headers={
            "User-Agent": BROWSER_UA,
            "Referer": OFFICIAL_REFERER,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _official_item_to_draw(item: dict) -> Draw:
    issue = str(item["code"]).strip()
    # The endpoint returns e.g. ``2026-11-01(日)``; keep only the ISO date.
    draw_date = str(item["date"]).strip().split("(")[0].split(" ")[0]
    raw_reds = str(item["red"]).replace("，", ",")
    reds = tuple(sorted(int(token) for token in raw_reds.split(",") if token.strip()))
    blue = int(str(item["blue"]).strip())
    return Draw(issue, draw_date, reds, blue)


def fetch_official_draws_with_stats(
    *,
    timeout: int = 30,
    page_size: int = OFFICIAL_PAGE_SIZE,
    max_pages: int = OFFICIAL_MAX_PAGES,
) -> tuple[list[Draw], ParseStats]:
    """Page through the official endpoint and return validated draws plus an audit record."""
    found: dict[str, Draw] = {}
    reasons: dict[str, int] = {}
    seen = 0
    total_reported = 0
    page = 1
    while page <= max_pages:
        payload = _official_request(page, page_size, timeout)
        if payload.get("state") not in (0, "0"):
            raise RuntimeError(f"official API rejected the query: {payload.get('message')!r}")
        items = payload.get("result") or []
        if not items:
            break
        total_reported = int(payload.get("total") or total_reported)
        for item in items:
            seen += 1
            try:
                draw = _official_item_to_draw(item)
            except (KeyError, TypeError, ValueError) as exc:
                key = f"invalid_item:{type(exc).__name__}"
                reasons[key] = reasons.get(key, 0) + 1
                continue
            found[draw.issue] = draw
        if total_reported and len(found) >= total_reported:
            break
        page += 1
    if not found:
        raise ValueError("official API returned no usable draws")
    stats = ParseStats(
        total_lines=seen,
        parsed_lines=len(found),
        skipped_line_count=seen - len(found),
        skipped_reasons=dict(sorted(reasons.items())),
    )
    return sorted(found.values(), key=lambda draw: (draw.draw_date, draw.issue)), stats


def fetch_official_draws(*, timeout: int = 30) -> list[Draw]:
    draws, _stats = fetch_official_draws_with_stats(timeout=timeout)
    return draws


# --------------------------------------------------------------------------- #
# Bundled snapshot and multi-source fallback
# --------------------------------------------------------------------------- #


def load_snapshot() -> list[Draw]:
    """Load the bundled CSV snapshot shipped with the package.

    It exists so a notebook can still run when every network source is blocked
    (Kaggle behind a restrictive egress proxy is the case that motivated it).
    The snapshot is a frozen copy: its last issue is only as new as the release.
    """
    if not SNAPSHOT_PATH.exists():  # pragma: no cover - packaging accident
        raise FileNotFoundError(f"bundled snapshot missing: {SNAPSHOT_PATH}")
    return load_csv(SNAPSHOT_PATH)


@dataclass(frozen=True)
class FetchOutcome:
    """Result of :func:`fetch_ssq_draws_with_fallback`."""

    draws: list[Draw]
    source: str
    parse_stats: ParseStats | None = None
    attempts: tuple[str, ...] = ()

    @property
    def is_snapshot(self) -> bool:
        return self.source == "snapshot"

    @property
    def latest_issue(self) -> str:
        return self.draws[-1].issue if self.draws else ""

    @property
    def latest_date(self) -> str:
        return self.draws[-1].draw_date if self.draws else ""


#: Text sources tried before the official API, in order.
TEXT_SOURCES: tuple[tuple[str, str], ...] = (
    ("17500-https", DEFAULT_SSQ_URL),
    ("17500-http", LEGACY_SSQ_URL),
)


def fetch_ssq_draws_with_fallback(
    *,
    timeout: int = 30,
    allow_snapshot: bool = True,
    text_sources: tuple[tuple[str, str], ...] = TEXT_SOURCES,
) -> FetchOutcome:
    """Try every source in order and never raise unless they all fail.

    Order: text dumps (HTTPS, then HTTP) -> official JSON API -> bundled snapshot.
    Each failure is recorded in ``attempts`` so a run that silently fell back to
    stale data is still auditable instead of looking like a fresh fetch.
    """
    attempts: list[str] = []

    for name, url in text_sources:
        try:
            draws, stats = parse_ssq_text_with_stats(fetch_ssq_text(url, timeout))
            return FetchOutcome(draws, name, stats, tuple(attempts))
        except Exception as exc:  # noqa: BLE001 - every source failure is recorded
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")

    try:
        draws, stats = fetch_official_draws_with_stats(timeout=timeout)
        return FetchOutcome(draws, "cwl-official", stats, tuple(attempts))
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"cwl-official: {type(exc).__name__}: {exc}")

    if allow_snapshot:
        try:
            draws = load_snapshot()
            return FetchOutcome(draws, "snapshot", None, tuple(attempts))
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"snapshot: {type(exc).__name__}: {exc}")

    detail = "\n".join(f"  - {line}" for line in attempts)
    raise RuntimeError(f"every SSQ data source failed:\n{detail}")

