"""Frequency, probability and significance analysis for SSQ research.

Deliberately standard-library only: the dependency-free baseline in ``core`` must
keep working on a bare Python install. ``numpy`` is not imported anywhere here.

Every function answers a question about *observed* data. None of them claims to
improve the odds of the next draw, and the module exists partly to make that
claim checkable: :func:`chi_square_uniform` tests whether the historical ball
distribution deviates from uniform by more than sampling noise would explain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .core import Draw

RED_NUMBERS = 33
BLUE_NUMBERS = 16
RED_PICK = 6

#: Two-sided 95% normal quantile, used by :func:`wilson_interval`.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the textbook ``p ± z*sqrt(p(1-p)/n)`` because the normal
    approximation degrades badly near 0 and 1, and a single blue ball sits at
    p = 1/16, which is close enough to matter for short samples.
    """
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class FrequencyRow:
    """Observed vs theoretical appearance rate for one ball."""

    number: int
    count: int
    trials: int
    expected_rate: float

    @property
    def rate(self) -> float:
        return self.count / self.trials if self.trials else 0.0

    @property
    def deviation(self) -> float:
        return self.rate - self.expected_rate

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.count, self.trials)

    @property
    def deviates_significantly(self) -> bool:
        """Whether the theoretical rate falls outside the 95% interval.

        Even ``True`` only means "this ball moved more than a single-ball 95%
        interval allows". Thirty-three balls are tested simultaneously, so one or
        two trips of this flag are expected by chance alone; it is not evidence
        that a number is hot in any predictive sense.
        """
        low, high = self.interval
        return not (low <= self.expected_rate <= high)


@dataclass(frozen=True)
class ChiSquareResult:
    """Goodness-of-fit test of observed ball counts against a uniform model."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    expected_count: float
    samples: int

    @property
    def uniform_plausible(self) -> bool:
        """``True`` when the data are consistent with a fair, uniform draw."""
        return self.p_value >= 0.05


def red_frequency(draws: Sequence[Draw]) -> list[FrequencyRow]:
    """Appearance count and rate for each of the 33 red balls."""
    counts = {number: 0 for number in range(1, RED_NUMBERS + 1)}
    for draw in draws:
        for number in draw.reds:
            counts[number] += 1
    trials = len(draws)
    expected = RED_PICK / RED_NUMBERS
    return [FrequencyRow(number, counts[number], trials, expected) for number in sorted(counts)]


def blue_frequency(draws: Sequence[Draw]) -> list[FrequencyRow]:
    """Appearance count and rate for each of the 16 blue balls."""
    counts = {number: 0 for number in range(1, BLUE_NUMBERS + 1)}
    for draw in draws:
        counts[draw.blue] += 1
    trials = len(draws)
    expected = 1.0 / BLUE_NUMBERS
    return [FrequencyRow(number, counts[number], trials, expected) for number in sorted(counts)]


def _regularized_gamma_q(shape: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(shape, x), i.e. the chi-square tail.

    Implemented here (series for x < a+1, continued fraction otherwise, following
    the usual Numerical Recipes arrangement) so the module does not need scipy.
    """
    if x <= 0.0:
        return 1.0
    if x < shape + 1.0:
        return 1.0 - _gamma_series(shape, x)
    return _gamma_continued_fraction(shape, x)


def _gamma_series(shape: float, x: float) -> float:
    iterations, epsilon = 500, 1e-14
    term = total = 1.0 / shape
    current = shape
    for _ in range(iterations):
        current += 1.0
        term *= x / current
        total += term
        if abs(term) < abs(total) * epsilon:
            break
    return total * math.exp(-x + shape * math.log(x) - math.lgamma(shape))


def _gamma_continued_fraction(shape: float, x: float) -> float:
    iterations, epsilon, tiny = 500, 1e-14, 1e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, iterations + 1):
        an = -index * (index - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return math.exp(-x + shape * math.log(x) - math.lgamma(shape)) * h


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """P(X >= statistic) for a chi-square distribution with the given dof."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if statistic <= 0.0:
        return 1.0
    return _regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)


def uniformity_test(counts: Iterable[int], expected_count: float, samples: int) -> ChiSquareResult:
    """Chi-square goodness of fit of ``counts`` against ``expected_count`` each."""
    observed = list(counts)
    if not observed:
        raise ValueError("counts must not be empty")
    if expected_count <= 0.0:
        raise ValueError("expected_count must be positive")
    statistic = sum((count - expected_count) ** 2 / expected_count for count in observed)
    dof = len(observed) - 1
    return ChiSquareResult(statistic, dof, chi_square_survival(statistic, dof), expected_count, samples)


def red_uniformity(draws: Sequence[Draw]) -> ChiSquareResult:
    """Test whether red balls appear at the uniform rate of 6/33 per period."""
    rows = red_frequency(draws)
    return uniformity_test((row.count for row in rows), len(draws) * RED_PICK / RED_NUMBERS, len(draws))


def blue_uniformity(draws: Sequence[Draw]) -> ChiSquareResult:
    """Test whether blue balls appear at the uniform rate of 1/16 per period."""
    rows = blue_frequency(draws)
    return uniformity_test((row.count for row in rows), len(draws) / BLUE_NUMBERS, len(draws))


@dataclass(frozen=True)
class PrizeOdds:
    """Theoretical probability of one prize level for a single ticket."""

    level: str
    combinations: int
    probability: float
    one_in: float


def _red_combinations(hits: int) -> int:
    """Number of red selections producing exactly ``hits`` matches."""
    return math.comb(RED_PICK, hits) * math.comb(RED_NUMBERS - RED_PICK, RED_PICK - hits)


def prize_odds() -> list[PrizeOdds]:
    """Theoretical odds per prize level, computed rather than hard-coded.

    A single ticket costs 2 CNY and is one of ``C(33,6) * 16 = 17,721,088``
    equally likely outcomes. These numbers do not change with the ticket you
    pick, which is the whole point of listing them next to any "prediction".
    """
    total_red = math.comb(RED_NUMBERS, RED_PICK)
    total = total_red * BLUE_NUMBERS
    blue_wrong = BLUE_NUMBERS - 1

    # Each prize level is the count of distinct tickets that land on it. The
    # lower levels combine two shapes (n red + blue, or n red without blue).
    table = (
        ("一等奖", 1),
        ("二等奖", blue_wrong),
        ("三等奖", _red_combinations(5)),
        ("四等奖", _red_combinations(5) * blue_wrong + _red_combinations(4)),
        ("五等奖", _red_combinations(4) * blue_wrong + _red_combinations(3)),
        ("六等奖", _red_combinations(0) + _red_combinations(1) + _red_combinations(2)),
    )
    return [PrizeOdds(level, combos, combos / total, total / combos) for level, combos in table]


def top_numbers(rows: Sequence[FrequencyRow], count: int = 5) -> list[FrequencyRow]:
    """The ``count`` most frequent balls, ties broken by the lower number.

    "Most frequent" is a description of the past, not a forecast: check
    :attr:`FrequencyRow.deviates_significantly` before reading anything into it.
    """
    return sorted(rows, key=lambda row: (-row.count, row.number))[:count]


def compare_proportions(first_hits: int, first_trials: int, second_hits: int, second_trials: int) -> float:
    """Two-sided p-value for the difference between two hit rates.

    Used to check whether an algorithm's apparent edge over the random baseline
    is larger than binomial noise. Returns 1.0 when there is no variance to test.
    """
    if first_trials <= 0 or second_trials <= 0:
        return 1.0
    pooled = (first_hits + second_hits) / (first_trials + second_trials)
    if pooled <= 0.0 or pooled >= 1.0:
        return 1.0
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / first_trials + 1.0 / second_trials))
    if standard_error == 0.0:
        return 1.0
    z_score = (first_hits / first_trials - second_hits / second_trials) / standard_error
    return math.erfc(abs(z_score) / math.sqrt(2.0))


@dataclass(frozen=True)
class ProbabilityReport:
    """Everything the probability panel needs, in one frozen container."""

    samples: int
    first_issue: str
    last_issue: str
    red_rows: tuple[FrequencyRow, ...]
    blue_rows: tuple[FrequencyRow, ...]
    red_uniformity: ChiSquareResult
    blue_uniformity: ChiSquareResult
    red_top: tuple[FrequencyRow, ...]
    blue_top: tuple[FrequencyRow, ...]
    prizes: tuple[PrizeOdds, ...]

    @property
    def jackpot(self) -> PrizeOdds:
        return self.prizes[0]

    @property
    def has_detectable_pattern(self) -> bool:
        """``True`` only if *both* ball pools deviate from uniform."""
        return not (self.red_uniformity.uniform_plausible and self.blue_uniformity.uniform_plausible)


def build_probability_report(draws: Sequence[Draw], top_count: int = 5) -> ProbabilityReport:
    """Assemble the full probability analysis for a historical sample."""
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if not ordered:
        raise ValueError("draws must not be empty")
    red_rows = red_frequency(ordered)
    blue_rows = blue_frequency(ordered)
    return ProbabilityReport(
        samples=len(ordered),
        first_issue=ordered[0].issue,
        last_issue=ordered[-1].issue,
        red_rows=tuple(red_rows),
        blue_rows=tuple(blue_rows),
        red_uniformity=red_uniformity(ordered),
        blue_uniformity=blue_uniformity(ordered),
        red_top=tuple(top_numbers(red_rows, top_count)),
        blue_top=tuple(top_numbers(blue_rows, top_count)),
        prizes=tuple(prize_odds()),
    )
