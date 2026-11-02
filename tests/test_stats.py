"""Tests for the dependency-free statistics module.

These run on a bare Python install: nothing here imports numpy, matplotlib or
statsmodels, and the whole file is fast enough to be part of the default suite.
"""

from __future__ import annotations

import math

import pytest

from ssq_research.core import Draw
from ssq_research.stats import (
    BLUE_NUMBERS,
    RED_NUMBERS,
    blue_frequency,
    blue_uniformity,
    build_probability_report,
    chi_square_survival,
    compare_proportions,
    prize_odds,
    red_frequency,
    red_uniformity,
    top_numbers,
    wilson_interval,
)


def synthetic_draws(count: int, blue_cycle: bool = True) -> list[Draw]:
    """Draws with a fixed red set and a cycling blue ball."""
    return [
        Draw(
            f"2026{index + 1:03d}",
            f"2026-01-{(index % 28) + 1:02d}",
            (1, 2, 3, 4, 5, 6),
            (index % BLUE_NUMBERS) + 1 if blue_cycle else 1,
        )
        for index in range(count)
    ]


def test_wilson_interval_brackets_the_observed_rate():
    low, high = wilson_interval(10, 100)

    assert low < 0.10 < high
    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    for successes, trials in ((0, 50), (50, 50), (0, 1), (1, 1)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_without_trials_is_the_whole_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_red_frequency_counts_every_appearance_and_uses_the_uniform_rate():
    rows = red_frequency(synthetic_draws(30))

    assert len(rows) == RED_NUMBERS
    # Every draw uses the same six reds, so those six appear 30 times each.
    assert [row.count for row in rows[:6]] == [30] * 6
    assert all(row.count == 0 for row in rows[6:])
    assert rows[0].expected_rate == pytest.approx(6 / 33)
    assert rows[0].rate == pytest.approx(1.0)


def test_blue_frequency_uses_one_sixteenth_as_the_uniform_rate():
    rows = blue_frequency(synthetic_draws(32))

    assert len(rows) == BLUE_NUMBERS
    assert rows[0].expected_rate == pytest.approx(1 / 16)
    assert sum(row.count for row in rows) == 32


def test_extreme_red_bias_is_flagged_as_a_real_deviation():
    result = red_uniformity(synthetic_draws(60))

    assert result.degrees_of_freedom == RED_NUMBERS - 1
    assert result.p_value < 0.001
    assert result.uniform_plausible is False


def test_a_perfectly_cycling_blue_pool_looks_uniform():
    result = blue_uniformity(synthetic_draws(160, blue_cycle=True))

    assert result.p_value > 0.05
    assert result.uniform_plausible is True


def test_chi_square_survival_matches_reference_values():
    # Reference points for the regularized upper incomplete gamma.
    assert chi_square_survival(0.0, 5) == 1.0
    assert chi_square_survival(32.0, 32) == pytest.approx(0.4667, abs=5e-4)
    assert chi_square_survival(50.0, 32) == pytest.approx(0.0223, abs=5e-4)
    assert chi_square_survival(100.0, 10) < 1e-15


def test_chi_square_survival_rejects_a_non_positive_degrees_of_freedom():
    with pytest.raises(ValueError):
        chi_square_survival(1.0, 0)


def test_prize_odds_reproduce_the_published_lottery_figures():
    odds = {item.level: item for item in prize_odds()}

    # These six numbers are the officially published per-ticket odds; getting
    # them wrong would make every "probability" claim in the report wrong.
    assert odds["一等奖"].one_in == pytest.approx(17_721_088.0, rel=1e-9)
    assert odds["二等奖"].one_in == pytest.approx(1_181_405.9, rel=1e-5)
    assert odds["三等奖"].one_in == pytest.approx(109_389.4, rel=1e-5)
    assert odds["四等奖"].one_in == pytest.approx(2_302.9, rel=1e-4)
    assert odds["五等奖"].one_in == pytest.approx(128.9, rel=1e-3)
    assert odds["六等奖"].probability == pytest.approx(0.058893, rel=1e-4)


def test_prize_odds_sum_to_the_whole_probability_of_winning_something():
    odds = prize_odds()
    total = sum(item.probability for item in odds)

    # Roughly one ticket in 15.7 wins any prize at all.
    assert 0.06 < total < 0.07
    assert all(item.probability > 0 for item in odds)


def test_top_numbers_sorts_by_count_then_by_the_lower_number():
    draws = [
        Draw("1", "2026-01-01", (7, 8, 9, 10, 11, 12), 1),
        Draw("2", "2026-01-02", (7, 8, 9, 10, 11, 13), 2),
        Draw("3", "2026-01-03", (7, 8, 9, 10, 14, 15), 3),
    ]
    top = top_numbers(red_frequency(draws), 3)

    assert [row.number for row in top] == [7, 8, 9]


def test_compare_proportions_returns_one_when_there_is_nothing_to_compare():
    assert compare_proportions(0, 0, 5, 10) == 1.0
    assert compare_proportions(0, 10, 0, 10) == 1.0


def test_compare_proportions_separates_a_large_gap_from_a_small_one():
    small = compare_proportions(10, 100, 12, 100)
    large = compare_proportions(10, 100, 40, 100)

    assert small > 0.05
    assert large < 0.001


def test_report_exposes_the_jackpot_first_and_keeps_the_top_five():
    report = build_probability_report(synthetic_draws(30))

    assert report.jackpot.level == "一等奖"
    assert len(report.red_top) == 5
    assert len(report.blue_top) == 5
    assert report.samples == 30
    assert report.has_detectable_pattern is True  # the red pool is deliberately skewed


def test_report_on_a_fair_sample_does_not_claim_a_pattern():
    draws = [
        Draw(f"{index}", f"2026-01-{(index % 28) + 1:02d}", tuple(sorted({(index % 33) + 1, (index + 1) % 33 + 1, (index + 2) % 33 + 1, (index + 3) % 33 + 1, (index + 4) % 33 + 1, (index + 5) % 33 + 1})), (index % 16) + 1)
        for index in range(400)
    ]
    if any(len(draw.reds) != 6 for draw in draws):
        pytest.skip("generator produced a duplicate red set")
    report = build_probability_report(draws)

    assert report.red_uniformity.uniform_plausible is True
    assert report.has_detectable_pattern is False
    assert math.isfinite(report.red_uniformity.statistic)
