"""Tests for the multi-algorithm scoring layer.

Skipped wholesale when numpy is unavailable, since every scorer is numpy-based.
TensorFlow and statsmodels are *not* required: the LSTM and ARIMA scorers are
exercised only through their import guards.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ssq_research.core import Draw, Pick  # noqa: E402
from ssq_research.scores import (  # noqa: E402
    FEATURE_COUNT,
    BLUE_NUMBERS,
    RED_NUMBERS,
    ArimaScorer,
    EnsembleScorer,
    FrequencyScorer,
    MarkovScorer,
    ScorerStrategy,
    available_algorithms,
    build_ensemble,
    build_scorers,
    normalize,
    pick_from_scores,
    top_indices,
)


def make_draws(count: int, reds=(1, 2, 3, 4, 5, 6), blue_sequence=None) -> list[Draw]:
    draws = []
    for index in range(count):
        blue = blue_sequence[index % len(blue_sequence)] if blue_sequence else 1
        draws.append(Draw(f"{index:04d}", f"2026-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}", tuple(reds), blue))
    return draws


def test_normalize_returns_a_distribution():
    result = normalize([1.0, 2.0, 3.0])

    assert result.shape == (3,)
    assert float(result.sum()) == pytest.approx(1.0)
    assert result[2] > result[1] > result[0]


def test_normalize_treats_nan_and_negatives_as_zero():
    result = normalize([float("nan"), -5.0, 4.0, float("inf")])

    assert float(result.sum()) == pytest.approx(1.0)
    assert result[2] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_normalize_falls_back_to_uniform_when_nothing_is_left():
    result = normalize([0.0, 0.0, 0.0, 0.0])

    assert float(result.sum()) == pytest.approx(1.0)
    assert np.allclose(result, 0.25)


def test_frequency_scorer_concentrates_on_the_balls_that_appear():
    history = make_draws(10, reds=(1, 2, 3, 4, 5, 6))
    scores = FrequencyScorer().score(history)

    assert scores.shape == (FEATURE_COUNT,)
    assert float(scores.sum()) == pytest.approx(1.0)
    assert scores[0] > 0
    assert scores[10] == pytest.approx(0.0)


def test_frequency_scorer_without_history_is_uniform():
    scores = FrequencyScorer().score([])

    assert np.allclose(scores, 1.0 / FEATURE_COUNT)


def test_recency_weighting_moves_mass_towards_recent_draws():
    history = make_draws(20, reds=(1, 2, 3, 4, 5, 6)) + make_draws(1, reds=(11, 12, 13, 14, 15, 16))
    plain = FrequencyScorer().score(history)
    recent = FrequencyScorer(name="recent", half_life=2.0).score(history)

    assert recent[10] > plain[10]


def test_markov_scorer_learns_a_deterministic_blue_transition():
    # Blue alternates 1, 2, 1, 2 ... so after 1 the next blue is always 2.
    history = make_draws(11, blue_sequence=[1, 2])
    scores = MarkovScorer().score(history)
    blue_scores = scores[RED_NUMBERS:]

    assert int(np.argmax(blue_scores)) + 1 == 2


def test_markov_scorer_needs_at_least_two_draws():
    scores = MarkovScorer().score(make_draws(1))

    assert np.allclose(scores, 1.0 / FEATURE_COUNT)


def test_markov_scorer_rejects_non_positive_smoothing():
    with pytest.raises(ValueError):
        MarkovScorer(smoothing=0.0).score(make_draws(5))


def test_ensemble_averages_its_components_and_stays_normalized():
    history = make_draws(12, blue_sequence=[1, 2, 3])
    ensemble = build_ensemble([FrequencyScorer(), MarkovScorer()])
    scores = ensemble.score(history)

    assert scores.shape == (FEATURE_COUNT,)
    assert float(scores.sum()) == pytest.approx(1.0)


def test_ensemble_rejects_mismatched_or_negative_weights():
    with pytest.raises(ValueError):
        EnsembleScorer(components=[FrequencyScorer()], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        EnsembleScorer(components=[FrequencyScorer()], weights=[-1.0])
    with pytest.raises(ValueError):
        EnsembleScorer(components=[])


def test_top_indices_returns_the_highest_scores_first():
    indices = top_indices(np.array([0.1, 0.9, 0.5, 0.3]), 2)

    assert list(indices) == [1, 2]


def test_top_indices_rejects_an_oversized_request():
    with pytest.raises(ValueError):
        top_indices(np.zeros(4), 5)


def test_argmax_pick_takes_the_highest_scores_and_is_deterministic():
    scores = np.zeros(FEATURE_COUNT)
    for index in range(6):
        scores[index] = 0.1
    scores[RED_NUMBERS + 3] = 0.4
    rng = np.random.default_rng(0)

    first = pick_from_scores(scores, rng, "argmax")
    second = pick_from_scores(scores, np.random.default_rng(99), "argmax")

    assert isinstance(first, Pick)
    assert first == second
    assert first.reds == (1, 2, 3, 4, 5, 6)
    assert first.blue == 4


def test_sampled_picks_are_legal_and_vary_with_the_seed():
    scores = np.ones(FEATURE_COUNT) / FEATURE_COUNT
    picks = {pick_from_scores(scores, np.random.default_rng(seed), "sample") for seed in range(40)}

    assert len(picks) > 1
    for pick in picks:
        assert len(pick.reds) == 6
        assert len(set(pick.reds)) == 6
        assert all(1 <= number <= 33 for number in pick.reds)
        assert 1 <= pick.blue <= BLUE_NUMBERS


def test_sampled_picks_stay_legal_when_the_distribution_is_degenerate():
    scores = np.zeros(FEATURE_COUNT)
    scores[0] = 1.0
    picks = [pick_from_scores(scores, np.random.default_rng(seed), "sample") for seed in range(10)]

    for pick in picks:
        assert len(set(pick.reds)) == 6
        assert 1 <= pick.blue <= BLUE_NUMBERS


def test_pick_from_scores_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        pick_from_scores(np.ones(FEATURE_COUNT), np.random.default_rng(0), "median")


def test_strategy_name_carries_the_mode_so_rows_never_mix():
    strategy = ScorerStrategy(FrequencyScorer(), "argmax")

    assert strategy.name == "frequency_argmax"
    with pytest.raises(ValueError):
        ScorerStrategy(FrequencyScorer(), "median")


def test_strategy_satisfies_the_core_strategy_protocol():
    strategy = ScorerStrategy(FrequencyScorer(), "sample")
    pick = strategy.pick(tuple(make_draws(10)), np.random.default_rng(3))

    assert strategy.name == "frequency_sample"
    assert isinstance(pick, Pick)


def test_build_scorers_ships_the_expected_dependency_free_set():
    scorers = build_scorers()

    assert set(scorers) == {"frequency", "markov", "frequency_recent"}
    assert "frequency_recent" not in build_scorers(include_recency=False)


def test_available_algorithms_reports_what_can_actually_be_imported():
    status = available_algorithms()

    assert status["frequency"] is True
    assert status["markov"] is True
    assert status["ensemble"] is True
    for key in ("arima", "lstm"):
        assert isinstance(status[key], bool)


def test_arima_scorer_without_statsmodels_raises_an_actionable_error():
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="statsmodels"):
            ArimaScorer().score(make_draws(20))
    else:
        pytest.skip("statsmodels is installed; the import guard cannot be exercised")
