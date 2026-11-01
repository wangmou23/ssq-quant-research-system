from __future__ import annotations

import numpy as np
import pytest

from ssq_research.core import Pick
from ssq_research.lstm import (
    BLUE_NUMBERS,
    FEATURE_COUNT,
    RED_NUMBERS,
    combine_heads,
    pick_argmax,
    pick_from_probabilities,
    pick_sampled,
    split_probabilities,
)


def uniform_probabilities() -> np.ndarray:
    return np.full(FEATURE_COUNT, 1.0 / FEATURE_COUNT, dtype="float32")


def peaked_probabilities() -> np.ndarray:
    """Reds 1/3/5/7/9/11 and blue 5 hold all the mass."""
    probabilities = np.zeros(FEATURE_COUNT, dtype="float32")
    probabilities[[0, 2, 4, 6, 8, 10]] = 1.0
    probabilities[RED_NUMBERS + 4] = 1.0
    return probabilities


def assert_legal(pick: Pick) -> None:
    # Pick.__post_init__ already validates; assert the contract explicitly too.
    assert len(pick.reds) == 6
    assert len(set(pick.reds)) == 6
    assert all(1 <= number <= 33 for number in pick.reds)
    assert 1 <= pick.blue <= 16


def test_split_and_combine_roundtrip():
    probabilities = np.linspace(0.0, 1.0, FEATURE_COUNT, dtype="float32")
    red_scores, blue_scores = split_probabilities(probabilities)
    assert red_scores.size == RED_NUMBERS
    assert blue_scores.size == BLUE_NUMBERS
    restored = combine_heads(np.expand_dims(red_scores, 0), np.expand_dims(blue_scores, 0))
    assert restored.shape == (1, FEATURE_COUNT)
    assert np.allclose(restored[0], probabilities)


def test_split_rejects_wrong_width():
    with pytest.raises(ValueError):
        split_probabilities(np.zeros(FEATURE_COUNT - 1))


def test_pick_argmax_is_deterministic():
    probabilities = uniform_probabilities()
    picks = {pick_argmax(probabilities) for _ in range(20)}
    assert len(picks) == 1

    peaked = pick_argmax(peaked_probabilities())
    assert peaked.reds == (1, 3, 5, 7, 9, 11)
    assert peaked.blue == 5
    assert_legal(peaked)


def test_pick_argmax_ignores_the_probability_scale():
    # A flat distribution still has a well defined argmax; that is exactly why a
    # collapsed model looks like a stable prediction.
    assert pick_argmax(uniform_probabilities()) == pick_argmax(uniform_probabilities() * 100.0)


def test_pick_sampled_same_seed_is_reproducible():
    probabilities = uniform_probabilities()
    first = [pick_sampled(probabilities, np.random.default_rng(7)) for _ in range(20)]
    second = [pick_sampled(probabilities, np.random.default_rng(7)) for _ in range(20)]
    assert first == second
    for pick in first:
        assert_legal(pick)


def test_pick_sampled_different_seeds_differ():
    probabilities = uniform_probabilities()
    picks = {pick_sampled(probabilities, np.random.default_rng(seed)) for seed in range(40)}
    assert len(picks) > 20


def test_pick_from_probabilities_actually_uses_the_seed():
    probabilities = uniform_probabilities()
    assert pick_from_probabilities(probabilities, seed=11) == pick_from_probabilities(probabilities, seed=11)
    picks = {pick_from_probabilities(probabilities, seed=seed) for seed in range(40)}
    # 0.1.0 ignored the seed and returned the argmax every time; this must fail there.
    assert len(picks) > 20


def test_pick_sampled_respects_a_peaked_distribution():
    probabilities = np.zeros(FEATURE_COUNT, dtype="float32")
    probabilities[RED_NUMBERS + 2] = 0.85
    other_blues = [index for index in range(BLUE_NUMBERS) if index != 2]
    probabilities[RED_NUMBERS + np.array(other_blues, dtype="int64")] = 0.01
    rng = np.random.default_rng(0)
    blues = [pick_sampled(probabilities, rng).blue for _ in range(200)]
    assert blues.count(3) / len(blues) > 0.6


def test_pick_sampled_handles_all_zero_probabilities():
    probabilities = np.zeros(FEATURE_COUNT, dtype="float32")
    for seed in range(10):
        assert_legal(pick_sampled(probabilities, np.random.default_rng(seed)))


def test_pick_sampled_handles_negative_nan_and_inf():
    for filler in (np.nan, np.inf, -5.0):
        probabilities = uniform_probabilities()
        probabilities[0] = filler
        probabilities[1] = filler
        probabilities[RED_NUMBERS] = filler
        pick = pick_sampled(probabilities, np.random.default_rng(3))
        assert_legal(pick)


def test_pick_sampled_handles_insufficient_support():
    # Only two reds carry mass; numpy would raise "Fewer non-zero entries in p
    # than size" without the uniform fallback.
    probabilities = np.zeros(FEATURE_COUNT, dtype="float32")
    probabilities[0] = 0.5
    probabilities[1] = 0.5
    probabilities[RED_NUMBERS] = 1.0
    for seed in range(20):
        pick = pick_sampled(probabilities, np.random.default_rng(seed))
        assert_legal(pick)
        assert pick.blue == 1


def test_pick_sampled_rejects_wrong_width():
    with pytest.raises(ValueError):
        pick_sampled(np.zeros(10), np.random.default_rng(0))
