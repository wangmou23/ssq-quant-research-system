"""Tests for the red/blue head layout of the 49-slot vector.

None of these need TensorFlow: they pin the single most error-prone seam of the
dual-head refactor, namely the boundary between red indices ``0:33`` and blue
indices ``33:49``. If the encoder, the target builder, the splitter and the
combiner ever disagree about that boundary, a model can train happily and still
predict the wrong slot, so each of them is checked against :func:`encode_draw`.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssq_research.core import Draw
from ssq_research.lstm import (
    BLUE_NUMBERS,
    BLUE_SLICE,
    FEATURE_COUNT,
    RED_NUMBERS,
    RED_SLICE,
    LstmConfig,
    _split_head_outputs,
    combine_heads,
    encode_draw,
    make_sequences,
    normalize_weights,
    pack_targets,
    split_probabilities,
)


def synthetic_draws(count: int) -> list[Draw]:
    """Deterministic, valid draws; blue cycles so every blue class shows up."""
    return [
        Draw(
            issue=str(2026001 + index),
            draw_date=f"2026-01-{index + 1:02d}",
            reds=tuple(range(1 + index % 20, 7 + index % 20)),
            blue=1 + index % BLUE_NUMBERS,
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# encode_draw: the reference layout every other function must agree with
# --------------------------------------------------------------------------- #


def test_encode_draw_sets_exactly_seven_slots():
    vector = encode_draw(Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1))
    assert vector.shape == (FEATURE_COUNT,)
    assert float(vector.sum()) == 7.0
    assert set(np.unique(vector)) <= {0.0, 1.0}


def test_encode_draw_puts_reds_in_the_red_block_and_blue_after_it():
    vector = encode_draw(Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1))
    assert vector[:RED_NUMBERS].sum() == 6.0
    assert vector[BLUE_SLICE].sum() == 1.0
    assert vector[RED_NUMBERS] == 1.0  # blue 1 maps to the first blue slot
    assert vector[FEATURE_COUNT - 1] == 0.0


@pytest.mark.parametrize("red_number", [1, 33])
def test_encode_draw_handles_red_boundaries(red_number):
    reds = tuple(sorted({red_number, 2, 3, 4, 5, 6}))
    vector = encode_draw(Draw("1", "2026-01-01", reds, 1))
    assert vector[red_number - 1] == 1.0
    assert vector[RED_NUMBERS:].sum() == 1.0  # red 33 must not bleed into blue


@pytest.mark.parametrize("blue", [1, 16])
def test_encode_draw_handles_blue_boundaries(blue):
    vector = encode_draw(Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), blue))
    assert vector[RED_NUMBERS + blue - 1] == 1.0
    assert vector[BLUE_SLICE].sum() == 1.0


# --------------------------------------------------------------------------- #
# make_sequences: legacy contract plus the dual-head targets
# --------------------------------------------------------------------------- #


def test_make_sequences_keeps_the_legacy_two_tuple_contract():
    x, y = make_sequences(synthetic_draws(10), 4)
    assert x.shape == (6, 4, FEATURE_COUNT)
    assert y.shape == (6, FEATURE_COUNT)


def test_make_sequences_heads_split_the_legacy_targets():
    draws = synthetic_draws(10)
    x_heads, y_red, y_blue = make_sequences(draws, 4, return_heads=True)
    x_legacy, y_legacy = make_sequences(draws, 4)

    assert x_heads.shape == x_legacy.shape
    assert y_red.shape == (6, RED_NUMBERS)
    assert y_blue.shape == (6,)
    np.testing.assert_allclose(y_red, y_legacy[:, RED_SLICE])
    # Multi-hot reds, so every row must carry exactly six ones.
    assert set(y_red.sum(axis=1).tolist()) == {6.0}


def test_make_sequences_blue_labels_index_the_same_slot_as_the_encoder():
    draws = synthetic_draws(10)
    window = 4
    _x, _y_red, y_blue = make_sequences(draws, window, return_heads=True)

    assert y_blue.dtype == np.int32
    assert int(y_blue.min()) >= 0
    assert int(y_blue.max()) <= BLUE_NUMBERS - 1
    # y_blue[k] must be the blue index of draws[window + k], computed with the
    # encoder's own layout rather than a second, hand-written offset.
    for offset, label in enumerate(y_blue.tolist()):
        encoded = encode_draw(draws[window + offset])
        assert int(label) == int(np.argmax(encoded[BLUE_SLICE]))
        assert int(label) + 1 == draws[window + offset].blue


@pytest.mark.parametrize("window", [0, -1, 10, 11])
def test_make_sequences_rejects_unusable_windows(window):
    with pytest.raises(ValueError):
        make_sequences(synthetic_draws(10), window)


# --------------------------------------------------------------------------- #
# pack_targets: the inverse of make_sequences(return_heads=True)
# --------------------------------------------------------------------------- #


def test_pack_targets_rebuilds_the_legacy_multi_hot_matrix():
    draws = synthetic_draws(10)
    _x, y_red, y_blue = make_sequences(draws, 4, return_heads=True)
    _x_legacy, y_legacy = make_sequences(draws, 4)

    packed = pack_targets(y_red, y_blue)
    assert packed.shape == y_legacy.shape
    np.testing.assert_allclose(packed, y_legacy)


def test_pack_targets_handles_the_first_and_last_blue_class():
    y_red = np.zeros((2, RED_NUMBERS), dtype="float32")
    packed = pack_targets(y_red, np.array([0, BLUE_NUMBERS - 1], dtype="int32"))
    assert packed.shape == (2, FEATURE_COUNT)
    assert packed[0, RED_NUMBERS] == 1.0
    assert packed[1, FEATURE_COUNT - 1] == 1.0
    assert packed.sum() == 2.0


# --------------------------------------------------------------------------- #
# split_probabilities / combine_heads
# --------------------------------------------------------------------------- #


def test_split_and_combine_are_exact_inverses():
    rng = np.random.default_rng(0)
    probabilities = rng.random(FEATURE_COUNT)
    red_scores, blue_scores = split_probabilities(probabilities)

    assert red_scores.shape == (RED_NUMBERS,)
    assert blue_scores.shape == (BLUE_NUMBERS,)
    restored = combine_heads(red_scores.reshape(1, -1), blue_scores.reshape(1, -1))
    np.testing.assert_allclose(restored[0], probabilities)


def test_split_probabilities_requires_exactly_one_49_slot_vector():
    with pytest.raises(ValueError):
        split_probabilities(np.zeros(FEATURE_COUNT - 1))
    # A batch is rejected too: predict_probabilities, not this function, owns
    # the batching contract, so a silent flatten here would hide shape bugs.
    with pytest.raises(ValueError):
        split_probabilities(np.zeros((3, FEATURE_COUNT)))


def test_combine_heads_rejects_mismatched_batch_sizes():
    with pytest.raises(ValueError):
        combine_heads(np.zeros((2, RED_NUMBERS)), np.zeros((3, BLUE_NUMBERS)))


def test_combine_heads_accepts_flat_head_outputs():
    combined = combine_heads(np.zeros(RED_NUMBERS), np.zeros(BLUE_NUMBERS))
    assert combined.shape == (1, FEATURE_COUNT)


# --------------------------------------------------------------------------- #
# _split_head_outputs: the compatibility seam for predict_probabilities
# --------------------------------------------------------------------------- #


def test_split_head_outputs_accepts_a_named_dict():
    raw = {"red_head": np.zeros((2, RED_NUMBERS)), "blue_head": np.zeros((2, BLUE_NUMBERS))}
    red_scores, blue_scores = _split_head_outputs(raw)
    assert red_scores.shape == (2, RED_NUMBERS)
    assert blue_scores.shape == (2, BLUE_NUMBERS)


def test_split_head_outputs_accepts_a_two_element_list():
    raw = [np.ones((1, RED_NUMBERS)), np.ones((1, BLUE_NUMBERS))]
    red_scores, blue_scores = _split_head_outputs(raw)
    assert red_scores.shape == (1, RED_NUMBERS)
    assert blue_scores.shape == (1, BLUE_NUMBERS)


def test_split_head_outputs_still_reads_a_legacy_single_head_array():
    legacy = np.arange(2 * FEATURE_COUNT, dtype="float32").reshape(2, FEATURE_COUNT)
    red_scores, blue_scores = _split_head_outputs(legacy)
    np.testing.assert_allclose(red_scores, legacy[:, :RED_NUMBERS])
    np.testing.assert_allclose(blue_scores, legacy[:, RED_NUMBERS:])


@pytest.mark.parametrize(
    "raw",
    [
        [np.zeros((1, RED_NUMBERS))],  # only one head
        [np.zeros((1, RED_NUMBERS)), np.zeros((1, BLUE_NUMBERS)), np.zeros((1, BLUE_NUMBERS))],
        np.zeros(FEATURE_COUNT),  # unbatched 1-D
        np.zeros((2, 10)),  # wrong width
    ],
)
def test_split_head_outputs_rejects_unexpected_shapes(raw):
    with pytest.raises(ValueError):
        _split_head_outputs(raw)


# --------------------------------------------------------------------------- #
# normalize_weights: the sampler's input contract
# --------------------------------------------------------------------------- #


def test_normalize_weights_sums_to_one():
    weights = normalize_weights(np.array([1.0, 1.0, 2.0, 6.0]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[3] == pytest.approx(0.6)


def test_normalize_weights_renormalises_scores_that_do_not_sum_to_one():
    # A softmax head sums to 1 already, a sigmoid head does not; both must work.
    assert normalize_weights(np.full(BLUE_NUMBERS, 1.0 / BLUE_NUMBERS)).sum() == pytest.approx(1.0)
    assert normalize_weights(np.full(BLUE_NUMBERS, 2.0 / BLUE_NUMBERS)).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, -5.0])
def test_normalize_weights_treats_non_finite_and_negative_scores_as_zero(bad_value):
    scores = np.ones(BLUE_NUMBERS)
    scores[0] = bad_value
    weights = normalize_weights(scores)
    assert weights[0] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_normalize_weights_falls_back_to_uniform_for_a_dead_distribution():
    weights = normalize_weights(np.zeros(BLUE_NUMBERS))
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, np.full(BLUE_NUMBERS, 1.0 / BLUE_NUMBERS))


def test_normalize_weights_rejects_an_empty_vector():
    with pytest.raises(ValueError):
        normalize_weights(np.array([]))


# --------------------------------------------------------------------------- #
# LstmConfig validation
# --------------------------------------------------------------------------- #


def test_lstm_config_defaults_expose_the_reproducibility_knobs():
    config = LstmConfig()
    assert isinstance(config.seed, int)
    assert config.deterministic is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window", 0),
        ("hidden_units", 0),
        ("layers", 0),
        ("dropout", 1.0),
        ("dropout", -0.1),
        ("epochs", -1),
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("retrain_every", 0),
        ("max_predictions", 0),
    ],
)
def test_lstm_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        LstmConfig(**{field: value})
