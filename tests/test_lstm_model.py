"""TensorFlow-backed contract tests for the dual-head LSTM.

The whole point of v0.2.0 was to replace one ``Dense(49, sigmoid)`` head with
``red_head`` (33, sigmoid) + ``blue_head`` (16, softmax). That change had no
test at all: reverting ``softmax`` to ``sigmoid`` left the suite fully green,
which is exactly how a "prediction" collapses back onto a constant blue ball.

The module is skipped when TensorFlow is absent (it is an optional dependency),
so the dependency-free part of the suite keeps working.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "tensorflow",
    reason="TensorFlow is an optional dependency (pip install -e '.[tensorflow]'/.[lstm])",
)

from ssq_research.core import Draw  # noqa: E402
from ssq_research.lstm import (  # noqa: E402
    BLUE_NUMBERS,
    BLUE_SLICE,
    FEATURE_COUNT,
    RED_NUMBERS,
    STRATEGY_ARGMAX,
    STRATEGY_SAMPLE,
    LstmConfig,
    build_model,
    make_sequences,
    predict_probabilities,
    train_model,
    walk_forward_lstm,
)


def synthetic_draws(count: int) -> list[Draw]:
    return [
        Draw(
            issue=str(2026001 + index),
            draw_date=f"2026-01-{index + 1:02d}",
            reds=tuple(range(1 + index % 20, 7 + index % 20)),
            blue=1 + index % BLUE_NUMBERS,
        )
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def window() -> int:
    return 4


@pytest.fixture(scope="module")
def contract_config(window: int) -> LstmConfig:
    return LstmConfig(window=window, hidden_units=4, layers=2, seed=0)


@pytest.fixture(scope="module")
def fast_config(window: int) -> LstmConfig:
    """Small and deterministic: dropout off, one epoch, minimal width."""
    return LstmConfig(
        window=window,
        hidden_units=4,
        layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        max_predictions=2,
        seed=0,
    )


@pytest.fixture(scope="module")
def trained_model(window: int, fast_config: LstmConfig):
    return train_model(synthetic_draws(40), fast_config, validation_split=0.0)


# --------------------------------------------------------------------------- #
# Architecture contract
# --------------------------------------------------------------------------- #


def test_build_model_exposes_two_named_heads_with_the_right_widths(contract_config):
    model = build_model(contract_config)

    assert list(model.output_names) == ["red_head", "blue_head"]
    red_output, blue_output = model.outputs
    assert int(red_output.shape[-1]) == RED_NUMBERS
    assert int(blue_output.shape[-1]) == BLUE_NUMBERS
    assert int(model.input_shape[-1]) == FEATURE_COUNT


def test_blue_head_normalises_with_softmax_and_red_head_stays_sigmoid(contract_config):
    """The single most important assertion in the suite.

    ``softmax`` is what lets the model express "one of 16"; ``sigmoid`` treats
    the 16 blue balls as independent Bernoulli trials whose optimum is the
    historical marginal vector, i.e. a constant. Reverting this one activation
    used to leave every test green.
    """
    model = build_model(contract_config)

    assert model.get_layer("red_head").activation.__name__ == "sigmoid"
    assert model.get_layer("blue_head").activation.__name__ == "softmax"


def test_each_head_is_trained_with_the_loss_its_activation_implies(contract_config):
    model = build_model(contract_config)
    losses = dict(model.loss)

    assert set(losses) == {"red_head", "blue_head"}
    assert losses["red_head"] == "binary_crossentropy"
    assert losses["blue_head"] == "sparse_categorical_crossentropy"


# --------------------------------------------------------------------------- #
# Target contract
# --------------------------------------------------------------------------- #


def test_dual_head_targets_match_what_sparse_categorical_ce_expects(fast_config):
    draws = synthetic_draws(40)
    _x, y_red, y_blue = make_sequences(draws, fast_config.window, return_heads=True)

    assert y_red.shape[1] == RED_NUMBERS
    assert y_blue.shape == (y_red.shape[0],)
    assert y_blue.dtype == np.int32
    assert int(y_blue.min()) >= 0
    assert int(y_blue.max()) <= BLUE_NUMBERS - 1
    assert set(np.unique(y_red).tolist()) <= {0.0, 1.0}


def test_model_can_fit_the_dual_head_targets(trained_model, window):
    """A shape or dtype mismatch here raises inside ``model.fit``."""
    draws = synthetic_draws(20)
    probabilities = predict_probabilities(trained_model, draws[-window:])

    assert probabilities.shape == (FEATURE_COUNT,)
    assert np.all(np.isfinite(probabilities))


# --------------------------------------------------------------------------- #
# predict_probabilities: normalisation proves which head is which
# --------------------------------------------------------------------------- #


def test_only_the_blue_half_is_normalised(trained_model, window):
    draws = synthetic_draws(20)
    probabilities = predict_probabilities(trained_model, draws[-window:])

    red_scores, blue_scores = probabilities[:RED_NUMBERS], probabilities[RED_NUMBERS:]
    # softmax over 16 classes -> sums to exactly 1
    assert float(np.sum(blue_scores)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.min(blue_scores)) > 0.0
    # 33 independent sigmoids -> must NOT be normalised; a sum of 1 here would
    # mean the red head was accidentally turned into a 33-way softmax.
    assert float(np.sum(red_scores)) != pytest.approx(1.0, abs=1e-2)


def test_predict_probabilities_accepts_a_prebuilt_batch(trained_model, window):
    batch = np.stack([np.stack([np.zeros(FEATURE_COUNT) for _ in range(window)]) for _ in range(3)])
    probabilities = predict_probabilities(trained_model, batch)
    assert probabilities.shape == (3, FEATURE_COUNT)
    for row in probabilities:
        assert float(np.sum(row[RED_NUMBERS:])) == pytest.approx(1.0, abs=1e-5)


def test_predict_probabilities_rejects_an_empty_window(trained_model):
    with pytest.raises(ValueError):
        predict_probabilities(trained_model, [])


# --------------------------------------------------------------------------- #
# walk_forward_lstm: two rows per period, reproducible, and honest about argmax
# --------------------------------------------------------------------------- #


def _blues_by_issue(rows, strategy):
    return {row["issue"]: row["pick_blue"] for row in rows if row["strategy"] == strategy}


@pytest.fixture(scope="module")
def wf_draws() -> list[Draw]:
    return synthetic_draws(14)


@pytest.fixture(scope="module")
def rows_seed_11(wf_draws, fast_config):
    return walk_forward_lstm(wf_draws, fast_config, min_history=6, seed=11)


@pytest.fixture(scope="module")
def rows_seed_99(wf_draws, fast_config):
    return walk_forward_lstm(wf_draws, fast_config, min_history=6, seed=99)


def test_every_period_emits_both_the_argmax_and_the_sampled_ticket(rows_seed_11):
    strategies = [row["strategy"] for row in rows_seed_11]
    assert set(strategies) == {STRATEGY_ARGMAX, STRATEGY_SAMPLE}

    issues = [row["issue"] for row in rows_seed_11]
    assert len(issues) == len(set(issues)) * 2
    for issue in set(issues):
        assert issues.count(issue) == 2

    for row in rows_seed_11:
        assert row["seed"] == 11
        assert row["deterministic"] is False
        assert 0.0 <= row["blue_max_probability"] <= 1.0
        assert 1 <= row["pick_blue"] <= BLUE_NUMBERS
        assert len(row["pick_reds"]) == 6


def test_walk_forward_is_reproducible_for_a_fixed_seed(wf_draws, fast_config, rows_seed_11):
    repeated = walk_forward_lstm(wf_draws, fast_config, min_history=6, seed=11)
    assert repeated == rows_seed_11


def test_argmax_blue_ignores_the_seed_while_the_sampled_blue_moves(rows_seed_11, rows_seed_99):
    # lstm_argmax is a point estimate: no rng, so a different sampling seed must
    # not move it. lstm_sample draws from the generator, so it must move.
    assert _blues_by_issue(rows_seed_11, STRATEGY_ARGMAX) == _blues_by_issue(rows_seed_99, STRATEGY_ARGMAX)
    assert _blues_by_issue(rows_seed_11, STRATEGY_SAMPLE) != _blues_by_issue(rows_seed_99, STRATEGY_SAMPLE)


def test_argmax_ticket_is_the_top_scoring_ticket(trained_model, window):
    from ssq_research.lstm import pick_argmax, split_probabilities

    probabilities = predict_probabilities(trained_model, synthetic_draws(20)[-window:])
    red_scores, blue_scores = split_probabilities(probabilities)
    pick = pick_argmax(probabilities)

    assert pick.blue == int(np.argmax(blue_scores)) + 1
    for number in pick.reds:
        assert number - 1 in set(np.argsort(red_scores)[-6:].tolist())


def test_set_global_seed_makes_every_rng_reproducible():
    """Guards the seeding helper the whole reproducibility story depends on."""
    import random as pyrandom

    from ssq_research.lstm import set_global_seed

    set_global_seed(1234)
    first = (pyrandom.random(), float(np.random.random()))
    set_global_seed(1234)
    second = (pyrandom.random(), float(np.random.random()))

    assert first == second
    set_global_seed(4321)
    assert (pyrandom.random(), float(np.random.random())) != first


def test_deterministic_mode_runs_without_crashing(wf_draws, window):
    """``deterministic=True`` opts into op-level determinism; it must not raise."""
    config = LstmConfig(
        window=window,
        hidden_units=4,
        layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        max_predictions=1,
        seed=0,
        deterministic=True,
    )
    rows = walk_forward_lstm(wf_draws, config, min_history=6, seed=5)
    assert len(rows) == 2
    assert all(row["deterministic"] is True for row in rows)
