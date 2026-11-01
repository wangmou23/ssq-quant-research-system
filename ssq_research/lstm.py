"""Optional TensorFlow LSTM pipeline for SSQ research.

This module is intentionally separate from the dependency-free baseline. It
does not claim predictive power; every evaluation must use walk-forward data.

Design note (v0.2.0)
--------------------
The 0.1.0 network used a single ``Dense(49, activation="sigmoid")`` head trained
with ``binary_crossentropy``, i.e. it treated the 33 red and 16 blue balls as 49
independent binary classifications. For that objective the optimal solution is
the vector of historical marginal frequencies, so the network could never
express "which number is due" - its ceiling is ``FrequencyStrategy``. Because the
blue marginals of a long SSQ history are extremely close to the uniform 1/16, the
``argmax`` over them landed on the same ball every period, which is why two
Kaggle runs returned an identical blue ball.

The fix splits the output into a semantic head per game rule:

* ``red_head``: ``Dense(33, sigmoid)`` + BCE - six of 33 is genuinely multi-label.
* ``blue_head``: ``Dense(16, softmax)`` + sparse categorical cross-entropy - one of
  16 is a single choice and needs a normalised distribution.

``predict_probabilities`` is the only place that concatenates the two heads back
into the legacy 49-slot vector, so the downstream contract is preserved while the
slicing logic exists exactly once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .core import Draw, Pick, build_result_row

#: Encoded feature layout shared by every function in this module:
#: indices ``0:33`` are red balls 1..33, indices ``33:49`` are blue balls 1..16.
RED_NUMBERS = 33
BLUE_NUMBERS = 16
FEATURE_COUNT = RED_NUMBERS + BLUE_NUMBERS
RED_SLICE = slice(0, RED_NUMBERS)
BLUE_SLICE = slice(RED_NUMBERS, FEATURE_COUNT)
REDS_PER_DRAW = 6

#: Identifier of the deterministic point-estimate strategy.
STRATEGY_ARGMAX = "lstm_argmax"
#: Identifier of the seeded without-replacement sampling strategy.
STRATEGY_SAMPLE = "lstm_sample"


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("LSTM utilities require optional dependency numpy") from exc
    return np


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("LSTM requires optional dependency tensorflow; install with [lstm]") from exc
    return tf


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow so repeated runs are comparable.

    Without this every run re-initialised the LSTM weights from a different
    random state, which made the published numbers impossible to reproduce.

    ``deterministic=True`` additionally enables
    ``tf.config.experimental.enable_op_determinism()``. That removes the last
    non-determinism from GPU/threaded kernels but it costs a lot of speed, so it
    stays off by default and is opt-in through :class:`LstmConfig`.
    """
    seed = int(seed)
    random.seed(seed)
    np = _numpy()
    np.random.seed(seed % (2**32))
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


@dataclass(frozen=True)
class LstmConfig:
    window: int = 10
    hidden_units: int = 32
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    retrain_every: int = 1
    max_predictions: int | None = None
    verbose: int = 0
    #: Global seed used for weight initialisation, dropout and sampling.
    seed: int = 0
    #: Opt-in op-level determinism; slower, so disabled by default.
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be at least 1")
        if self.layers < 1:
            raise ValueError("layers must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        if self.max_predictions is not None and self.max_predictions < 1:
            raise ValueError("max_predictions must be at least 1 when set")


def encode_draw(draw: Draw) -> np.ndarray:
    """Encode one draw as 49 binary indicators (33 red + 16 blue)."""
    np = _numpy()
    vector = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for number in draw.reds:
        vector[number - 1] = 1.0
    vector[RED_NUMBERS + draw.blue - 1] = 1.0
    return vector


def make_sequences(draws: Iterable[Draw], window: int, return_heads: bool = False):
    """Build sliding-window training pairs.

    With ``return_heads=False`` (default, unchanged contract) the targets are the
    legacy ``(N, 49)`` multi-hot matrix. With ``return_heads=True`` the targets are
    split to match the dual-head model: ``y_red`` is ``(N, 33)`` multi-hot and
    ``y_blue`` is ``(N,)`` integer class labels in ``0..15`` (``blue - 1``), which
    is what ``sparse_categorical_crossentropy`` expects.

    The blue label is derived from the exact same ``BLUE_SLICE`` used by
    ``encode_draw`` and ``split_probabilities``, so the red/blue boundary can not
    drift between the encoder and the target builder.
    """
    np = _numpy()
    ordered = list(draws)
    if window < 1 or len(ordered) <= window:
        raise ValueError("draw count must be greater than window")
    encoded = np.stack([encode_draw(draw) for draw in ordered])
    sequences = np.stack([encoded[i - window:i] for i in range(window, len(encoded))])
    targets = encoded[window:]
    if not return_heads:
        return sequences, targets
    y_red = targets[:, RED_SLICE]
    # One-hot -> class index; ``-1`` because blue 1 maps to class 0.
    y_blue = (np.argmax(targets[:, BLUE_SLICE], axis=1)).astype("int32")
    return sequences, y_red, y_blue


def pack_targets(y_red: np.ndarray, y_blue: np.ndarray) -> np.ndarray:
    """Recombine dual-head targets into the legacy ``(N, 49)`` multi-hot matrix."""
    np = _numpy()
    red = np.asarray(y_red, dtype=np.float32)
    blue_labels = np.asarray(y_blue, dtype="int32").reshape(-1)
    blue = np.zeros((blue_labels.size, BLUE_NUMBERS), dtype=np.float32)
    blue[np.arange(blue_labels.size), blue_labels] = 1.0
    return np.concatenate([red, blue], axis=1)


def combine_heads(red_scores: np.ndarray, blue_scores: np.ndarray) -> np.ndarray:
    """Concatenate dual-head outputs into one ``(batch, 49)`` probability matrix."""
    np = _numpy()
    red = np.asarray(red_scores, dtype=np.float32).reshape(-1, RED_NUMBERS)
    blue = np.asarray(blue_scores, dtype=np.float32).reshape(-1, BLUE_NUMBERS)
    if red.shape[0] != blue.shape[0]:
        raise ValueError("red_head and blue_head outputs must share the same batch size")
    return np.concatenate([red, blue], axis=1)


def split_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the legacy 49-slot vector into ``(red_scores 33, blue_scores 16)``."""
    np = _numpy()
    array = np.asarray(probabilities, dtype="float64").reshape(-1)
    if array.size != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} probabilities, got {array.size}")
    return array[RED_SLICE], array[BLUE_SLICE]


def _split_head_outputs(raw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalise whatever ``keras.Model.predict`` returned into two head arrays."""
    np = _numpy()
    if isinstance(raw, dict):
        return np.asarray(raw["red_head"]), np.asarray(raw["blue_head"])
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise ValueError(f"expected two model outputs, got {len(raw)}")
        return np.asarray(raw[0]), np.asarray(raw[1])
    array = np.asarray(raw)
    if array.ndim == 2 and array.shape[1] == FEATURE_COUNT:
        # Legacy single-head model: keep old checkpoints usable.
        return array[:, RED_SLICE], array[:, BLUE_SLICE]
    raise ValueError("unexpected model output; expected dual-head (red_head, blue_head) tensors")


def build_model(config: LstmConfig) -> Any:
    """Build the dual-head Keras LSTM; TensorFlow is an optional dependency."""
    tf = _tensorflow()
    keras = tf.keras
    set_global_seed(config.seed, config.deterministic)
    inputs = keras.Input(shape=(config.window, FEATURE_COUNT), name="history")
    x = inputs
    for layer_index in range(config.layers):
        x = keras.layers.LSTM(
            config.hidden_units,
            return_sequences=layer_index < config.layers - 1,
            name=f"lstm_{layer_index + 1}",
        )(x)
        if config.dropout:
            x = keras.layers.Dropout(config.dropout, name=f"dropout_{layer_index + 1}")(x)
    red_outputs = keras.layers.Dense(RED_NUMBERS, activation="sigmoid", name="red_head")(x)
    blue_outputs = keras.layers.Dense(BLUE_NUMBERS, activation="softmax", name="blue_head")(x)
    model = keras.Model(inputs, [red_outputs, blue_outputs], name="ssq_dual_head_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss={"red_head": "binary_crossentropy", "blue_head": "sparse_categorical_crossentropy"},
        loss_weights={"red_head": 1.0, "blue_head": 1.0},
    )
    return model


def train_model(draws: Iterable[Draw], config: LstmConfig, validation_split: float = 0.2) -> Any:
    """Train only on the supplied historical prefix; callers own the split."""
    _tensorflow()
    set_global_seed(config.seed, config.deterministic)
    sequences, y_red, y_blue = make_sequences(draws, config.window, return_heads=True)
    model = build_model(config)
    model.fit(
        sequences,
        [y_red, y_blue],
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=validation_split,
        shuffle=False,
        verbose=config.verbose,
    )
    return model


def predict_probabilities(model: Any, recent_draws: Sequence[Draw] | np.ndarray, verbose: int = 0) -> np.ndarray:
    """Return the legacy 49-slot probability vector (33 red + 16 blue).

    ``recent_draws`` is either one window of :class:`~ssq_research.core.Draw`
    objects (returns shape ``(49,)``) or an already batched array of shape
    ``(batch, window, 49)`` (returns ``(batch, 49)``). This is the single place
    where the two heads are stitched together; nothing else should slice the
    model output.
    """
    np = _numpy()
    single = not isinstance(recent_draws, np.ndarray)
    if single:
        window_draws = list(recent_draws)
        if not window_draws:
            raise ValueError("recent_draws must contain at least one draw")
        batch = np.expand_dims(np.stack([encode_draw(draw) for draw in window_draws]), 0)
    else:
        batch = np.asarray(recent_draws, dtype=np.float32)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, 0)
    raw = model.predict(batch, verbose=verbose)
    combined = combine_heads(*_split_head_outputs(raw))
    return combined[0] if single else combined


def normalize_weights(scores: np.ndarray) -> np.ndarray:
    """Turn arbitrary scores into a legal ``p`` vector for ``np.random.Generator.choice``.

    NaN/inf slots are treated as zero, negative scores are clipped away, and the
    all-zero (or non-finite total) case falls back to the uniform distribution so
    the sampler always stays valid instead of raising.
    """
    np = _numpy()
    weights = np.asarray(scores, dtype="float64").reshape(-1)
    if weights.size == 0:
        raise ValueError("cannot sample from an empty score vector")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(weights.size, 1.0 / weights.size, dtype="float64")
    weights = weights / total
    # numpy rejects ``p`` whose sum deviates from 1 by more than ~1e-8; a float
    # division can drift, so renormalise once more.
    return weights / weights.sum()


def _sample_without_replacement(rng: np.random.Generator, weights: np.ndarray, size: int) -> list[int]:
    """Draw ``size`` distinct 1-based positions; fall back to uniform if needed."""
    np = _numpy()
    population = np.arange(1, weights.size + 1)
    if int(np.count_nonzero(weights > 0.0)) < size:
        # Not enough non-zero support to draw ``size`` distinct positions; numpy
        # would raise "Fewer non-zero entries in p than size". Uniform is the
        # honest fallback: it is what a flat model means anyway.
        weights = np.full(weights.size, 1.0 / weights.size, dtype="float64")
    chosen = rng.choice(population, size=size, replace=False, p=weights)
    return [int(value) for value in np.sort(chosen).tolist()]


def pick_argmax(probabilities: np.ndarray) -> Pick:
    """Deterministic point estimate: top-6 red scores and top-1 blue score.

    This strategy contains **no randomness whatsoever**. For a fixed probability
    vector it always returns the same pick, and a nearly-flat blue distribution
    means the same ball wins every time. Report it as a point estimate; never
    present it as a sampled prediction.
    """
    np = _numpy()
    red_scores, blue_scores = split_probabilities(probabilities)
    order = np.argsort(red_scores, kind="stable")
    reds = tuple(sorted((order[-REDS_PER_DRAW:] + 1).tolist()))
    blue = int(np.argmax(blue_scores) + 1)
    return Pick(tuple(int(number) for number in reds), blue)


def pick_sampled(probabilities: np.ndarray, rng: np.random.Generator) -> Pick:
    """Sample one ticket without replacement from the model distribution.

    Six distinct red balls are drawn from the normalised 33-way red weights and
    one blue ball from the normalised 16-way blue weights, using a caller-supplied
    ``numpy.random.Generator``. Passing the same generator state reproduces the
    same ticket; that is the only reason ``seed`` exists.
    """
    np = _numpy()
    red_scores, blue_scores = split_probabilities(probabilities)
    reds = _sample_without_replacement(rng, normalize_weights(red_scores), REDS_PER_DRAW)
    blue = _sample_without_replacement(rng, normalize_weights(blue_scores), 1)[0]
    return Pick(tuple(sorted(reds)), int(blue))


def pick_from_probabilities(probabilities: np.ndarray, seed: int = 0) -> Pick:
    """Backward-compatible wrapper around :func:`pick_sampled`.

    In 0.1.0 this function accepted ``seed`` and silently ignored it, so every
    call returned the argmax regardless of the seed. It now really samples with
    ``numpy.random.default_rng(seed)``; use :func:`pick_argmax` when a
    deterministic point estimate is actually wanted.
    """
    np = _numpy()
    return pick_sampled(probabilities, np.random.default_rng(int(seed)))


def walk_forward_lstm(draws: Iterable[Draw], config: LstmConfig, min_history: int, seed: int = 0) -> list[dict]:
    """Retrain on each prefix and evaluate only the following target draw.

    Every prefix produces **two** rows: the deterministic ``lstm_argmax`` point
    estimate and the seeded ``lstm_sample`` ticket. Reporting both makes the
    degeneracy visible - if ``lstm_argmax`` collapses onto one blue ball while
    ``lstm_sample`` spreads over 16, the model distribution is flat and the
    argmax was never a prediction. Each row also carries ``seed`` and
    ``deterministic`` so a run can be reproduced exactly.

    ``seed`` drives the sampling generator (and the global RNG used here), while
    ``config.seed`` drives weight initialisation and dropout inside
    :func:`train_model`; pass the same value to both for a fully reproducible run.
    """
    np = _numpy()
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if min_history <= config.window or min_history >= len(ordered):
        raise ValueError("min_history must be greater than window and smaller than draw count")
    if config.retrain_every < 1:
        raise ValueError("retrain_every must be at least 1")
    start = min_history
    if config.max_predictions is not None:
        start = max(min_history, len(ordered) - config.max_predictions)
    set_global_seed(seed, config.deterministic)
    rng = np.random.default_rng(int(seed))
    results: list[dict] = []
    model = None
    for index in range(start, len(ordered)):
        prefix = ordered[:index]
        if model is None or (index - start) % config.retrain_every == 0:
            model = train_model(prefix, config, validation_split=0.0)
        probabilities = predict_probabilities(model, prefix[-config.window:], verbose=0)
        target = ordered[index]
        _, blue_scores = split_probabilities(probabilities)
        extra = {
            "training_end_issue": prefix[-1].issue,
            "seed": int(seed),
            "deterministic": config.deterministic,
            "blue_max_probability": round(float(np.max(blue_scores)), 6),
        }
        results.append(build_result_row(STRATEGY_ARGMAX, target, pick_argmax(probabilities), extra))
        results.append(build_result_row(STRATEGY_SAMPLE, target, pick_sampled(probabilities, rng), extra))
    return results
