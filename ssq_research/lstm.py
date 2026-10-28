"""Optional TensorFlow LSTM pipeline for SSQ research.

This module is intentionally separate from the dependency-free baseline. It
does not claim predictive power; every evaluation must use walk-forward data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .core import Draw, Pick


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("LSTM utilities require optional dependency numpy") from exc
    return np


@dataclass(frozen=True)
class LstmConfig:
    window: int = 10
    hidden_units: int = 32
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001


def encode_draw(draw: Draw) -> np.ndarray:
    """Encode one draw as 49 binary indicators (33 red + 16 blue)."""
    np = _numpy()
    vector = np.zeros(49, dtype=np.float32)
    for number in draw.reds:
        vector[number - 1] = 1.0
    vector[33 + draw.blue - 1] = 1.0
    return vector


def make_sequences(draws: Iterable[Draw], window: int) -> tuple[np.ndarray, np.ndarray]:
    np = _numpy()
    ordered = list(draws)
    if window < 1 or len(ordered) <= window:
        raise ValueError("draw count must be greater than window")
    encoded = np.stack([encode_draw(draw) for draw in ordered])
    return np.stack([encoded[i - window:i] for i in range(window, len(encoded))]), encoded[window:]


def build_model(config: LstmConfig) -> Any:
    """Build a multi-layer Keras LSTM; TensorFlow is an optional dependency."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("LSTM requires optional dependency tensorflow; install with [lstm]") from exc
    inputs = keras.Input(shape=(config.window, 49), name="history")
    x = inputs
    for layer_index in range(config.layers):
        x = keras.layers.LSTM(config.hidden_units, return_sequences=layer_index < config.layers - 1)(x)
        if config.dropout:
            x = keras.layers.Dropout(config.dropout)(x)
    outputs = keras.layers.Dense(49, activation="sigmoid", name="next_draw") (x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss="binary_crossentropy")
    return model


def train_model(draws: Iterable[Draw], config: LstmConfig, validation_split: float = 0.2) -> Any:
    """Train only on the supplied historical prefix; callers own the split."""
    x, y = make_sequences(draws, config.window)
    model = build_model(config)
    model.fit(x, y, epochs=config.epochs, batch_size=config.batch_size, validation_split=validation_split, shuffle=False, verbose=0)
    return model


def pick_from_probabilities(probabilities: np.ndarray, seed: int = 0) -> Pick:
    np = _numpy()
    red_scores = probabilities[:33]
    blue_scores = probabilities[33:]
    reds = tuple(sorted((np.argsort(red_scores)[-6:] + 1).tolist()))
    blue = int(np.argmax(blue_scores) + 1)
    return Pick(reds, blue)


def walk_forward_lstm(draws: Iterable[Draw], config: LstmConfig, min_history: int, seed: int = 0) -> list[dict]:
    """Retrain on each prefix and evaluate only the following target draw."""
    np = _numpy()
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if min_history <= config.window or min_history >= len(ordered):
        raise ValueError("min_history must be greater than window and smaller than draw count")
    results = []
    for index in range(min_history, len(ordered)):
        prefix = ordered[:index]
        model = train_model(prefix, config, validation_split=0.0)
        probabilities = model.predict(np.expand_dims(np.stack([encode_draw(d) for d in prefix[-config.window:]]), 0), verbose=0)[0]
        pick = pick_from_probabilities(probabilities, seed=seed + index)
        target = ordered[index]
        results.append({"strategy": "lstm", "issue": target.issue, "red_hits": len(set(pick.reds) & set(target.reds)), "blue_hits": int(pick.blue == target.blue), "training_end_issue": prefix[-1].issue})
    return results

