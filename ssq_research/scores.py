"""Score-producing algorithms behind one shared contract.

Every algorithm answers the same narrow question: given the draws strictly
*before* the target period, return a 49-element vector (33 red then 16 blue) of
non-negative scores. Turning scores into a ticket is a separate step, so the
same algorithm can be evaluated deterministically (``argmax``) or stochastically
(``sample``) and always against the random baseline from :mod:`ssq_research.core`.

The four families here differ in what they *look at*, not in whether they can
beat a fair draw:

* frequency -- how often each ball appeared, optionally recency-weighted
* markov    -- conditional appearance rates and blue-to-blue transitions
* arima     -- per-ball autoregression on the binary appearance series
* ensemble  -- a weighted vote across the above

None of them is expected to beat ``RandomStrategy``. The point of running them
side by side is that the comparison is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .core import Draw, Pick

RED_NUMBERS = 33
BLUE_NUMBERS = 16
FEATURE_COUNT = RED_NUMBERS + BLUE_NUMBERS


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only on bare installs
        raise RuntimeError(
            'algorithm scoring requires numpy; install with: pip install -e ".[algorithms]"'
        ) from exc
    return np


@runtime_checkable
class Scorer(Protocol):
    """Anything that can turn history into 49 scores."""

    name: str

    def score(self, history: Sequence[Draw]) -> "object":  # numpy.ndarray of shape (49,)
        ...


def normalize(weights) -> "object":
    """Map arbitrary scores to a proper probability distribution.

    NaN, infinity and negative values are treated as zero rather than propagating,
    and a fully degenerate input falls back to uniform. The result sums to 1 to
    within floating point error, which ``numpy``'s sampler requires.
    """
    np = _numpy()
    array = np.asarray(weights, dtype=float).ravel()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(array.shape, 1.0 / array.size, dtype=float)
    result = array / total
    return result / float(result.sum())


@dataclass
class FrequencyScorer:
    """Relative appearance rate per ball.

    ``half_life`` enables exponential recency weighting measured in periods: a
    draw ``half_life`` periods old counts half as much as the latest one. Set it
    to ``None`` for plain historical counts, which is the classic hot/cold number
    view.
    """

    name: str = "frequency"
    half_life: float | None = None

    def score(self, history: Sequence[Draw]):
        np = _numpy()
        scores = np.zeros(FEATURE_COUNT, dtype=float)
        if not history:
            return normalize(scores)
        total = len(history)
        for index, draw in enumerate(history):
            if self.half_life is None:
                weight = 1.0
            else:
                weight = 0.5 ** ((total - 1 - index) / float(self.half_life))
            for number in draw.reds:
                scores[number - 1] += weight
            scores[RED_NUMBERS + draw.blue - 1] += weight
        return normalize(scores)


@dataclass
class MarkovScorer:
    """Two chains, one per ball pool.

    Red balls use a two-state chain per number -- "did it appear in the previous
    draw?" -- and predict ``P(appear next | state now)`` with Laplace smoothing,
    so a short sample cannot produce a zero or a division by zero.

    Blue balls use a 16x16 transition matrix counted over consecutive periods,
    which can express "after 7 the next blue is often 12" in a way a plain
    frequency count cannot. Whether that conditional actually carries signal is
    exactly what the backtest is for.
    """

    name: str = "markov"
    smoothing: float = 1.0

    def score(self, history: Sequence[Draw]):
        np = _numpy()
        scores = np.zeros(FEATURE_COUNT, dtype=float)
        if len(history) < 2:
            return normalize(scores)
        smoothing = float(self.smoothing)
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")

        appeared_given = np.zeros(RED_NUMBERS)
        appeared_total = np.zeros(RED_NUMBERS)
        absent_given = np.zeros(RED_NUMBERS)
        absent_total = np.zeros(RED_NUMBERS)
        previous = set(history[0].reds)
        for draw in history[1:]:
            current = set(draw.reds)
            for number in range(1, RED_NUMBERS + 1):
                index = number - 1
                if number in previous:
                    appeared_total[index] += 1.0
                    if number in current:
                        appeared_given[index] += 1.0
                else:
                    absent_total[index] += 1.0
                    if number in current:
                        absent_given[index] += 1.0
            previous = current
        for number in range(1, RED_NUMBERS + 1):
            index = number - 1
            if number in previous:
                scores[index] = (appeared_given[index] + smoothing) / (appeared_total[index] + 2.0 * smoothing)
            else:
                scores[index] = (absent_given[index] + smoothing) / (absent_total[index] + 2.0 * smoothing)

        matrix = np.full((BLUE_NUMBERS, BLUE_NUMBERS), smoothing)
        for before, after in zip(history, history[1:]):
            matrix[before.blue - 1, after.blue - 1] += 1.0
        row = matrix[history[-1].blue - 1]
        scores[RED_NUMBERS:] = row / row.sum()
        return normalize(scores)


@dataclass
class ArimaScorer:
    """Per-ball ARIMA on the binary appearance series.

    Fitting 49 models per period is expensive, so ``max_history`` caps the series
    and the default order is deliberately tiny. A plain mean of the recent window
    is used as a fallback whenever a fit fails to converge, which keeps a single
    bad series from aborting a whole backtest.
    """

    name: str = "arima"
    order: tuple[int, int, int] = (1, 0, 0)
    max_history: int = 240
    fallback_window: int = 10

    def score(self, history: Sequence[Draw]):
        np = _numpy()
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise RuntimeError(
                'ARIMA scoring requires statsmodels; install with: pip install -e ".[algorithms]"'
            ) from exc
        window = list(history)[-self.max_history:]
        if len(window) < 12:
            return normalize(np.zeros(FEATURE_COUNT))
        series = np.zeros((FEATURE_COUNT, len(window)), dtype=float)
        for index, draw in enumerate(window):
            for number in draw.reds:
                series[number - 1, index] = 1.0
            series[RED_NUMBERS + draw.blue - 1, index] = 1.0
        scores = np.zeros(FEATURE_COUNT, dtype=float)
        for row_index in range(FEATURE_COUNT):
            values = series[row_index]
            if float(values.sum()) in (0.0, float(len(values))):
                scores[row_index] = float(values.mean())
                continue
            try:
                forecast = ARIMA(values, order=self.order, trend="c").fit().forecast(1)[0]
            except Exception:  # convergence failures are common on binary series
                forecast = float(values[-self.fallback_window:].mean())
            scores[row_index] = max(0.0, float(forecast))
        return normalize(scores)


@dataclass
class LstmScorer:
    """Adapter around an already-trained dual-head Keras model.

    The model is trained elsewhere (see :mod:`ssq_research.lstm`); this only keeps
    the score contract uniform so LSTM rows sit in the same comparison table as
    the statistical baselines.
    """

    model: object
    window: int
    name: str = "lstm"

    def score(self, history: Sequence[Draw]):
        from .lstm import predict_probabilities

        recent = list(history)[-self.window:]
        if len(recent) < self.window:
            return normalize(_numpy().zeros(FEATURE_COUNT))
        return normalize(predict_probabilities(self.model, recent))


@dataclass
class EnsembleScorer:
    """Weighted vote across several scorers.

    Components are normalized to distributions *before* mixing, so a scorer that
    happens to emit large raw numbers cannot swamp the vote. Equal weights by
    default.

    An ensemble of models that each fail to beat a fair draw does not beat it
    either; averaging reduces variance between their outputs, not the underlying
    randomness of the draw.
    """

    components: Sequence[Scorer]
    weights: Sequence[float] | None = None
    name: str = "ensemble"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("ensemble needs at least one component")
        if self.weights is not None:
            if len(self.weights) != len(self.components):
                raise ValueError("weights must match components")
            if any(weight < 0 for weight in self.weights):
                raise ValueError("weights must be non-negative")

    def score(self, history: Sequence[Draw]):
        np = _numpy()
        mixed = np.zeros(FEATURE_COUNT, dtype=float)
        if self.weights is None:
            weights = [1.0] * len(self.components)
        else:
            weights = [float(weight) for weight in self.weights]
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("weights must not sum to zero")
        for component, weight in zip(self.components, weights):
            if weight == 0.0:
                continue
            mixed += normalize(component.score(history)) * (weight / total)
        return normalize(mixed)


def top_indices(weights, count: int):
    """Indices of the ``count`` highest scores, best first."""
    np = _numpy()
    array = np.asarray(weights, dtype=float).ravel()
    if count > array.size:
        raise ValueError("count exceeds the number of available scores")
    return np.argsort(array)[::-1][:count]


def _sample_without_replacement(weights, size: int, rng):
    np = _numpy()
    distribution = normalize(weights)
    support = int(np.count_nonzero(distribution))
    if support < size:
        # Not enough non-zero mass: fall back to a uniform draw over the whole
        # pool instead of raising, so a degenerate scorer cannot abort a run.
        return rng.choice(np.arange(distribution.size), size=size, replace=False)
    distribution = distribution / float(distribution.sum())
    return rng.choice(np.arange(distribution.size), size=size, replace=False, p=distribution)


def pick_from_scores(scores, rng, mode: str = "sample") -> Pick:
    """Turn 49 scores into one ticket.

    ``mode="argmax"`` is a deterministic point estimate with no randomness at all;
    ``mode="sample"`` draws without replacement in proportion to the scores, so
    the same scores produce varied tickets across seeds.
    """
    np = _numpy()
    if mode not in {"argmax", "sample"}:
        raise ValueError('mode must be "argmax" or "sample"')
    distribution = normalize(scores)
    if mode == "argmax":
        red_indices = top_indices(distribution[:RED_NUMBERS], 6)
        blue_index = int(top_indices(distribution[RED_NUMBERS:], 1)[0])
    else:
        red_indices = _sample_without_replacement(distribution[:RED_NUMBERS], 6, rng)
        blue_index = int(_sample_without_replacement(distribution[RED_NUMBERS:], 1, rng)[0])
    reds = tuple(sorted(int(index) + 1 for index in np.sort(red_indices)))
    return Pick(reds, blue_index + 1)


@dataclass
class ScorerStrategy:
    """Adapt any :class:`Scorer` to the ``Strategy`` protocol used by ``core``.

    ``name`` carries the mode suffix so backtest rows from the deterministic and
    stochastic variants never get mixed up in a summary table.
    """

    scorer: Scorer
    mode: str = "sample"

    def __post_init__(self) -> None:
        if self.mode not in {"argmax", "sample"}:
            raise ValueError('mode must be "argmax" or "sample"')

    @property
    def name(self) -> str:
        return f"{self.scorer.name}_{self.mode}"

    def pick(self, history, rng) -> Pick:
        return pick_from_scores(self.scorer.score(history), rng, self.mode)


def available_algorithms() -> dict[str, bool]:
    """Which optional scoring backends can actually be imported right now."""
    status = {"frequency": True, "markov": True, "ensemble": True}
    try:
        _numpy()
    except RuntimeError:
        status["frequency"] = status["markov"] = status["ensemble"] = False
    try:
        import statsmodels  # noqa: F401

        status["arima"] = True
    except ImportError:
        status["arima"] = False
    try:
        import tensorflow  # noqa: F401

        status["lstm"] = True
    except ImportError:
        status["lstm"] = False
    return status


def build_scorers(include_recency: bool = True) -> dict[str, Scorer]:
    """The dependency-free scorer set: plain frequency, recency-weighted, markov."""
    scorers: dict[str, Scorer] = {
        "frequency": FrequencyScorer(),
        "markov": MarkovScorer(),
    }
    if include_recency:
        scorers["frequency_recent"] = FrequencyScorer(name="frequency_recent", half_life=30.0)
    return scorers


def build_ensemble(scorers: Sequence[Scorer], weights: Sequence[float] | None = None) -> EnsembleScorer:
    """Ensemble of the given scorers, equal weights unless stated otherwise."""
    return EnsembleScorer(components=list(scorers), weights=weights)
