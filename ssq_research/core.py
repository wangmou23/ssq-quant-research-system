from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .prize import evaluate_prize

RED_BALL_COUNT = 33
BLUE_BALL_COUNT = 16
REDS_PER_DRAW = 6
BET_COST = 2.0
#: Theoretical upper bound of blue-ball entropy, ``log2(16) == 4`` bits. Any
#: empirical value far below this on a long backtest means the strategy collapsed.
BLUE_ENTROPY_MAX = math.log2(BLUE_BALL_COUNT)
#: A blue-ball picker that only ever visits this many of the 16 balls is treated
#: as degenerate regardless of how evenly it spreads across them. Needed because
#: four perfectly even balls give exactly ``log2(4) == 2.0`` bits, which the
#: ``entropy < 2.0`` test alone would miss.
MAX_HEALTHY_DISTINCT_BLUE = 7
#: Minimum number of scored picks before ``degenerate`` is decided at all.
MIN_TRIALS_FOR_DEGENERACY = 16


@dataclass(frozen=True)
class Draw:
    issue: str
    draw_date: str
    reds: tuple[int, ...]
    blue: int

    def __post_init__(self) -> None:
        if len(self.reds) != 6 or len(set(self.reds)) != 6 or not all(1 <= n <= 33 for n in self.reds):
            raise ValueError("red balls must contain six distinct values from 1..33")
        if not 1 <= self.blue <= 16:
            raise ValueError("blue ball must be in 1..16")


@dataclass(frozen=True)
class Pick:
    reds: tuple[int, ...]
    blue: int

    def __post_init__(self) -> None:
        if len(self.reds) != 6 or len(set(self.reds)) != 6 or not all(1 <= n <= 33 for n in self.reds):
            raise ValueError("pick red balls must contain six distinct values from 1..33")
        if not 1 <= self.blue <= 16:
            raise ValueError("pick blue ball must be in 1..16")


class Strategy(Protocol):
    name: str

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick: ...


class RandomStrategy:
    name = "random"

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick:
        return Pick(tuple(sorted(rng.sample(range(1, 34), 6))), rng.randint(1, 16))


class FrequencyStrategy:
    """Pick the most frequent red balls and the most frequent blue ball.

    Tie-breaking differs between the two halves on purpose:

    * Red balls fill six distinct positions out of 33, so a fully deterministic
      tie-break (highest count, then lowest number) still yields a fresh
      six-number combination as soon as any count changes. Determinism here is
      harmless and keeps the strategy easy to audit.
    * The blue ball fills a single position. With a deterministic tie-break
      (``max(..., key=lambda n: (count, -n))``) the pick is frozen to the
      smallest number for as long as the tie lasts, and on near-uniform lottery
      data the top count is tied for almost the whole backtest. That silently
      produced a constant blue ball for hundreds of consecutive draws, which is
      indistinguishable from a broken model. The tie is therefore resolved with
      the caller's ``rng``: still reproducible for a fixed seed, but no longer
      degenerate.
    """

    name = "frequency"

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick:
        if not history:
            return RandomStrategy().pick(history, rng)
        red_counts = {n: 0 for n in range(1, RED_BALL_COUNT + 1)}
        blue_counts = {n: 0 for n in range(1, BLUE_BALL_COUNT + 1)}
        for draw in history:
            for n in draw.reds:
                red_counts[n] += 1
            blue_counts[draw.blue] += 1
        reds = tuple(sorted(n for n, _ in sorted(red_counts.items(), key=lambda item: (-item[1], item[0]))[:REDS_PER_DRAW]))
        best_blue_count = max(blue_counts.values())
        blue_candidates = sorted(n for n, count in blue_counts.items() if count == best_blue_count)
        blue = rng.choice(blue_candidates)
        return Pick(reds, blue)


def load_csv(path: Path) -> list[Draw]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        draws = []
        for row_number, row in enumerate(rows, start=2):
            try:
                reds = tuple(sorted(int(row[f"red_{i}"]) for i in range(1, 7)))
                draws.append(Draw(row["issue"], row["draw_date"], reds, int(row["blue"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid SSQ CSV row {row_number}: {row}") from exc
    issues = [draw.issue for draw in draws]
    if len(issues) != len(set(issues)):
        raise ValueError("duplicate issue found in SSQ CSV")
    return sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))


def build_result_row(strategy_name: str, target: Draw, pick: Pick, extra: dict | None = None) -> dict:
    """Score ``pick`` against ``target`` and pack one auditable backtest row.

    ``net`` is ``None`` whenever the payout is unknown (first/second prize,
    whose amounts depend on the draw pool). Recording ``None`` keeps the sum of
    settled rows honest; faking ``0.0`` used to book every jackpot as a 2 CNY
    loss and bias the whole equity curve downwards.
    """
    prize = evaluate_prize(pick, target.reds, target.blue)
    row = {
        "strategy": strategy_name,
        "issue": target.issue,
        "red_hits": len(set(pick.reds) & set(target.reds)),
        "blue_hits": int(pick.blue == target.blue),
        "level": prize.level,
        "payout": prize.payout,
        "payout_basis": prize.payout_basis,
        "cost": BET_COST,
        "net": None if prize.payout is None else prize.payout - BET_COST,
        "pick_reds": list(pick.reds),
        "pick_blue": pick.blue,
    }
    if extra:
        row.update(extra)
    return row


def run_backtest(draws: Iterable[Draw], strategies: Iterable[Strategy], window: int = 30, seed: int = 0) -> list[dict]:
    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if window < 1 or window >= len(ordered):
        raise ValueError("window must be at least 1 and smaller than draw count")
    results: list[dict] = []
    for strategy in strategies:
        rng = random.Random(seed)
        for index in range(window, len(ordered)):
            target = ordered[index]
            history = tuple(ordered[index - window:index])
            pick = strategy.pick(history, rng)
            results.append(build_result_row(strategy.name, target, pick))
    return results


def write_json(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _longest_run(values: list[int]) -> int:
    """Length of the longest run of one repeated value."""
    if not values:
        return 0
    longest = 1
    current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest


def blue_degeneracy_report(rows: Iterable[dict]) -> dict:
    """Measure how degenerate the blue-ball picks of a backtest are.

    A healthy sampler explores all 16 blue balls; a collapsed one keeps
    returning the same number. The metrics below make that visible without
    needing TensorFlow:

    ``distinct_blue``
        Number of different blue balls picked. A random baseline should reach
        16 on a few hundred trials; ``1`` to ``3`` means the pick is locked.
    ``blue_entropy``
        Empirical Shannon entropy of the blue distribution in bits. Compare
        against ``blue_entropy_max == 4.0`` (``log2(16)``): near ``0`` means a
        single number carries all the mass.
    ``top1_share``
        Share of trials taken by the most frequent blue ball.
    ``longest_run``
        Longest streak of identical blue balls in the given row order.
    ``degenerate``
        Convenience flag. It is only decided when at least 16 picks were scored
        (``total <= 15`` is too few to be reliable, so the flag stays ``False``
        and the raw metrics must be read instead). Above that floor a run is
        flagged when ``distinct_blue <= 7`` or the entropy is below half of the
        maximum (``2.0`` bits). The ``distinct_blue`` clause exists because the
        entropy test alone leaves a blind spot: four blue balls that are picked
        perfectly evenly give exactly ``2.0`` bits, which ``entropy < 2.0`` would
        miss even though only a quarter of the ball space is ever explored.

    Rows without a ``pick_blue`` field are ignored for the blue statistics so
    that hand-written or legacy rows cannot raise here.
    """
    ordered_rows = list(rows)
    picks = [int(row["pick_blue"]) for row in ordered_rows if row.get("pick_blue") is not None]
    trials = len(ordered_rows)
    if not picks:
        return {
            "trials": trials,
            "blue_picks": 0,
            "distinct_blue": 0,
            "blue_entropy": 0.0,
            "blue_entropy_max": BLUE_ENTROPY_MAX,
            "top1_share": 0.0,
            "longest_run": 0,
            "degenerate": False,
        }
    counts: dict[int, int] = {}
    for blue in picks:
        counts[blue] = counts.get(blue, 0) + 1
    total = len(picks)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    entropy = 0.0 if entropy <= 0.0 else entropy  # also avoids the -0.0 repr
    top1_share = max(counts.values()) / total
    distinct_blue = len(counts)
    degenerate = total >= MIN_TRIALS_FOR_DEGENERACY and (distinct_blue <= MAX_HEALTHY_DISTINCT_BLUE or entropy < BLUE_ENTROPY_MAX / 2.0)
    return {
        "trials": trials,
        "blue_picks": total,
        "distinct_blue": distinct_blue,
        "blue_entropy": round(entropy, 6),
        "blue_entropy_max": BLUE_ENTROPY_MAX,
        "top1_share": round(top1_share, 6),
        "longest_run": _longest_run(picks),
        "degenerate": degenerate,
    }


def summarize(rows: Iterable[dict]) -> list[dict]:
    """Aggregate backtest rows per strategy.

    Rows whose payout is unknown (``payout is None``) are excluded from the
    ``payout``/``net`` sums and counted separately in ``variable_prize_count`` so
    that an unknown jackpot can never be mistaken for a settled loss.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)
    summary: list[dict] = []
    for name, items in sorted(grouped.items()):
        trials = len(items)
        if trials == 0:  # pragma: no cover - defensive, grouping cannot be empty
            continue
        settled = [item for item in items if item.get("net") is not None]
        report = blue_degeneracy_report(items)
        summary.append(
            {
                "strategy": name,
                "trials": trials,
                "settled_trials": len(settled),
                "variable_prize_count": trials - len(settled),
                "cost": sum(float(item.get("cost") or 0.0) for item in items),
                "payout": sum(float(item["payout"]) for item in settled) if settled else 0.0,
                "net": sum(float(item["net"]) for item in settled) if settled else None,
                "avg_red_hits": sum(item["red_hits"] for item in items) / trials,
                "avg_blue_hits": sum(item["blue_hits"] for item in items) / trials,
                "distinct_blue": report["distinct_blue"],
                "blue_entropy": report["blue_entropy"],
            }
        )
    return summary
