from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol


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


class Strategy(Protocol):
    name: str

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick: ...


class RandomStrategy:
    name = "random"

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick:
        return Pick(tuple(sorted(rng.sample(range(1, 34), 6))), rng.randint(1, 16))


class FrequencyStrategy:
    name = "frequency"

    def pick(self, history: tuple[Draw, ...], rng: random.Random) -> Pick:
        if not history:
            return RandomStrategy().pick(history, rng)
        red_counts = {n: 0 for n in range(1, 34)}
        blue_counts = {n: 0 for n in range(1, 17)}
        for draw in history:
            for n in draw.reds:
                red_counts[n] += 1
            blue_counts[draw.blue] += 1
        reds = tuple(sorted(n for n, _ in sorted(red_counts.items(), key=lambda item: (-item[1], item[0]))[:6]))
        blue = max(blue_counts, key=lambda n: (blue_counts[n], -n))
        return Pick(reds, blue)


def load_csv(path: Path) -> list[Draw]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        draws = []
        for row in rows:
            reds = tuple(sorted(int(row[f"red_{i}"]) for i in range(1, 7)))
            draws.append(Draw(row["issue"], row["draw_date"], reds, int(row["blue"])))
    return sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))


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
            red_hits = len(set(pick.reds) & set(target.reds))
            blue_hits = int(pick.blue == target.blue)
            results.append({"strategy": strategy.name, "issue": target.issue, "red_hits": red_hits, "blue_hits": blue_hits, "cost": 2.0})
    return results


def write_json(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

