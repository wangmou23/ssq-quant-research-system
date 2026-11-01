from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Pick


@dataclass(frozen=True)
class PrizeResult:
    """Outcome of one bet.

    ``payout`` is ``None`` when the amount cannot be known from history alone
    (first and second prize, whose value depends on the draw pool). ``None`` is
    used instead of ``0.0`` so that downstream accounting cannot silently book an
    unknown jackpot as a zero-revenue, negative-net ticket.
    """

    level: str
    payout: float | None
    payout_basis: str


def evaluate_prize(pick: Pick, actual_reds: tuple[int, ...], actual_blue: int) -> PrizeResult:
    """Evaluate prize level using fixed historical baseline amounts.

    Actual first/second prize amounts vary with the draw pool; those levels are
    deliberately reported with ``payout=None`` and must be replaced by official
    data before any revenue figure is quoted.
    """
    red_hits = len(set(pick.reds) & set(actual_reds))
    blue_hit = pick.blue == actual_blue
    if red_hits == 6 and blue_hit:
        return PrizeResult("一等奖", None, "variable_estimate_required")
    if red_hits == 6:
        return PrizeResult("二等奖", None, "variable_estimate_required")
    if red_hits == 5 and blue_hit:
        return PrizeResult("三等奖", 3000.0, "fixed_baseline")
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return PrizeResult("四等奖", 200.0, "fixed_baseline")
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return PrizeResult("五等奖", 10.0, "fixed_baseline")
    if blue_hit:
        return PrizeResult("六等奖", 5.0, "fixed_baseline")
    return PrizeResult("未中奖", 0.0, "fixed_baseline")
