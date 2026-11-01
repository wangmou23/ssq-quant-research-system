from __future__ import annotations

import pytest

from ssq_research.core import (
    BLUE_ENTROPY_MAX,
    blue_degeneracy_report,
    summarize,
)


def make_row(index: int, blue: int, **overrides) -> dict:
    """A minimal backtest row shaped like ``core.build_result_row`` output."""
    row = {
        "strategy": "test",
        "issue": f"2026{index + 1:03d}",
        "red_hits": 0,
        "blue_hits": 0,
        "level": "未中奖",
        "payout": 0.0,
        "payout_basis": "fixed_baseline",
        "cost": 2.0,
        "net": -2.0,
        "pick_blue": blue,
    }
    row.update(overrides)
    return row


def make_rows(blues: list[int]) -> list[dict]:
    return [make_row(index, blue) for index, blue in enumerate(blues)]


def test_locked_blue_ball_is_reported_as_degenerate():
    report = blue_degeneracy_report(make_rows([7] * 40))
    assert report["trials"] == 40
    assert report["blue_picks"] == 40
    assert report["distinct_blue"] == 1
    assert report["longest_run"] == 40
    assert report["top1_share"] == 1.0
    assert report["blue_entropy"] < 1e-6
    assert report["blue_entropy_max"] == pytest.approx(BLUE_ENTROPY_MAX)
    assert report["degenerate"] is True


def test_rotating_blue_ball_is_reported_as_healthy():
    report = blue_degeneracy_report(make_rows([1 + index % 16 for index in range(160)]))
    assert report["distinct_blue"] == 16
    assert report["longest_run"] == 1
    assert report["top1_share"] == pytest.approx(1 / 16)
    assert report["blue_entropy"] == pytest.approx(BLUE_ENTROPY_MAX, abs=1e-6)
    assert report["degenerate"] is False


def test_partially_collapsed_blue_ball_is_flagged():
    # Only two blue balls ever appear: still degenerate even though the streak is
    # broken, because the distribution carries ~1 bit instead of 4.
    report = blue_degeneracy_report(make_rows([1, 2] * 20))
    assert report["distinct_blue"] == 2
    assert report["longest_run"] == 1
    assert report["blue_entropy"] == pytest.approx(1.0, abs=1e-6)
    assert report["degenerate"] is True


def test_four_evenly_spread_blue_balls_are_still_flagged():
    # Four balls with an even 25% share give exactly log2(4) == 2.0 bits, so the
    # old ``entropy < 2.0`` test alone missed it (a real reporting blind spot).
    # The ``distinct_blue <= 7`` clause now catches it.
    report = blue_degeneracy_report(make_rows([1, 2, 3, 4] * 10))
    assert report["distinct_blue"] == 4
    assert report["top1_share"] == pytest.approx(0.25)
    assert report["blue_entropy"] == pytest.approx(2.0, abs=1e-6)
    assert report["degenerate"] is True


def test_too_few_trials_are_not_judged():
    # The flag stays False below MIN_TRIALS_FOR_DEGENERACY even when the picks are
    # obviously collapsed; the raw metrics remain readable.
    report = blue_degeneracy_report(make_rows([9] * 15))
    assert report["blue_picks"] == 15
    assert report["distinct_blue"] == 1
    assert report["blue_entropy"] == 0.0
    assert report["degenerate"] is False


def test_report_tolerates_rows_without_pick_blue():
    rows = [make_row(0, 3), {"strategy": "test", "issue": "2026002", "red_hits": 0, "blue_hits": 1}]
    report = blue_degeneracy_report(rows)
    assert report["trials"] == 2
    assert report["blue_picks"] == 1
    assert report["distinct_blue"] == 1
    assert report["longest_run"] == 1


def test_report_handles_empty_and_blue_free_input():
    empty = blue_degeneracy_report([])
    assert empty["trials"] == 0
    assert empty["distinct_blue"] == 0
    assert empty["blue_entropy"] == 0.0
    assert empty["longest_run"] == 0
    assert empty["degenerate"] is False
    assert blue_degeneracy_report(iter([]))["trials"] == 0


def test_summarize_exposes_degeneracy_and_skips_unknown_payouts():
    rows = [
        make_row(0, 4),
        make_row(1, 4, blue_hits=1, level="六等奖", payout=5.0, net=3.0),
        make_row(2, 9, level="一等奖", payout=None, payout_basis="variable_estimate_required", net=None),
    ]
    summary = summarize(rows)
    assert len(summary) == 1
    aggregate = summary[0]
    assert aggregate["strategy"] == "test"
    assert aggregate["trials"] == 3
    assert aggregate["settled_trials"] == 2
    assert aggregate["variable_prize_count"] == 1
    assert aggregate["cost"] == pytest.approx(6.0)
    assert aggregate["payout"] == pytest.approx(5.0)
    assert aggregate["net"] == pytest.approx(1.0)
    assert aggregate["avg_red_hits"] == 0.0
    assert aggregate["avg_blue_hits"] == pytest.approx(1 / 3)
    assert aggregate["distinct_blue"] == 2
    assert aggregate["blue_entropy"] == pytest.approx(0.9182958, abs=1e-6)


def test_summarize_without_settled_rows_keeps_net_unknown():
    rows = [make_row(index, 2, payout=None, net=None) for index in range(3)]
    aggregate = summarize(rows)[0]
    assert aggregate["variable_prize_count"] == 3
    assert aggregate["settled_trials"] == 0
    assert aggregate["payout"] == 0.0
    assert aggregate["net"] is None


def test_summarize_on_empty_input_does_not_divide_by_zero():
    assert summarize([]) == []
    assert summarize(iter([])) == []
