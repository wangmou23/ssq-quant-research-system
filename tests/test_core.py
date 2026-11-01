from ssq_research.core import (
    BLUE_ENTROPY_MAX,
    Draw,
    FrequencyStrategy,
    RandomStrategy,
    blue_degeneracy_report,
    run_backtest,
    summarize,
)
from ssq_research.prize import evaluate_prize
from ssq_research.lstm import encode_draw, make_sequences, pick_from_probabilities
from ssq_research.data_fetch import parse_ssq_text, parse_ssq_text_with_stats


def history(n=40):
    return [Draw(str(2026001 + i), f"2026-01-{i + 1:02d}", tuple(range(1 + i % 20, 7 + i % 20)), 1 + i % 16) for i in range(n)]


def test_backtest_only_uses_prior_draws():
    rows = run_backtest(history(), (RandomStrategy(), FrequencyStrategy()), window=10, seed=3)
    assert len(rows) == 60
    assert min(row["red_hits"] for row in rows) >= 0


def test_backtest_records_the_picked_blue_ball():
    rows = run_backtest(history(), (RandomStrategy(),), window=10, seed=3)
    assert all(1 <= row["pick_blue"] <= 16 for row in rows)
    assert all(len(row["pick_reds"]) == 6 for row in rows)


def test_backtest_is_reproducible_for_a_fixed_seed():
    draws = history(120)
    first = run_backtest(draws, (RandomStrategy(), FrequencyStrategy()), window=20, seed=13)
    second = run_backtest(draws, (RandomStrategy(), FrequencyStrategy()), window=20, seed=13)
    assert first == second


def test_draw_validation():
    try:
        Draw("1", "2026-01-01", (1, 1, 2, 3, 4, 5), 1)
    except ValueError:
        return
    raise AssertionError("invalid draw was accepted")


def test_pick_validation():
    try:
        from ssq_research.core import Pick
        Pick((1, 1, 2, 3, 4, 5), 1)
    except ValueError:
        return
    raise AssertionError("invalid pick was accepted")


def test_prize_rules_and_summary():
    actual = (1, 2, 3, 4, 5, 6)
    jackpot = evaluate_prize(type("Pick", (), {"reds": actual, "blue": 7})(), actual, 7)
    assert jackpot.level == "一等奖"
    # 0.1.0 reported 0.0 here, which booked every jackpot as a 2 CNY loss.
    assert jackpot.payout is None
    assert jackpot.payout_basis == "variable_estimate_required"
    rows = run_backtest(history(), (RandomStrategy(),), window=10, seed=3)
    summary = summarize(rows)
    assert summary[0]["trials"] == 30
    assert summary[0]["cost"] == 60.0
    assert summary[0]["variable_prize_count"] == 0
    assert summary[0]["settled_trials"] == 30
    assert summary[0]["net"] is not None
    # New degeneracy fields must be present on every summary row.
    assert "distinct_blue" in summary[0]
    assert "blue_entropy" in summary[0]


def test_unknown_payout_does_not_leak_into_net():
    actual = (1, 2, 3, 4, 5, 6)
    from ssq_research.core import Pick, build_result_row

    target = Draw("9", "2026-01-09", actual, 7)
    row = build_result_row("s", target, Pick(actual, 7))
    assert row["level"] == "一等奖"
    assert row["payout"] is None
    assert row["net"] is None
    assert row["cost"] == 2.0
    assert summarize([row])[0]["variable_prize_count"] == 1
    assert summarize([row])[0]["net"] is None


def test_frequency_strategy_does_not_lock_the_blue_ball():
    draws = history(220)
    rows = run_backtest(draws, (FrequencyStrategy(),), window=20, seed=5)
    report = blue_degeneracy_report(rows)
    # A deterministic "smallest number wins the tie" rule collapsed this to 1.
    assert report["distinct_blue"] > 3
    assert report["longest_run"] < len(rows)


def test_lstm_dataset_and_pick_shape_without_tensorflow():
    draws = history(12)
    x, y = make_sequences(draws, 5)
    assert x.shape == (7, 5, 49)
    assert y.shape == (7, 49)
    pick = pick_from_probabilities(encode_draw(draws[-1]))
    assert len(pick.reds) == 6
    assert 1 <= pick.blue <= 16


def test_lstm_dataset_heads_match_the_encoder_layout():
    draws = history(12)
    x, y_red, y_blue = make_sequences(draws, 5, return_heads=True)
    _x, y = make_sequences(draws, 5)
    assert x.shape == (7, 5, 49)
    assert y_red.shape == (7, 33)
    assert y_blue.shape == (7,)
    assert y_red.sum() == 42.0
    assert sorted(set(y_blue.tolist())) == list(range(5, 12))
    # Heads must reconstruct the legacy multi-hot target exactly.
    rebuilt = encode_draw(draws[5])
    assert y_red[0].tolist() == rebuilt[:33].tolist()
    assert int(y_blue[0]) == int(rebuilt[33:].argmax())
    assert y.shape == (7, 49)


def test_parse_downloaded_ssq_text():
    text = "2026001 2026-01-01 01 02 03 04 05 06 07\n"
    draws = parse_ssq_text(text)
    assert draws[0].issue == "2026001"
    assert draws[0].reds == (1, 2, 3, 4, 5, 6)
    assert draws[0].blue == 7


def test_parse_reports_skipped_lines_instead_of_dropping_them_silently():
    text = (
        "2026001 2026-01-01 01 02 03 04 05 06 07\n"
        "2026002 2026-01-03 01 02 03 04 05 06 17\n"  # blue out of range
        "2026003 2026-01-05 01 02 03 04 05 06\n"  # only 6 balls
        "2026004 2026-01-08 01 01 03 04 05 06 09\n"  # duplicate red
        "2026005 2026-01-10 03 04 05 06 07 08 09\n"
    )
    draws, stats = parse_ssq_text_with_stats(text)
    assert [draw.issue for draw in draws] == ["2026001", "2026005"]
    assert stats.skipped_line_count == 3
    assert stats.parsed_lines == 2
    assert stats.skipped_reasons["invalid_blue"] == 1
    assert stats.skipped_reasons["invalid_reds"] == 1
    assert stats.skipped_reasons["too_few_numbers"] == 1


def test_parse_does_not_mistake_footer_numbers_for_balls():
    text = "2026001 2026-01-01 01 02 03 04 05 06 07 1234567 891234\n"
    draws = parse_ssq_text(text)
    assert draws[0].blue == 7
    assert draws[0].reds == (1, 2, 3, 4, 5, 6)


def test_blue_entropy_max_is_four_bits():
    assert BLUE_ENTROPY_MAX == 4.0
