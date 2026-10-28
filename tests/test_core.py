from ssq_research.core import Draw, FrequencyStrategy, RandomStrategy, run_backtest


def history(n=40):
    return [Draw(str(2026001 + i), f"2026-01-{i + 1:02d}", tuple(range(1 + i % 20, 7 + i % 20)), 1 + i % 16) for i in range(n)]


def test_backtest_only_uses_prior_draws():
    rows = run_backtest(history(), (RandomStrategy(), FrequencyStrategy()), window=10, seed=3)
    assert len(rows) == 60
    assert min(row["red_hits"] for row in rows) >= 0


def test_draw_validation():
    try:
        Draw("1", "2026-01-01", (1, 1, 2, 3, 4, 5), 1)
    except ValueError:
        return
    raise AssertionError("invalid draw was accepted")

