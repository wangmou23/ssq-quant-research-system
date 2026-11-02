from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from .core import Draw, FrequencyStrategy, RandomStrategy, blue_degeneracy_report, load_csv, run_backtest, summarize, write_json


def demo() -> None:
    start = date(2026, 1, 1)
    draws = [Draw(str(2026001 + i), (start + timedelta(days=i)).isoformat(), tuple(range(1 + i % 20, 7 + i % 20)), 1 + i % 16) for i in range(40)]
    rows = run_backtest(draws, (RandomStrategy(), FrequencyStrategy()), window=10, seed=7)
    print(f"rows={len(rows)} strategies={sorted({row['strategy'] for row in rows})}")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


def algorithm_strategies(include_arima: bool = False) -> list:
    """Score-based strategies; every algorithm runs in both argmax and sampled mode.

    Imported lazily so ``demo`` and ``backtest`` keep working on a bare install
    without numpy.
    """
    from .scores import ArimaScorer, ScorerStrategy, build_ensemble, build_scorers

    scorers = list(build_scorers().values())
    strategies: list = []
    for scorer in scorers:
        strategies.append(ScorerStrategy(scorer, "argmax"))
        strategies.append(ScorerStrategy(scorer, "sample"))
    strategies.append(ScorerStrategy(build_ensemble(scorers), "sample"))
    if include_arima:
        strategies.append(ScorerStrategy(ArimaScorer(), "argmax"))
        strategies.append(ScorerStrategy(ArimaScorer(), "sample"))
    return strategies


def write_report(
    draws: list[Draw],
    output: Path,
    *,
    window: int = 100,
    seed: int = 0,
    include_algorithms: bool = True,
    include_arima: bool = False,
    max_periods: int = 0,
) -> list[dict]:
    """Run the comparison backtest and write the self-contained HTML dashboard."""
    from .report import build_report_html

    ordered = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    if max_periods > 0:
        ordered = ordered[-(max_periods + window):]
    strategies = [RandomStrategy(), FrequencyStrategy()]
    if include_algorithms:
        strategies.extend(algorithm_strategies(include_arima))
    rows = run_backtest(ordered, strategies, window=window, seed=seed)
    document = build_report_html(ordered, rows, seed=seed, window=window)
    output.write_text(document, encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SSQ statistical research and fair backtesting")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", type=Path, required=True)
    backtest.add_argument("--window", type=int, default=30)
    backtest.add_argument("--seed", type=int, default=0)
    backtest.add_argument("--output", type=Path, default=Path("backtest.json"))

    report = sub.add_parser("report", help="generate the HTML research dashboard")
    report.add_argument("--csv", type=Path, required=True)
    report.add_argument("--output", type=Path, default=Path("report.html"))
    report.add_argument("--window", type=int, default=100)
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--max-periods", type=int, default=0, help="cap backtest periods; 0 means all")
    report.add_argument("--no-algorithms", action="store_true", help="only run the random and frequency baselines")
    report.add_argument("--with-arima", action="store_true", help="also run the ARIMA scorer (needs statsmodels)")

    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "backtest":
        rows = run_backtest(load_csv(args.csv), (RandomStrategy(), FrequencyStrategy()), window=args.window, seed=args.seed)
        write_json(rows, args.output)
        print(f"wrote={args.output} rows={len(rows)}")
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        for name in sorted({row["strategy"] for row in rows}):
            strategy_rows = [row for row in rows if row["strategy"] == name]
            print(name, json.dumps(blue_degeneracy_report(strategy_rows), ensure_ascii=False))
    else:
        rows = write_report(
            load_csv(args.csv),
            args.output,
            window=args.window,
            seed=args.seed,
            include_algorithms=not args.no_algorithms,
            include_arima=args.with_arima,
            max_periods=args.max_periods,
        )
        print(f"wrote={args.output} rows={len(rows)}")
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
