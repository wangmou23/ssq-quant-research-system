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


def main() -> None:
    parser = argparse.ArgumentParser(description="SSQ statistical research and fair backtesting")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", type=Path, required=True)
    backtest.add_argument("--window", type=int, default=30)
    backtest.add_argument("--seed", type=int, default=0)
    backtest.add_argument("--output", type=Path, default=Path("backtest.json"))
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        rows = run_backtest(load_csv(args.csv), (RandomStrategy(), FrequencyStrategy()), window=args.window, seed=args.seed)
        write_json(rows, args.output)
        print(f"wrote={args.output} rows={len(rows)}")
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        for name in sorted({row["strategy"] for row in rows}):
            strategy_rows = [row for row in rows if row["strategy"] == name]
            print(name, json.dumps(blue_degeneracy_report(strategy_rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
