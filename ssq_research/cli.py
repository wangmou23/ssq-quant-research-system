from __future__ import annotations

import argparse
import random
from pathlib import Path

from .core import Draw, FrequencyStrategy, RandomStrategy, load_csv, run_backtest, write_json


def demo() -> None:
    draws = [Draw(str(2026001 + i), f"2026-01-{i + 1:02d}", tuple(range(1 + i % 20, 7 + i % 20)), 1 + i % 16) for i in range(40)]
    rows = run_backtest(draws, (RandomStrategy(), FrequencyStrategy()), window=10, seed=7)
    print(f"rows={len(rows)} strategies={sorted({row['strategy'] for row in rows})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SSQ statistical research and fair backtesting")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--csv", type=Path, required=True)
    backtest.add_argument("--window", type=int, default=30)
    backtest.add_argument("--output", type=Path, default=Path("backtest.json"))
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        rows = run_backtest(load_csv(args.csv), (RandomStrategy(), FrequencyStrategy()), window=args.window)
        write_json(rows, args.output)
        print(f"wrote={args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()

