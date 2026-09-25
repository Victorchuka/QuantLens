from __future__ import annotations

import argparse
import sys
from pathlib import Path

from market_agent.agent import AgentConfig, MarketAnalysisAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-agent",
        description="Pull financial data, train a lightweight ML model, and generate market analysis reports.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "SPY"],
        help="Ticker symbols to analyze. Default: AAPL MSFT NVDA SPY",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Calendar days of historical data to fetch. Default: 730",
    )
    parser.add_argument(
        "--risk-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Controls how strict the signal thresholds are. Default: balanced",
    )
    parser.add_argument(
        "--provider",
        choices=["yfinance", "sample"],
        default="yfinance",
        help=(
            "Data source to use. yfinance downloads real historical Yahoo Finance data; "
            "sample generates fictional offline data. Default: yfinance"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory for cached market data CSV files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated markdown reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig(
        symbols=[symbol.upper() for symbol in args.symbols],
        lookback_days=args.lookback_days,
        risk_profile=args.risk_profile,
        provider=args.provider,
        cache_dir=args.cache_dir,
        reports_dir=args.reports_dir,
    )
    agent = MarketAnalysisAgent(config)
    try:
        report_path = agent.run()
    except (ConnectionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Report generated: {report_path}")
    return 0
