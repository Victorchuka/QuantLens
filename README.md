# Autonomous Market Analysis Agent

An autonomous market analysis project that pulls historical financial data, builds technical features, trains a lightweight machine learning model, scores tickers, and generates a decision-ready markdown report.

This project is designed to be resume friendly: it shows data ingestion, feature engineering, model training, signal generation, evaluation, and clean software packaging without requiring paid APIs.

## What It Does

- Pulls real daily OHLCV market data from Yahoo Finance through `yfinance` by default.
- Includes an offline synthetic-data provider for demonstrations and tests.
- Builds features such as momentum, volatility, moving-average spread, volume trend, and drawdown.
- Trains a small logistic-regression classifier from scratch using Python's standard library.
- Predicts the probability of a positive next-day move.
- Converts model output into `BUY`, `HOLD`, or `SELL` style signals.
- Generates a markdown report under `reports/`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m market_agent --symbols AAPL MSFT NVDA --lookback-days 365
```

If the project and its dependencies are already installed:

```bash
PYTHONPATH=src python -m market_agent --symbols AAPL MSFT NVDA --lookback-days 365
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Example

```bash
PYTHONPATH=src python -m market_agent \
  --symbols AAPL MSFT NVDA SPY \
  --lookback-days 730 \
  --risk-profile balanced
```

Offline deterministic demo:

```bash
PYTHONPATH=src python -m market_agent --provider sample --symbols AAPL MSFT NVDA
```

The default `yfinance` provider uses real historical data. The `sample` provider
uses generated fictional data and is intended only for offline demonstrations and tests.

The agent writes a report like:

```text
reports/market_report_2026-09-23_142300.md
```

## Architecture

```text
src/market_agent/
  agent.py       # Orchestrates the autonomous analysis workflow
  cli.py         # Command-line interface
  data.py        # Financial data provider and cache
  features.py    # Feature engineering and target creation
  model.py       # Lightweight ML model and metrics
  reporting.py   # Markdown report generation
```

## How The Model Works

The classifier predicts whether the next daily close will be higher than the current close. It uses engineered features from recent price and volume history, then outputs a probability score. The agent combines that probability with recent trend and volatility to produce a human-readable market signal.

This is not financial advice. It is an educational ML engineering project I came up with.

## My Resume Pitch

Built an autonomous market analysis agent in Python that ingests financial time series data, engineers trading features, trains a custom logistic regression model, evaluates predictive performance, and generates explainable market reports for multiple tickers.

## Future Improvements

- Add Alpha Vantage, Polygon, or Financial Modeling Prep provider support.
- Add portfolio-level risk allocation.
- Add backtesting and strategy comparison.
- Publish reports automatically with GitHub Actions.
- Add sentiment features from financial news.

# Finally (my little disclaimer)
QuantLens is a research and educational prototype, not a guaranteed stock prediction or automated trading system. Financial markets are noisy, and historical patterns do not guarantee future performance.
