from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path

from market_agent.data import MarketDataClient, PriceBar


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, ticker: str, period: str) -> list[PriceBar]:
        """Return daily OHLCV price bars."""


class StooqProvider(MarketDataProvider):
    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.client = MarketDataClient(cache_dir=Path(cache_dir))

    def fetch(self, ticker: str, period: str) -> list[PriceBar]:
        days = _period_to_calendar_days(period)
        end = date.today()
        start = end - timedelta(days=days)
        return self.client.get_daily_prices(ticker, start, end)


class YFinanceProvider(MarketDataProvider):
    """Download real historical daily OHLCV data through yfinance."""

    def fetch(self, ticker: str, period: str) -> list[PriceBar]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "The yfinance package is required for real market data. "
                "Install the project with: python3 -m pip install -e ."
            ) from exc

        frame = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if frame.empty:
            raise ValueError(
                f"No Yahoo Finance market data returned for ticker {ticker!r}. "
                "Check the symbol and internet connection."
            )
        if hasattr(frame.columns, "get_level_values") and getattr(frame.columns, "nlevels", 1) > 1:
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.reset_index()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

        bars: list[PriceBar] = []
        for row in frame.to_dict("records"):
            bars.append(
                PriceBar(
                    date=row["date"].date(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
            )
        return bars


class SampleDataProvider(MarketDataProvider):
    """Deterministic synthetic prices so demos and tests work without network access."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def fetch(self, ticker: str, period: str) -> list[PriceBar]:
        rows = {"1y": 252, "2y": 504, "5y": 1260}.get(period, 504)
        rng = random.Random(f"{self.seed}:{ticker}:{period}")
        today = date.today()
        start = today - timedelta(days=int(rows * 1.6) + 10)
        bars: list[PriceBar] = []
        close = 100.0
        current = start

        while current <= today:
            if current.weekday() < 5:
                seasonal = math.sin(len(bars) / 20) * 0.001
                daily_return = rng.gauss(0.00035 + seasonal, 0.012)
                close *= math.exp(daily_return)
                open_price = close * (1 + rng.gauss(0, 0.003))
                high = max(open_price, close) * (1 + rng.uniform(0.001, 0.02))
                low = min(open_price, close) * (1 - rng.uniform(0.001, 0.02))
                bars.append(
                    PriceBar(
                        date=current,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=rng.randint(1_000_000, 5_000_000),
                    )
                )
            current += timedelta(days=1)
        return bars[-rows:]


def _period_to_calendar_days(period: str) -> int:
    if period.endswith("y") and period[:-1].isdigit():
        return int(period[:-1]) * 365
    if period.endswith("d") and period[:-1].isdigit():
        return int(period[:-1])
    return 730
