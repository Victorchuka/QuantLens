from __future__ import annotations

import math
from datetime import date

from .data import PriceBar
from .features import build_dataset, latest_feature_row
from .model import LogisticMarketModel
from .models import AnalysisReport, Prediction


def analyze(ticker: str, prices: list[PriceBar], seed: int = 7) -> AnalysisReport:
    dataset = build_dataset(prices)
    if len(dataset.rows) < 80:
        raise ValueError("At least 80 usable observations are required for analysis.")

    split = max(60, int(len(dataset.rows) * 0.8))
    train = dataset.rows[:split]
    model = LogisticMarketModel(dataset.feature_names, learning_rate=0.08, epochs=500)
    model.fit(train)
    latest = latest_feature_row(prices, dataset.feature_names)
    probability = model.predict_probability(latest.features)
    direction = "up" if probability >= 0.5 else "down"

    closes = [bar.close for bar in prices]
    daily_returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
    peak = max(closes)
    sma_20 = _mean(closes[-20:])
    sma_50 = _mean(closes[-50:])
    trend = "bullish" if closes[-1] > sma_20 > sma_50 else "bearish" if closes[-1] < sma_20 < sma_50 else "mixed"
    rsi = _rsi(closes[-15:])

    signals = []
    if trend == "bullish":
        signals.append("Price is above its 20-day and 50-day averages.")
    elif trend == "bearish":
        signals.append("Price is below its 20-day and 50-day averages.")
    if rsi >= 70:
        signals.append("RSI suggests overbought momentum.")
    elif rsi <= 30:
        signals.append("RSI suggests oversold momentum.")
    else:
        signals.append("RSI is in a neutral range.")
    signals.append(f"Model assigns {probability:.0%} probability to an up day next session.")

    return AnalysisReport(
        ticker=ticker.upper(),
        generated_at=date.today(),
        data_start=prices[0].date,
        data_end=prices[-1].date,
        observations=len(prices),
        last_close=float(closes[-1]),
        return_1d=float(daily_returns[-1]),
        return_period=float(closes[-1] / closes[0] - 1),
        volatility_annualized=float(_stddev(daily_returns) * math.sqrt(252)),
        drawdown_from_peak=float(closes[-1] / peak - 1),
        trend=trend,
        prediction=Prediction(direction, probability, "LogisticMarketModel", len(train)),
        signals=tuple(signals),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _rsi(closes: list[float]) -> float:
    gains = []
    losses = []
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    average_gain = _mean(gains)
    average_loss = _mean(losses)
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))
