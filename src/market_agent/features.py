from __future__ import annotations

import math
from dataclasses import dataclass

from market_agent.data import PriceBar


@dataclass(frozen=True)
class FeatureRow:
    features: dict[str, float]
    target: int


@dataclass(frozen=True)
class FeatureDataset:
    feature_names: list[str]
    rows: list[FeatureRow]


FEATURE_NAMES = [
    "return_1",
    "return_5",
    "return_10",
    "sma_5_over_20",
    "volatility_10",
    "volume_ratio_5_20",
    "drawdown_20",
]


def build_dataset(bars: list[PriceBar]) -> FeatureDataset:
    rows: list[FeatureRow] = []
    for index in range(20, len(bars) - 1):
        features = _features_at(bars, index)
        target = 1 if bars[index + 1].close > bars[index].close else 0
        rows.append(FeatureRow(features=features, target=target))
    return FeatureDataset(feature_names=list(FEATURE_NAMES), rows=rows)


def latest_feature_row(bars: list[PriceBar], feature_names: list[str]) -> FeatureRow:
    if len(bars) < 21:
        raise ValueError("At least 21 price bars are required for latest feature extraction.")
    features = _features_at(bars, len(bars) - 1)
    ordered = {name: features[name] for name in feature_names}
    return FeatureRow(features=ordered, target=0)


def _features_at(bars: list[PriceBar], index: int) -> dict[str, float]:
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]

    close = closes[index]
    sma_5 = _mean(closes[index - 4 : index + 1])
    sma_20 = _mean(closes[index - 19 : index + 1])
    vol_5 = _mean(volumes[index - 4 : index + 1])
    vol_20 = _mean(volumes[index - 19 : index + 1])
    high_20 = max(closes[index - 19 : index + 1])
    returns_10 = [_safe_return(closes[i - 1], closes[i]) for i in range(index - 9, index + 1)]

    return {
        "return_1": _safe_return(closes[index - 1], close),
        "return_5": _safe_return(closes[index - 5], close),
        "return_10": _safe_return(closes[index - 10], close),
        "sma_5_over_20": _safe_ratio(sma_5, sma_20) - 1.0,
        "volatility_10": _stddev(returns_10),
        "volume_ratio_5_20": _safe_ratio(vol_5, vol_20) - 1.0,
        "drawdown_20": _safe_ratio(close, high_20) - 1.0,
    }


def _safe_return(previous: float, current: float) -> float:
    return _safe_ratio(current, previous) - 1.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)

