from __future__ import annotations

import math
from dataclasses import dataclass

from market_agent.features import FeatureRow


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    samples: int

    @staticmethod
    def empty() -> "ModelMetrics":
        return ModelMetrics(accuracy=0.0, precision=0.0, recall=0.0, samples=0)


class LogisticMarketModel:
    """Small logistic-regression classifier implemented with standard Python."""

    def __init__(self, feature_names: list[str], learning_rate: float = 0.08, epochs: int = 600) -> None:
        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {name: 0.0 for name in feature_names}
        self.bias = 0.0
        self.means = {name: 0.0 for name in feature_names}
        self.stdevs = {name: 1.0 for name in feature_names}

    def fit(self, rows: list[FeatureRow]) -> None:
        if not rows:
            raise ValueError("Cannot train model with no rows.")
        self._fit_scaler(rows)

        for _ in range(self.epochs):
            for row in rows:
                scaled = self._scale(row.features)
                prediction = self._sigmoid(self._linear_score(scaled))
                error = prediction - row.target
                for name in self.feature_names:
                    self.weights[name] -= self.learning_rate * error * scaled[name]
                self.bias -= self.learning_rate * error

    def predict_probability(self, features: dict[str, float]) -> float:
        scaled = self._scale(features)
        return self._sigmoid(self._linear_score(scaled))

    def predict(self, features: dict[str, float]) -> int:
        return 1 if self.predict_probability(features) >= 0.5 else 0

    def evaluate(self, rows: list[FeatureRow]) -> ModelMetrics:
        if not rows:
            return ModelMetrics.empty()

        true_positive = false_positive = true_negative = false_negative = 0
        for row in rows:
            predicted = self.predict(row.features)
            if predicted == 1 and row.target == 1:
                true_positive += 1
            elif predicted == 1 and row.target == 0:
                false_positive += 1
            elif predicted == 0 and row.target == 0:
                true_negative += 1
            else:
                false_negative += 1

        total = len(rows)
        accuracy = (true_positive + true_negative) / total
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        return ModelMetrics(accuracy=accuracy, precision=precision, recall=recall, samples=total)

    def _fit_scaler(self, rows: list[FeatureRow]) -> None:
        for name in self.feature_names:
            values = [row.features[name] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            stdev = math.sqrt(variance)
            self.means[name] = mean
            self.stdevs[name] = stdev if stdev > 1e-12 else 1.0

    def _scale(self, features: dict[str, float]) -> dict[str, float]:
        return {
            name: (features[name] - self.means[name]) / self.stdevs[name]
            for name in self.feature_names
        }

    def _linear_score(self, features: dict[str, float]) -> float:
        return self.bias + sum(self.weights[name] * features[name] for name in self.feature_names)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1 / (1 + z)
        z = math.exp(value)
        return z / (1 + z)


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

