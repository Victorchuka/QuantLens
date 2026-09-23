from __future__ import annotations

import unittest

from market_agent.features import FeatureRow
from market_agent.model import LogisticMarketModel


class ModelTests(unittest.TestCase):
    def test_model_learns_simple_signal(self) -> None:
        rows = [
            FeatureRow(features={"momentum": -1.0}, target=0),
            FeatureRow(features={"momentum": -0.8}, target=0),
            FeatureRow(features={"momentum": 0.8}, target=1),
            FeatureRow(features={"momentum": 1.0}, target=1),
        ]
        model = LogisticMarketModel(["momentum"], learning_rate=0.1, epochs=300)

        model.fit(rows)

        self.assertLess(model.predict_probability({"momentum": -1.0}), 0.5)
        self.assertGreater(model.predict_probability({"momentum": 1.0}), 0.5)


if __name__ == "__main__":
    unittest.main()

