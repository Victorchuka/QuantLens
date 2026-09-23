from __future__ import annotations

import unittest
from datetime import date, timedelta

from market_agent.data import PriceBar
from market_agent.features import FEATURE_NAMES, build_dataset, latest_feature_row


class FeatureTests(unittest.TestCase):
    def test_build_dataset_creates_expected_rows(self) -> None:
        bars = make_bars(30)

        dataset = build_dataset(bars)

        self.assertEqual(dataset.feature_names, FEATURE_NAMES)
        self.assertEqual(len(dataset.rows), 9)
        self.assertTrue(set(FEATURE_NAMES).issubset(dataset.rows[0].features))

    def test_latest_feature_row_uses_requested_order(self) -> None:
        bars = make_bars(30)
        names = ["return_1", "drawdown_20"]

        row = latest_feature_row(bars, names)

        self.assertEqual(list(row.features), names)


def make_bars(count: int) -> list[PriceBar]:
    start = date(2025, 1, 1)
    bars: list[PriceBar] = []
    for index in range(count):
        close = 100 + index
        bars.append(
            PriceBar(
                date=start + timedelta(days=index),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000_000 + index * 1000,
            )
        )
    return bars


if __name__ == "__main__":
    unittest.main()

