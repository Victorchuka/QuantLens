import json
import unittest

from market_agent.agent import MarketAnalysisAgent
from market_agent.cli import build_parser
from market_agent.models import AnalysisConfig
from market_agent.providers import SampleDataProvider
from market_agent.reporting import to_json, to_markdown


class AgentTests(unittest.TestCase):
    def test_cli_uses_real_market_data_by_default(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.provider, "yfinance")

    def test_lookback_days_map_to_supported_yfinance_periods(self) -> None:
        self.assertEqual(MarketAnalysisAgent._period_for_lookback(365), "1y")
        self.assertEqual(MarketAnalysisAgent._period_for_lookback(730), "2y")
        self.assertEqual(MarketAnalysisAgent._period_for_lookback(2_000), "10y")

    def test_sample_provider_produces_reproducible_analysis(self) -> None:
        agent = MarketAnalysisAgent(SampleDataProvider(seed=11))
        report = agent.run(AnalysisConfig("DEMO", provider="sample", seed=11))

        self.assertEqual(report.ticker, "DEMO")
        self.assertEqual(report.observations, 504)
        self.assertEqual(report.prediction.model, "LogisticMarketModel")
        self.assertGreaterEqual(report.prediction.probability_up, 0)
        self.assertLessEqual(report.prediction.probability_up, 1)
        self.assertTrue(report.signals)

    def test_reports_are_serializable(self) -> None:
        report = MarketAnalysisAgent(SampleDataProvider()).run(AnalysisConfig("TEST", provider="sample"))

        payload = json.loads(to_json(report))
        self.assertEqual(payload["ticker"], "TEST")
        self.assertIn("# Market Analysis: TEST", to_markdown(report))


if __name__ == "__main__":
    unittest.main()
