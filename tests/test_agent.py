import json
import unittest

from market_agent.agent import MarketAnalysisAgent
from market_agent.models import AnalysisConfig
from market_agent.providers import SampleDataProvider
from market_agent.reporting import to_json, to_markdown


class AgentTests(unittest.TestCase):
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
