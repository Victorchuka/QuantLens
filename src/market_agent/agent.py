from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import overload

from market_agent.analysis import analyze
from market_agent.data import MarketDataClient, PriceBar
from market_agent.features import build_dataset, latest_feature_row
from market_agent.model import LogisticMarketModel, ModelMetrics
from market_agent.models import AnalysisConfig, AnalysisReport
from market_agent.providers import MarketDataProvider, SampleDataProvider, YFinanceProvider
from market_agent.reporting import AnalysisResult, write_report


@dataclass(frozen=True)
class AgentConfig:
    symbols: list[str]
    lookback_days: int
    risk_profile: str
    provider: str
    cache_dir: Path
    reports_dir: Path


class MarketAnalysisAgent:
    """Coordinates the autonomous fetch -> learn -> score -> report workflow."""

    def __init__(self, config: AgentConfig | MarketDataProvider) -> None:
        self.config: AgentConfig | None = None
        self.provider: MarketDataProvider | None = None
        self.data_client: MarketDataClient | None = None

        if isinstance(config, AgentConfig):
            self.config = config
            self.data_client = MarketDataClient(config.cache_dir)
        else:
            self.provider = config

    @overload
    def run(self) -> Path:
        ...

    @overload
    def run(self, config: AnalysisConfig) -> AnalysisReport:
        ...

    def run(self, config: AnalysisConfig | None = None) -> Path | AnalysisReport:
        if config is not None:
            if self.provider is None:
                raise ValueError("A MarketDataProvider is required when running with AnalysisConfig.")
            prices = self.provider.fetch(config.ticker, config.period)
            return analyze(config.ticker, prices, seed=config.seed)

        if self.config is None or self.data_client is None:
            raise ValueError("AgentConfig is required for report generation mode.")

        end = date.today()
        start = end - timedelta(days=self.config.lookback_days)

        results: list[AnalysisResult] = []
        for symbol in self.config.symbols:
            if self.config.provider == "sample":
                bars = SampleDataProvider(seed=7).fetch(symbol, "2y")
            elif self.config.provider == "yfinance":
                period = self._period_for_lookback(self.config.lookback_days)
                bars = YFinanceProvider().fetch(symbol, period)
            else:
                bars = self.data_client.get_daily_prices(symbol, start, end)
            result = self._analyze_symbol(symbol, bars)
            results.append(result)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_path = self.config.reports_dir / f"market_report_{timestamp}.md"
        write_report(
            report_path,
            results,
            self.config.risk_profile,
            start,
            end,
            data_source=self.config.provider,
        )
        return report_path

    @staticmethod
    def _period_for_lookback(lookback_days: int) -> str:
        if lookback_days <= 365:
            return "1y"
        if lookback_days <= 730:
            return "2y"
        if lookback_days <= 1_825:
            return "5y"
        if lookback_days <= 3_650:
            return "10y"
        return "max"

    def _analyze_symbol(self, symbol: str, bars: list[PriceBar]) -> AnalysisResult:
        dataset = build_dataset(bars)
        if len(dataset.rows) < 40:
            raise ValueError(
                f"{symbol} has only {len(dataset.rows)} usable rows after feature engineering. "
                "Try increasing --lookback-days."
            )

        split_index = max(1, int(len(dataset.rows) * 0.75))
        train_rows = dataset.rows[:split_index]
        test_rows = dataset.rows[split_index:]

        model = LogisticMarketModel(feature_names=dataset.feature_names)
        model.fit(train_rows)
        metrics = model.evaluate(test_rows) if test_rows else ModelMetrics.empty()

        latest = latest_feature_row(bars, dataset.feature_names)
        probability = model.predict_probability(latest.features)
        last_bar = bars[-1]
        signal, rationale = self._signal_from_probability(probability, latest.features)

        return AnalysisResult(
            symbol=symbol,
            last_close=last_bar.close,
            last_date=last_bar.date,
            probability_up=probability,
            signal=signal,
            rationale=rationale,
            metrics=metrics,
            feature_snapshot=latest.features,
        )

    def _signal_from_probability(self, probability: float, features: dict[str, float]) -> tuple[str, str]:
        risk_profile = self.config.risk_profile.lower()
        buy_threshold = {"conservative": 0.62, "balanced": 0.57, "aggressive": 0.53}[risk_profile]
        sell_threshold = {"conservative": 0.42, "balanced": 0.45, "aggressive": 0.47}[risk_profile]

        trend = features.get("sma_5_over_20", 0.0)
        volatility = features.get("volatility_10", 0.0)

        if probability >= buy_threshold and trend >= -0.02:
            return (
                "BUY",
                f"Model probability is above the {risk_profile} buy threshold with acceptable trend confirmation.",
            )
        if probability >= buy_threshold:
            return (
                "HOLD",
                "Model probability is strong, but recent trend confirmation is still weak.",
            )
        if probability <= sell_threshold:
            return (
                "SELL",
                f"Model probability is below the {risk_profile} sell threshold, indicating weak near-term setup.",
            )
        if volatility > 0.05:
            return (
                "HOLD",
                "Signal stayed neutral because recent volatility is elevated relative to the model score.",
            )
        return (
            "HOLD",
            "Model score is not strong enough to justify a directional signal.",
        )
