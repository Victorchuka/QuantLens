from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    ticker: str
    period: str = "2y"
    provider: str = "yfinance"
    lookback: int = 20
    seed: int = 7


@dataclass(frozen=True)
class Prediction:
    direction: str
    probability_up: float
    model: str
    training_rows: int


@dataclass(frozen=True)
class AnalysisReport:
    ticker: str
    generated_at: date
    data_start: date
    data_end: date
    observations: int
    last_close: float
    return_1d: float
    return_period: float
    volatility_annualized: float
    drawdown_from_peak: float
    trend: str
    prediction: Prediction
    signals: tuple[str, ...]
    caveat: str = "For research and education only; not financial advice."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)