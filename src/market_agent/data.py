from __future__ import annotations

import csv
import io
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class PriceBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataClient:
    """Fetches and caches daily market data from Stooq's public CSV endpoint."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[PriceBar]:
        cached = self._read_cache(symbol, start, end)
        if cached:
            return cached

        bars = self._fetch_from_stooq(symbol, start, end)
        if not bars:
            raise ValueError(f"No market data returned for {symbol}. Check the symbol and date range.")
        self._write_cache(symbol, start, end, bars)
        return bars

    def _fetch_from_stooq(self, symbol: str, start: date, end: date) -> list[PriceBar]:
        stooq_symbol = self._to_stooq_symbol(symbol)
        url = (
            "https://stooq.com/q/d/l/"
            f"?s={stooq_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "market-analysis-agent/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise ConnectionError(
                    "Could not verify Stooq's HTTPS certificate. On macOS, run Python's "
                    "'Install Certificates.command' or update your certificate store, then try again."
                ) from exc
            raise ConnectionError(
                "Could not download financial data from Stooq. "
                "Check your internet connection or try again later."
            ) from exc
        return parse_price_csv(payload)

    def _cache_path(self, symbol: str, start: date, end: date) -> Path:
        clean_symbol = symbol.upper().replace("/", "-")
        return self.cache_dir / f"{clean_symbol}_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    def _read_cache(self, symbol: str, start: date, end: date) -> list[PriceBar]:
        path = self._cache_path(symbol, start, end)
        if not path.exists():
            return []
        return parse_price_csv(path.read_text(encoding="utf-8"))

    def _write_cache(self, symbol: str, start: date, end: date, bars: list[PriceBar]) -> None:
        path = self._cache_path(symbol, start, end)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
            for bar in bars:
                writer.writerow([bar.date.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume])

    @staticmethod
    def _to_stooq_symbol(symbol: str) -> str:
        symbol = symbol.lower()
        if "." in symbol:
            return symbol
        return f"{symbol}.us"


def parse_price_csv(payload: str) -> list[PriceBar]:
    reader = csv.DictReader(io.StringIO(payload))
    bars: list[PriceBar] = []
    for row in reader:
        if not row or row.get("Close") in (None, "", "N/D"):
            continue
        bars.append(
            PriceBar(
                date=datetime.strptime(row["Date"], "%Y-%m-%d").date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(float(row["Volume"])),
            )
        )
    bars.sort(key=lambda item: item.date)
    return bars
