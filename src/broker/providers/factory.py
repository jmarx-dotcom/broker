"""Wählt den Provider anhand der Konfiguration."""

from __future__ import annotations

from broker.config import Config
from broker.providers.base import MarketDataProvider
from broker.providers.cache import DayCache


def get_provider(config: Config, use_cache: bool = True) -> MarketDataProvider:
    cache = DayCache(config.cache_dir, enabled=use_cache)
    name = config.provider.lower()

    if name in ("yfinance", "yahoo"):
        from broker.providers.yahoo import YahooProvider

        return YahooProvider(cache=cache)

    if name == "fmp":
        raise NotImplementedError(
            "Der FMP-Provider ist noch nicht implementiert. Die Schnittstelle "
            "steht in providers/base.py — eine Implementierung braucht nur "
            "fundamentals(), history() und news(). Bis dahin: BROKER_PROVIDER=yfinance"
        )

    raise ValueError(f"Unbekannter Provider: {config.provider!r}")
