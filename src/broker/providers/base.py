"""Provider-Schnittstelle.

Alles, was das Tool an Marktdaten braucht, geht durch dieses Protokoll. Damit
ist der Wechsel von yfinance auf einen Bezahlanbieter (FMP, EODHD) eine neue
Datei in diesem Verzeichnis und eine Zeile in der Konfiguration — der Rest des
Codes bleibt unberührt.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from broker.models import Fundamentals, NewsItem, PriceHistory


class ProviderError(RuntimeError):
    """Ein Datenabruf ist fehlgeschlagen. Der Screener überspringt den Titel."""


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def fundamentals(self, ticker: str) -> Fundamentals:
        """Kennzahlen zu einem Titel. Fehlende Felder bleiben None."""
        ...

    def history(self, ticker: str, period: str = "3y") -> PriceHistory:
        """Tägliche Kurshistorie."""
        ...

    def news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        """Aktuelle Meldungen. Darf leer sein."""
        ...
