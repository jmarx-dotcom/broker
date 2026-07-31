"""yfinance-Provider.

yfinance ist eine inoffizielle Schnittstelle zu Yahoo Finance: kostenlos, aber
mit Lücken (besonders bei Forward-EPS europäischer Nebenwerte) und gelegentlichen
Ausfällen. Für den produktiven Dauerbetrieb ist ein Bezahlanbieter die bessere
Wahl — siehe providers/base.py.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from broker.models import Fundamentals, NewsItem, PriceHistory
from broker.providers.base import ProviderError
from broker.providers.cache import DayCache, cache_key

log = logging.getLogger(__name__)


def _num(value: object) -> float | None:
    """Wandelt Yahoo-Werte in float um. Yahoo liefert gern 'Infinity' oder None."""
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


class YahooProvider:
    name = "yfinance"

    def __init__(self, cache: DayCache | None = None) -> None:
        self.cache = cache or DayCache("cache", enabled=False)

    # -- interne Helfer ---------------------------------------------------

    def _ticker(self, ticker: str):
        import yfinance as yf

        return yf.Ticker(ticker)

    def _info(self, ticker: str) -> dict:
        def fetch() -> dict:
            try:
                info = self._ticker(ticker).info
            except Exception as exc:
                raise ProviderError(f"info({ticker}) fehlgeschlagen: {exc}") from exc
            if not isinstance(info, dict) or not info:
                raise ProviderError(f"info({ticker}) lieferte keine Daten")
            return info

        return self.cache.get_or_compute("yf_info", cache_key(ticker), fetch)

    def _quarterly_eps(self, ticker: str) -> pd.Series | None:
        """Quartals-EPS aus der GuV — Basis für die historische KGV-Reihe.

        Yahoo liefert keine EPS-Zeitreihe direkt, aber Nettogewinn und
        Aktienanzahl. EPS = Nettogewinn / Aktien, mit der heutigen Aktienzahl
        als Näherung — Rückkäufe verzerren das leicht, für einen Perzentil-
        Vergleich reicht es.
        """

        def fetch() -> pd.Series | None:
            try:
                stock = self._ticker(ticker)
                stmt = stock.quarterly_income_stmt
                shares = _num(stock.info.get("sharesOutstanding"))
            except Exception as exc:
                log.debug("Quartalszahlen für %s nicht verfügbar: %s", ticker, exc)
                return None

            if stmt is None or getattr(stmt, "empty", True) or not shares:
                return None

            row = None
            for label in ("Diluted EPS", "Basic EPS"):
                if label in stmt.index:
                    row = stmt.loc[label]
                    break

            if row is None:
                for label in ("Net Income", "Net Income Common Stockholders"):
                    if label in stmt.index:
                        row = stmt.loc[label] / shares
                        break

            if row is None:
                return None

            series = pd.Series(row).dropna().astype(float)
            if series.empty:
                return None
            series.index = pd.to_datetime(series.index)
            return series.sort_index()

        return self.cache.get_or_compute("yf_eps", cache_key(ticker), fetch)

    # -- Provider-Schnittstelle -------------------------------------------

    def fundamentals(self, ticker: str) -> Fundamentals:
        info = self._info(ticker)
        dividend_yield = _num(info.get("dividendYield"))
        # Yahoo liefert die Dividendenrendite mal als 0.031, mal als 3.1.
        if dividend_yield is not None and dividend_yield > 1:
            dividend_yield /= 100.0

        return Fundamentals(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            currency=info.get("currency"),
            market_cap=_num(info.get("marketCap")),
            shares_outstanding=_num(info.get("sharesOutstanding")),
            trailing_eps=_num(info.get("trailingEps")),
            forward_eps=_num(info.get("forwardEps")),
            trailing_pe=_num(info.get("trailingPE")),
            forward_pe=_num(info.get("forwardPE")),
            price_to_book=_num(info.get("priceToBook")),
            dividend_yield=dividend_yield,
            payout_ratio=_num(info.get("payoutRatio")),
            total_debt=_num(info.get("totalDebt")),
            total_cash=_num(info.get("totalCash")),
            ebitda=_num(info.get("ebitda")),
            free_cashflow=_num(info.get("freeCashflow")),
            revenue_growth=_num(info.get("revenueGrowth")),
            earnings_growth=_num(info.get("earningsGrowth"))
            or _num(info.get("earningsQuarterlyGrowth")),
            return_on_equity=_num(info.get("returnOnEquity")),
            profit_margin=_num(info.get("profitMargins")),
            quarterly_eps=self._quarterly_eps(ticker),
        )

    def history(self, ticker: str, period: str = "3y") -> PriceHistory:
        def fetch() -> pd.DataFrame:
            try:
                frame = self._ticker(ticker).history(period=period, auto_adjust=True)
            except Exception as exc:
                raise ProviderError(f"history({ticker}) fehlgeschlagen: {exc}") from exc
            if frame is None or frame.empty:
                raise ProviderError(f"history({ticker}) lieferte keine Daten")
            return frame

        frame = self.cache.get_or_compute("yf_hist", cache_key(ticker, period), fetch)
        return PriceHistory(ticker=ticker, frame=frame)

    def news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        def fetch() -> list[NewsItem]:
            try:
                raw = self._ticker(ticker).news or []
            except Exception as exc:
                log.debug("News für %s nicht verfügbar: %s", ticker, exc)
                return []

            items: list[NewsItem] = []
            for entry in raw[:limit]:
                # yfinance hat das Newsformat mehrfach geändert; beide Varianten.
                content = entry.get("content") if isinstance(entry, dict) else None
                if isinstance(content, dict):
                    title = content.get("title")
                    publisher = (content.get("provider") or {}).get("displayName")
                    url = (content.get("canonicalUrl") or {}).get("url")
                    summary = content.get("summary")
                    published = _parse_date(content.get("pubDate"))
                else:
                    title = entry.get("title")
                    publisher = entry.get("publisher")
                    url = entry.get("link")
                    summary = None
                    published = _parse_epoch(entry.get("providerPublishTime"))

                if title:
                    items.append(
                        NewsItem(
                            title=title,
                            publisher=publisher,
                            published=published,
                            url=url,
                            summary=summary,
                        )
                    )
            return items

        return self.cache.get_or_compute("yf_news", cache_key(ticker, limit), fetch)


def _parse_epoch(value: object) -> date | None:
    ts = _num(value)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts).date()
    except (OverflowError, OSError, ValueError):
        return None


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None
