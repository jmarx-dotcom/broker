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


def share_equivalent(
    shares: float | None, market_cap: float | None, price: float | None
) -> float | None:
    """Aktienzahl, die zum gemeldeten Gesamtwert des Unternehmens passt.

    Bei Titeln mit Stamm- und Vorzugsaktien meldet Yahoo die
    Marktkapitalisierung des ganzen Unternehmens, `sharesOutstanding` aber nur
    für die abgefragte Gattung. Wer damit den Gewinn je Aktie ausrechnet, teilt
    den Gewinn des ganzen Unternehmens durch einen Bruchteil der Aktien und
    erhält ein Vielfaches des echten EPS — bei VW etwa das 2,4-fache.

    Für die historische KGV-Reihe ist das fatal: Sie fiele um denselben Faktor
    zu niedrig aus, das aktuelle KGV läge scheinbar über ihrem gesamten
    Verlauf, und der Titel bekäme im Perzentil-Vergleich null Punkte, obwohl er
    günstig ist.

    Marktkapitalisierung geteilt durch Kurs ergibt die Zahl der Aktien dieser
    Gattung, die das ganze Unternehmen wert wäre — und passt damit zum
    Gesamtgewinn. Liegt sie nicht deutlich über der gemeldeten Aktienzahl,
    bleibt es bei dieser.
    """
    if not market_cap or not price or price <= 0:
        return shares
    implied = market_cap / price
    if shares and implied <= shares * 1.15:
        return shares
    return implied


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
        Vergleich reicht es. Bei mehreren Aktiengattungen muss die Aktienzahl
        zum Gesamtgewinn passen, siehe `share_equivalent`.
        """

        def fetch() -> pd.Series | None:
            try:
                stock = self._ticker(ticker)
                stmt = stock.quarterly_income_stmt
                info = stock.info
            except Exception as exc:
                log.debug("Quartalszahlen für %s nicht verfügbar: %s", ticker, exc)
                return None

            shares = share_equivalent(
                _num(info.get("sharesOutstanding")),
                _num(info.get("marketCap")),
                _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice")),
            )

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

    def _statements(self, ticker: str) -> dict:
        """Liest Bilanz, GuV und Aktienzahl aus den Abschlüssen.

        Yahoo benennt die Zeilen nicht einheitlich und ändert die Bezeichnungen
        gelegentlich. Deshalb wird für jede Kennzahl eine Liste plausibler
        Bezeichnungen durchprobiert — fehlt eine, bleibt das Feld leer, statt
        den ganzen Titel scheitern zu lassen. Bei kleinen Nebenwerten sind
        diese Daten oft lückenhaft; der Screener bestraft das nicht.
        """

        def fetch() -> dict:
            try:
                stock = self._ticker(ticker)
                income = stock.income_stmt
                balance = stock.balance_sheet
                quarterly_income = stock.quarterly_income_stmt
            except Exception as exc:
                log.debug("Abschlüsse für %s nicht verfügbar: %s", ticker, exc)
                return {}

            def latest(frame, *labels: str) -> float | None:
                if frame is None or getattr(frame, "empty", True):
                    return None
                for label in labels:
                    if label in frame.index:
                        series = frame.loc[label].dropna()
                        if not series.empty:
                            return _num(series.iloc[0])
                return None

            def row(frame, *labels: str):
                if frame is None or getattr(frame, "empty", True):
                    return None
                for label in labels:
                    if label in frame.index:
                        series = pd.Series(frame.loc[label]).dropna().astype(float)
                        if not series.empty:
                            series.index = pd.to_datetime(series.index)
                            return series.sort_index()
                return None

            pretax = latest(income, "Pretax Income", "Income Before Tax")
            tax = latest(income, "Tax Provision", "Income Tax Expense")
            tax_rate = None
            if pretax and tax is not None and pretax > 0:
                tax_rate = max(0.0, min(0.6, tax / pretax))

            return {
                "ebit": latest(income, "EBIT", "Operating Income", "Total Operating Income As Reported"),
                "interest_expense": latest(
                    income, "Interest Expense", "Interest Expense Non Operating",
                    "Net Interest Income",
                ),
                "tax_rate": tax_rate,
                "current_assets": latest(balance, "Current Assets", "Total Current Assets"),
                "current_liabilities": latest(
                    balance, "Current Liabilities", "Total Current Liabilities"
                ),
                "total_equity": latest(
                    balance, "Stockholders Equity", "Total Equity Gross Minority Interest",
                    "Common Stock Equity",
                ),
                "quarterly_revenue": row(quarterly_income, "Total Revenue", "Operating Revenue"),
                "quarterly_net_income": row(
                    quarterly_income, "Net Income", "Net Income Common Stockholders"
                ),
                "shares_history": row(
                    balance, "Ordinary Shares Number", "Share Issued", "Common Stock"
                ),
            }

        return self.cache.get_or_compute("yf_stmt", cache_key(ticker), fetch)

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
            **self._statements(ticker),
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
