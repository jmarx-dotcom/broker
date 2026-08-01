"""FRED-Client für die volkswirtschaftlichen Zeitreihen.

FRED (Federal Reserve Bank of St. Louis) ist kostenlos und deckt sowohl die
US- als auch — über die von der OECD und EZB gespiegelten Reihen — die
europäischen Kernindikatoren ab. Der Key ist in 30 Sekunden angelegt:
https://fredaccount.stlouisfed.org/apikeys
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import requests

from broker.models import MacroSeries
from broker.providers.cache import DayCache, cache_key

log = logging.getLogger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    fred_id: str
    label: str
    unit: str
    #: True, wenn der Wert bereits in Prozent notiert (z. B. 4.25 = 4,25 %).
    percent: bool = True


#: Die Reihen, aus denen das Makrobild gebaut wird.
SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("fed_funds", "DFF", "US-Leitzins (effektiv)", "%"),
    SeriesSpec("us_10y", "DGS10", "US-Staatsanleihe 10J", "%"),
    SeriesSpec("us_2y", "DGS2", "US-Staatsanleihe 2J", "%"),
    SeriesSpec("yield_curve", "T10Y2Y", "Renditekurve 10J–2J", "%"),
    SeriesSpec("us_cpi", "CPIAUCSL", "US-Verbraucherpreise", "Index", percent=False),
    SeriesSpec("us_unemployment", "UNRATE", "US-Arbeitslosenquote", "%"),
    SeriesSpec("ez_10y", "IRLTLT01EZM156N", "Euroraum-Anleihe 10J", "%"),
    SeriesSpec("ez_cpi", "CP0000EZ19M086NEST", "Euroraum-Verbraucherpreise", "Index", percent=False),
    SeriesSpec("oil_brent", "DCOILBRENTEU", "Ölpreis Brent", "USD", percent=False),
    SeriesSpec("eur_usd", "DEXUSEU", "Wechselkurs EUR/USD", "", percent=False),
    SeriesSpec("vix", "VIXCLS", "Volatilitätsindex VIX", "", percent=False),
    # Konjunktur: Industrieproduktion reagiert schneller als das BIP, das
    # Verbrauchervertrauen noch etwas früher. Das BIP ist träge, taugt aber
    # als Bestätigung.
    SeriesSpec("industrial_production", "INDPRO", "US-Industrieproduktion", "Index", percent=False),
    SeriesSpec("gdp", "GDPC1", "US-BIP (real)", "Mrd. USD", percent=False),
    SeriesSpec("consumer_sentiment", "UMCSENT", "US-Verbrauchervertrauen", "Index", percent=False),
    # Inflationserwartung statt nur gemessener Inflation: Die Breakeven-Rate
    # sagt, womit der Markt rechnet — für Aktien relevanter als der Rückblick.
    SeriesSpec("inflation_expectation", "T10YIE", "Inflationserwartung 10J", "%"),
    # Risikoaufschlag für Hochzinsanleihen. Der zuverlässigste Stressindikator
    # überhaupt: Er steigt, bevor Aktien fallen, und erfasst geopolitische
    # Schocks mit, ohne dass man sie einzeln modellieren muss.
    SeriesSpec("high_yield_spread", "BAMLH0A0HYM2", "Risikoaufschlag Hochzinsanleihen", "%"),
)


class FredClient:
    def __init__(
        self,
        api_key: str | None,
        cache: DayCache | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.cache = cache or DayCache("cache", enabled=False)
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _observations(self, fred_id: str, days: int = 500) -> list[tuple[date, float]]:
        """Beobachtungen der letzten `days` Tage, aufsteigend sortiert."""

        def fetch() -> list[tuple[date, float]]:
            params = {
                "series_id": fred_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": (date.today() - timedelta(days=days)).isoformat(),
            }
            response = requests.get(FRED_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()

            result: list[tuple[date, float]] = []
            for obs in payload.get("observations", []):
                raw = obs.get("value")
                if raw in (None, "", "."):  # FRED markiert Lücken mit "."
                    continue
                try:
                    value = float(raw)
                    stamp = datetime.strptime(obs["date"], "%Y-%m-%d").date()
                except (ValueError, KeyError):
                    continue
                result.append((stamp, value))
            return result

        return self.cache.get_or_compute("fred", cache_key(fred_id, days), fetch)

    def fetch_series(self, spec: SeriesSpec) -> MacroSeries | None:
        """Holt eine Reihe und berechnet die 3- und 12-Monats-Veränderung."""
        if not self.available:
            return None
        try:
            observations = self._observations(spec.fred_id)
        except Exception as exc:
            log.warning("FRED-Reihe %s nicht abrufbar: %s", spec.fred_id, exc)
            return None

        if not observations:
            return None

        as_of, current = observations[-1]

        def value_near(target: date) -> float | None:
            """Letzte Beobachtung am oder vor dem Zieldatum."""
            candidates = [v for d, v in observations if d <= target]
            return candidates[-1] if candidates else None

        past_3m = value_near(as_of - timedelta(days=91))
        past_12m = value_near(as_of - timedelta(days=365))

        def delta(past: float | None) -> float | None:
            if past is None:
                return None
            if spec.percent:
                return current - past  # Prozentpunkte
            if past == 0:
                return None
            return (current / past) - 1.0  # relative Veränderung

        return MacroSeries(
            key=spec.key,
            label=spec.label,
            value=current,
            change_3m=delta(past_3m),
            change_12m=delta(past_12m),
            unit=spec.unit,
            as_of=as_of,
        )

    def fetch_all(self) -> dict[str, MacroSeries]:
        result: dict[str, MacroSeries] = {}
        for spec in SERIES:
            series = self.fetch_series(spec)
            if series is not None:
                result[spec.key] = series

        # Scheitert *jede* Reihe, liegt es fast nie an den Reihen selbst,
        # sondern am Key. Der ist im Log maskiert und sieht dort immer richtig
        # aus — deshalb hier ein konkreter Hinweis statt elf gleicher Warnungen.
        if self.available and not result:
            log.error(
                "Keine einzige FRED-Reihe abrufbar. Bei durchgängig 400er-Fehlern "
                "ist der API-Key das Problem: er muss aus genau 32 Zeichen "
                "bestehen (Kleinbuchstaben und Ziffern). Anführungszeichen oder "
                "angehängte Leerzeichen führen zu genau diesem Bild."
            )
        return result
