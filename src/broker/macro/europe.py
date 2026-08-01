"""Europäische Konjunkturdaten von Eurostat und der EZB.

Das Makromodul war bislang stark US-lastig — Leitzins, Arbeitsmarkt und
Industrieproduktion kamen alle aus FRED, obwohl mehr als die Hälfte des
Universums europäisch notiert. Diese beiden Quellen schließen die Lücke:

* **Eurostat** liefert BIP, Industrieproduktion, Arbeitslosenquote und
  Verbraucherpreise für den Euroraum. JSON-Format, keine Registrierung.
* **Das EZB Data Portal** liefert Leitzins, Renditekurve und Wechselkurse.
  CSV-Format, ebenfalls ohne Anmeldung.

Beide brauchen **keinen API-Key**. Fällt eine Reihe aus, fehlt sie im
Makrobild — der Lauf geht weiter, wie bei FRED auch.

Ein Vorbehalt: Die Reihenkennungen unten konnten in dieser Umgebung nicht
gegen die Live-Schnittstellen geprüft werden, weil kein Netzzugang bestand.
Sie folgen der dokumentierten Systematik beider Anbieter; sollte eine davon
nicht stimmen, erscheint sie als Warnung im Log und die übrigen laufen weiter.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime

import requests

from broker.models import MacroSeries
from broker.providers.cache import DayCache, cache_key

log = logging.getLogger(__name__)

EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
)
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"


@dataclass(frozen=True)
class EurostatSpec:
    key: str
    dataset: str
    label: str
    unit: str
    #: Filter, damit nur die Zeitachse variiert — sonst enthält die Antwort
    #: mehrere Reihen nebeneinander.
    filters: dict[str, str]
    percent: bool = False
    #: Ersatzfiltersätze, falls der erste nichts liefert. Eurostat verwendet
    #: je Datensatz eigene Codelisten: derselbe Begriff heißt mal TOTAL, mal
    #: Y_GE15, und der Euroraum ist je nach Alter der Reihe EA20 oder EA19.
    #: Ein falscher Wert erzeugt keinen Fehler, sondern eine leere Antwort.
    fallbacks: tuple[dict[str, str], ...] = ()

    @property
    def filter_sets(self) -> tuple[dict[str, str], ...]:
        return (self.filters, *self.fallbacks)


@dataclass(frozen=True)
class EcbSpec:
    key: str
    flow: str
    series_key: str
    label: str
    unit: str
    percent: bool = True


EUROSTAT_SERIES: tuple[EurostatSpec, ...] = (
    EurostatSpec(
        "ez_industrial_production", "sts_inpr_m", "Euroraum-Industrieproduktion",
        "Index",
        {"geo": "EA20", "s_adj": "SCA", "nace_r2": "B-D", "unit": "I21"},
    ),
    EurostatSpec(
        "ez_unemployment", "une_rt_m", "Euroraum-Arbeitslosenquote", "%",
        {"geo": "EA20", "s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT"},
        percent=True,
        fallbacks=(
            # Der Euroraum hieß bis 2023 EA19; ältere Reihen führen ihn so.
            {"geo": "EA19", "s_adj": "SA", "age": "TOTAL", "sex": "T",
             "unit": "PC_ACT"},
            # Manche Datensätze kennen kein TOTAL, sondern nur Altersbänder.
            {"geo": "EA20", "s_adj": "SA", "age": "Y_GE15", "sex": "T",
             "unit": "PC_ACT"},
            {"geo": "EA20", "s_adj": "SA", "age": "Y25-74", "sex": "T",
             "unit": "PC_ACT"},
        ),
    ),
    EurostatSpec(
        "ez_gdp", "namq_10_gdp", "Euroraum-BIP (real)", "Index",
        {"geo": "EA20", "s_adj": "SCA", "unit": "CLV_I15", "na_item": "B1GQ"},
    ),
    EurostatSpec(
        "ez_hicp", "prc_hicp_manr", "Euroraum-Inflation (HVPI)", "%",
        {"geo": "EA", "coicop": "CP00", "unit": "RCH_A"},
        percent=True,
    ),
)

ECB_SERIES: tuple[EcbSpec, ...] = (
    EcbSpec(
        "ez_policy_rate", "FM", "D.U2.EUR.4F.KR.MRR_FR.LEV",
        "EZB-Hauptrefinanzierungssatz", "%",
    ),
    EcbSpec(
        "ez_yield_10y", "YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
        "Euroraum-Renditekurve 10J", "%",
    ),
    EcbSpec(
        "ez_yield_2y", "YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
        "Euroraum-Renditekurve 2J", "%",
    ),
)


def _to_series(
    key: str, label: str, unit: str, observations: list[tuple[date, float]], percent: bool
) -> MacroSeries | None:
    """Baut aus Beobachtungen eine MacroSeries mit 3- und 12-Monats-Vergleich."""
    if not observations:
        return None
    observations = sorted(observations)
    as_of, current = observations[-1]

    def value_near(months: int) -> float | None:
        cutoff = date(as_of.year, as_of.month, 1)
        year, month = cutoff.year, cutoff.month - months
        while month <= 0:
            month += 12
            year -= 1
        target = date(year, month, 1)
        candidates = [v for d, v in observations if d <= target]
        return candidates[-1] if candidates else None

    def delta(past: float | None) -> float | None:
        if past is None:
            return None
        if percent:
            return current - past
        if past == 0:
            return None
        return current / past - 1.0

    return MacroSeries(
        key=key, label=label, value=current,
        change_3m=delta(value_near(3)), change_12m=delta(value_near(12)),
        unit=unit, as_of=as_of,
    )


def _parse_period(text: str) -> date | None:
    """Versteht die Periodenformate beider Anbieter: 2026-07, 2026Q2, 2026-07-15."""
    text = text.strip()
    for pattern in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    if "Q" in text:  # Quartalsangabe wie 2026Q2 oder 2026-Q2
        try:
            year, quarter = text.replace("-", "").split("Q")
            return date(int(year), int(quarter) * 3, 1)
        except (ValueError, IndexError):
            return None
    return None


def _time_lookup(payload: dict):
    """Bildet einen Wertindex auf seine Periode ab.

    JSON-stat legt alle Werte flach in ein Objekt; `id` nennt die Achsen in
    ihrer Reihenfolge, `size` deren Längen. Der flache Index entspricht genau
    dann der Position auf der Zeitachse, wenn alle übrigen Achsen die Länge 1
    haben — dafür sind die Filter da.

    Genau diese Voraussetzung wurde bisher angenommen statt geprüft. Trifft
    sie nicht zu, enthält die Antwort mehrere Reihen nebeneinander, und die
    Zuordnung liefert stillschweigend ein Gemisch, das wie eine saubere
    Zeitreihe aussieht.
    """
    index = (
        payload.get("dimension", {})
        .get("time", {})
        .get("category", {})
        .get("index", {})
    )
    if not index:
        return None
    by_position = {position: period for period, position in index.items()}

    ids = payload.get("id") or []
    sizes = payload.get("size") or []
    if len(ids) == len(sizes) and "time" in ids:
        extra = [
            f"{name} ({size})"
            for name, size in zip(ids, sizes)
            if name != "time" and int(size) > 1
        ]
        if extra:
            # Nicht auflösbar: Die Werte ließen sich zwar Perioden zuordnen,
            # aber jede Periode träfe mehrere davon. Lieber abbrechen und den
            # nächsten Filtersatz probieren.
            log.warning(
                "Eurostat-Antwort enthält mehrere Reihen nebeneinander — "
                "diese Achsen sind nicht auf einen Wert eingegrenzt: %s.",
                ", ".join(extra),
            )
            return None

    return lambda flat: by_position.get(flat)


class EurostatClient:
    """Liest Eurostat-Reihen im JSON-stat-Format.

    Liefert ein Filtersatz nichts, werden die hinterlegten Ersatzfilter
    durchprobiert — eine leere Antwort ist bei Eurostat kein Fehler, sondern
    der Normalfall bei einem Code, den dieser Datensatz nicht kennt.
    """

    def __init__(self, cache: DayCache | None = None, timeout: float = 25.0) -> None:
        self.cache = cache or DayCache("cache", enabled=False)
        self.timeout = timeout

    def _load(self, spec: EurostatSpec, filters: dict[str, str]) -> list[tuple[date, float]]:
        def load() -> list[tuple[date, float]]:
            params = {"format": "JSON", "lang": "DE", **filters}
            response = requests.get(
                EUROSTAT_URL.format(dataset=spec.dataset),
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()

            values = payload.get("value") or {}
            lookup = _time_lookup(payload)
            if not values or lookup is None:
                return []

            observations: list[tuple[date, float]] = []
            for position, value in values.items():
                try:
                    period = lookup(int(position))
                except (TypeError, ValueError):
                    continue
                if period is None or value is None:
                    continue
                stamp = _parse_period(period)
                if stamp is not None:
                    observations.append((stamp, float(value)))
            return observations

        try:
            return self.cache.get_or_compute(
                "eurostat", cache_key(spec.dataset, sorted(filters.items())), load
            )
        except Exception as exc:
            log.warning("Eurostat-Reihe %s nicht abrufbar: %s", spec.dataset, exc)
            return []

    def _available_codes(self, spec: EurostatSpec) -> dict[str, list[str]]:
        """Fragt die Achsenbeschreibung des Datensatzes ab.

        Eurostat liefert die vollständigen Codelisten in jeder Antwort mit —
        auch in einer, die keine Werte enthält. `lastTimePeriod=1` begrenzt die
        Datenmenge auf eine Periode, die Beschreibung bleibt vollständig.

        Damit muss niemand raten, welcher Code gilt: Schlägt eine Reihe fehl,
        stehen die gültigen Werte in derselben Logzeile.
        """
        try:
            response = requests.get(
                EUROSTAT_URL.format(dataset=spec.dataset),
                params={"format": "JSON", "lang": "DE", "lastTimePeriod": "1"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            dimensions = response.json().get("dimension") or {}
        except Exception as exc:
            log.debug("Codeliste für %s nicht abrufbar: %s", spec.dataset, exc)
            return {}

        codes: dict[str, list[str]] = {}
        for axis in spec.filters:
            index = dimensions.get(axis, {}).get("category", {}).get("index") or {}
            if index:
                codes[axis] = sorted(index)
        return codes

    def fetch(self, spec: EurostatSpec) -> MacroSeries | None:
        for attempt, filters in enumerate(spec.filter_sets):
            observations = self._load(spec, filters)
            if observations:
                if attempt:
                    log.info(
                        "Eurostat-Reihe %s über Ersatzfilter geladen: %s",
                        spec.dataset, filters,
                    )
                return _to_series(
                    spec.key, spec.label, spec.unit, observations, spec.percent
                )

        # Antwort kam an, enthielt aber nichts Verwertbares. Fast immer ein
        # Code, den dieser Datensatz nicht kennt — ohne diese Meldung fiele die
        # Reihe lautlos aus und niemand würde es merken.
        codes = self._available_codes(spec)
        if codes:
            log.warning(
                "Eurostat-Reihe %s lieferte mit keinem von %d Filtersätzen "
                "Daten. Der Datensatz kennt diese Codes: %s. Versucht wurde "
                "zuletzt: %s.",
                spec.dataset,
                len(spec.filter_sets),
                "; ".join(
                    f"{axis} = {', '.join(values[:12])}"
                    + (" …" if len(values) > 12 else "")
                    for axis, values in codes.items()
                ),
                spec.filter_sets[-1],
            )
        else:
            log.warning(
                "Eurostat-Reihe %s lieferte mit keinem von %d Filtersätzen "
                "Daten, und die Codeliste war nicht abrufbar. Zuletzt "
                "versucht: %s.",
                spec.dataset, len(spec.filter_sets), spec.filter_sets[-1],
            )
        return None


class EcbClient:
    """Liest EZB-Reihen als CSV — deutlich robuster zu parsen als deren XML."""

    def __init__(self, cache: DayCache | None = None, timeout: float = 25.0) -> None:
        self.cache = cache or DayCache("cache", enabled=False)
        self.timeout = timeout

    def fetch(self, spec: EcbSpec) -> MacroSeries | None:
        def load() -> list[tuple[date, float]]:
            response = requests.get(
                ECB_URL.format(flow=spec.flow, key=spec.series_key),
                params={"format": "csvdata", "lastNObservations": "800"},
                timeout=self.timeout,
            )
            response.raise_for_status()

            observations: list[tuple[date, float]] = []
            for row in csv.DictReader(io.StringIO(response.text)):
                period = row.get("TIME_PERIOD") or row.get("TIME")
                raw = row.get("OBS_VALUE")
                if not period or raw in (None, ""):
                    continue
                stamp = _parse_period(period)
                if stamp is None:
                    continue
                try:
                    observations.append((stamp, float(raw)))
                except ValueError:
                    continue
            return observations

        try:
            observations = self.cache.get_or_compute(
                "ecb", cache_key(spec.flow, spec.series_key), load
            )
        except Exception as exc:
            log.warning("EZB-Reihe %s nicht abrufbar: %s", spec.series_key, exc)
            return None

        if not observations:
            log.warning(
                "EZB-Reihe %s lieferte keine Beobachtungen — prüfe die "
                "Reihenkennung.", spec.series_key,
            )
            return None

        return _to_series(spec.key, spec.label, spec.unit, observations, spec.percent)


def fetch_european_series(cache: DayCache | None = None) -> dict[str, MacroSeries]:
    """Holt alle europäischen Reihen. Kein API-Key nötig."""
    result: dict[str, MacroSeries] = {}

    eurostat = EurostatClient(cache=cache)
    for spec in EUROSTAT_SERIES:
        series = eurostat.fetch(spec)
        if series is not None:
            result[spec.key] = series

    ecb = EcbClient(cache=cache)
    for spec in ECB_SERIES:
        series = ecb.fetch(spec)
        if series is not None:
            result[spec.key] = series

    # Renditekurve für den Euroraum aus den beiden Laufzeiten ableiten.
    ten, two = result.get("ez_yield_10y"), result.get("ez_yield_2y")
    if ten and two and ten.value is not None and two.value is not None:
        result["ez_yield_curve"] = MacroSeries(
            key="ez_yield_curve",
            label="Euroraum-Renditekurve 10J–2J",
            value=ten.value - two.value,
            unit="%",
            as_of=ten.as_of,
        )

    expected = len(EUROSTAT_SERIES) + len(ECB_SERIES)
    if not result:
        log.warning(
            "Keine europäischen Makrodaten abrufbar — das Makrobild stützt sich "
            "auf die US-Reihen. Eurostat und EZB brauchen keinen Key; prüfe "
            "gegebenenfalls die Netzwerkverbindung."
        )
    else:
        # Explizit protokollieren, welche Reihen ankamen. Eine fehlende Reihe
        # sieht im Log sonst genauso aus wie eine erfolgreiche.
        log.info(
            "Europäische Makrodaten: %d von %d Reihen (%s).",
            len(result) - ("ez_yield_curve" in result),
            expected,
            ", ".join(sorted(k for k in result if k != "ez_yield_curve")),
        )
    return result
