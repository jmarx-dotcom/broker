"""Der Screening-Lauf: Universum laden, Daten holen, bewerten, ranken.

Ablauf in zwei Durchgängen, weil der Branchenmedian das gesamte Universum
braucht, bevor ein einzelner Titel dagegen verglichen werden kann:

  1. Fundamentaldaten für alle Titel holen, harte Filter anwenden,
     Branchenmediane berechnen.
  2. Für die verbliebenen Titel Kurshistorie holen und bewerten.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from broker.analysis import (
    analyze_quality,
    analyze_technical,
    analyze_valuation,
    assess_leverage,
    check_data_quality,
    combine_scores,
    sector_medians,
)
from broker.config import Config
from broker.macro.regime import bond_yield_for
from broker.models import Candidate, Fundamentals, MacroRegime, PriceHistory
from broker.providers.base import MarketDataProvider, ProviderError
from broker.universe import UniverseEntry

log = logging.getLogger(__name__)

#: Referenzindizes für die relative Stärke, je Region.
BENCHMARKS = {"US": "^GSPC", "DE": "^GDAXI", "EU": "^STOXX50E"}


@dataclass
class ScreeningStats:
    universe_size: int = 0
    fundamentals_ok: int = 0
    passed_filters: int = 0
    scored: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.universe_size} Titel im Universum, "
            f"{self.fundamentals_ok} mit Daten, "
            f"{self.passed_filters} nach Filtern, "
            f"{self.scored} bewertet, "
            f"{len(self.errors)} Fehler"
        )


@dataclass
class ScreeningResult:
    candidates: list[Candidate]
    regime: MacroRegime
    stats: ScreeningStats
    #: Zufallsstichprobe aus den *aussortierten* Titeln — die Kontrollgruppe.
    #: Ohne sie könnte das Journal nur zeigen, wie sich die Treffer entwickelt
    #: haben, nicht ob die Auswahl überhaupt etwas taugt.
    control: list[Candidate] = field(default_factory=list)


#: Größe der Kontrollgruppe je Lauf. In derselben Größenordnung wie die
#: Trefferliste, damit beide Gruppen etwa gleich schnell Stichprobe sammeln.
CONTROL_SAMPLE_SIZE = 15


def draw_control_group(
    scored: list[Candidate],
    threshold: float,
    size: int = CONTROL_SAMPLE_SIZE,
    run_date: date | None = None,
) -> list[Candidate]:
    """Zieht eine Zufallsstichprobe aus den aussortierten Titeln.

    Der Zufall ist an das Datum gebunden: Ein wiederholter Lauf am selben Tag
    zieht dieselbe Gruppe. Sonst könnte man — auch ungewollt — so lange neu
    würfeln, bis die Kontrollgruppe schlecht aussieht, und hätte damit genau
    die Beliebigkeit eingebaut, gegen die die Kontrollgruppe schützen soll.
    """
    rejected = [
        c for c in scored
        if c.total_score < threshold and c.technical.price is not None
    ]
    if not rejected:
        return []
    rng = random.Random((run_date or date.today()).isoformat())
    return rng.sample(rejected, min(size, len(rejected)))


def _benchmark_key(entry: UniverseEntry) -> str:
    if entry.region == "US":
        return "US"
    if entry.region == "DE":
        return "DE"
    return "EU"


class Screener:
    def __init__(
        self, config: Config, provider: MarketDataProvider, workers: int = 8
    ) -> None:
        self.config = config
        self.provider = provider
        self.workers = max(1, workers)

    # -- Datenbeschaffung -------------------------------------------------

    def _fetch_fundamentals(
        self, entries: list[UniverseEntry], stats: ScreeningStats
    ) -> dict[str, Fundamentals]:
        result: dict[str, Fundamentals] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.provider.fundamentals, e.ticker): e for e in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result[entry.ticker] = future.result()
                except (ProviderError, Exception) as exc:
                    stats.errors[entry.ticker] = f"Fundamentaldaten: {exc}"
        stats.fundamentals_ok = len(result)
        return result

    def _fetch_histories(
        self, tickers: list[str], stats: ScreeningStats, period: str = "3y"
    ) -> dict[str, PriceHistory]:
        result: dict[str, PriceHistory] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.provider.history, t, period): t for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result[ticker] = future.result()
                except (ProviderError, Exception) as exc:
                    stats.errors[ticker] = f"Kurshistorie: {exc}"
        return result

    def _fetch_benchmarks(self) -> dict[str, pd.Series]:
        benchmarks: dict[str, pd.Series] = {}
        for key, ticker in BENCHMARKS.items():
            try:
                benchmarks[key] = self.provider.history(ticker, period="3y").close
            except Exception as exc:
                log.warning("Referenzindex %s nicht verfügbar: %s", ticker, exc)
        return benchmarks

    # -- Filter -----------------------------------------------------------

    def _market_cap_eur(self, f: Fundamentals, eur_usd: float | None) -> float | None:
        """Marktkapitalisierung in Euro, damit der Filter überall gleich streng ist.

        Ohne Umrechnung wäre die 300-Millionen-Schwelle für Dollar-Titel eine
        andere als für Euro-Titel — je nach Kurs um etwa ein Zehntel verschoben.
        """
        if f.market_cap is None:
            return None
        currency = (f.currency or "").upper()
        if currency in ("", "EUR"):
            return f.market_cap
        if currency == "USD" and eur_usd:
            return f.market_cap / eur_usd
        return f.market_cap  # unbekannte Währung: unverändert, aber nicht verwerfen

    def _passes_hard_filters(
        self, f: Fundamentals, eur_usd: float | None = None
    ) -> str | None:
        """Gibt den Ablehnungsgrund zurück oder None, wenn der Titel durchkommt."""
        t = self.config.thresholds

        market_cap = self._market_cap_eur(f, eur_usd)
        if market_cap is None or market_cap < t.min_market_cap:
            return "Marktkapitalisierung zu klein oder unbekannt"
        if f.trailing_pe is None:
            return "Kein KGV verfügbar"
        if f.trailing_pe <= t.min_trailing_pe:
            return "Negatives KGV (Verlust)"
        if f.trailing_pe > t.max_trailing_pe:
            return "KGV oberhalb der Obergrenze"
        return None

    # -- Hauptlauf --------------------------------------------------------

    def run(self, entries: list[UniverseEntry], regime: MacroRegime) -> ScreeningResult:
        stats = ScreeningStats(universe_size=len(entries))
        by_ticker = {e.ticker: e for e in entries}

        log.info("Hole Fundamentaldaten für %d Titel …", len(entries))
        fundamentals = self._fetch_fundamentals(entries, stats)

        # Branchenmediane über das komplette Universum, nicht nur die Treffer —
        # sonst vergleicht man günstige Titel nur mit anderen günstigen Titeln.
        medians = sector_medians(list(fundamentals.values()))
        log.info("Branchenmediane für %d Sektoren berechnet.", len(medians))

        # DEXUSEU ist USD je EUR — für die Umrechnung von Dollar-Werten.
        eur_usd_series = regime.series.get("eur_usd")
        eur_usd = eur_usd_series.value if eur_usd_series else None
        if eur_usd:
            log.info("Wechselkurs EUR/USD %.4f für die Größenschwelle.", eur_usd)

        survivors: list[str] = []
        for ticker, f in fundamentals.items():
            reason = self._passes_hard_filters(f, eur_usd)
            if reason is None:
                survivors.append(ticker)
            else:
                log.debug("%s aussortiert: %s", ticker, reason)
        stats.passed_filters = len(survivors)
        log.info("%d Titel nach den harten Filtern.", len(survivors))

        log.info("Hole Kurshistorien …")
        histories = self._fetch_histories(survivors, stats)
        benchmarks = self._fetch_benchmarks()

        candidates: list[Candidate] = []
        for ticker in survivors:
            history = histories.get(ticker)
            if history is None:
                continue
            if len(history) < self.config.thresholds.min_history_days:
                stats.errors[ticker] = "Zu kurze Kurshistorie"
                continue

            f = fundamentals[ticker]
            entry = by_ticker[ticker]
            benchmark = benchmarks.get(_benchmark_key(entry))

            valuation = analyze_valuation(
                f,
                history,
                sector_medians=medians,
                bond_yield=bond_yield_for(regime, entry.region),
            )
            technical = analyze_technical(history, benchmark=benchmark)
            quality = analyze_quality(f)
            data_quality = check_data_quality(f, history)
            macro_score = regime.score_for(f.sector)

            if data_quality.severe_flags:
                log.warning(
                    "%s: Rohdaten widersprüchlich — %s",
                    ticker,
                    data_quality.severe_flags[0].message,
                )

            total = combine_scores(
                valuation,
                technical,
                quality,
                macro_score,
                self.config.weights,
                data_quality,
            )
            leverage = assess_leverage(
                ticker,
                technical.annualized_volatility,
                days=60,
                factor=3.0,
                annual_financing_rate=(bond_yield_for(regime, entry.region) or 0.03) + 0.03,
            )

            candidates.append(
                Candidate(
                    ticker=ticker,
                    fundamentals=f,
                    valuation=valuation,
                    technical=technical,
                    quality=quality,
                    data_quality=data_quality,
                    leverage=leverage,
                    macro_score=macro_score,
                    total_score=total,
                )
            )

        stats.scored = len(candidates)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        threshold = self.config.thresholds.min_score
        selected = [c for c in candidates if c.total_score >= threshold]
        selected = selected[: self.config.max_candidates]

        control = draw_control_group(candidates, threshold)

        log.info(
            "%d Titel bewertet, %d über der Score-Schwelle von %.0f, "
            "%d als Kontrollgruppe gezogen.",
            len(candidates),
            len(selected),
            threshold,
            len(control),
        )
        return ScreeningResult(
            candidates=selected, regime=regime, stats=stats, control=control
        )
