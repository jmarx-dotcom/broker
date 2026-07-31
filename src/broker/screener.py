"""Der Screening-Lauf: Universum laden, Daten holen, bewerten, ranken.

Ablauf in zwei Durchgängen, weil der Branchenmedian das gesamte Universum
braucht, bevor ein einzelner Titel dagegen verglichen werden kann:

  1. Fundamentaldaten für alle Titel holen, harte Filter anwenden,
     Branchenmediane berechnen.
  2. Für die verbliebenen Titel Kurshistorie holen und bewerten.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from broker.analysis import (
    analyze_quality,
    analyze_technical,
    analyze_valuation,
    check_data_quality,
    combine_scores,
    sector_median_pe,
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

    def _passes_hard_filters(self, f: Fundamentals) -> str | None:
        """Gibt den Ablehnungsgrund zurück oder None, wenn der Titel durchkommt."""
        t = self.config.thresholds

        if f.market_cap is None or f.market_cap < t.min_market_cap:
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
        medians = sector_median_pe(list(fundamentals.values()))
        log.info("Branchenmediane für %d Sektoren berechnet.", len(medians))

        survivors: list[str] = []
        for ticker, f in fundamentals.items():
            reason = self._passes_hard_filters(f)
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
            candidates.append(
                Candidate(
                    ticker=ticker,
                    fundamentals=f,
                    valuation=valuation,
                    technical=technical,
                    quality=quality,
                    data_quality=data_quality,
                    macro_score=macro_score,
                    total_score=total,
                )
            )

        stats.scored = len(candidates)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        threshold = self.config.thresholds.min_score
        selected = [c for c in candidates if c.total_score >= threshold]
        selected = selected[: self.config.max_candidates]

        log.info(
            "%d Titel bewertet, %d über der Score-Schwelle von %.0f.",
            len(candidates),
            len(selected),
            threshold,
        )
        return ScreeningResult(candidates=selected, regime=regime, stats=stats)
