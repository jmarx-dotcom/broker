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
import time
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


#: Unterhalb dieser Abdeckung sagt ein Lauf nichts mehr über den Markt aus.
#:
#: Die Hälfte ist keine feinjustierte Zahl, sondern die Grenze, ab der der Satz
#: "keine Treffer" aufhört, eine Aussage über den Markt zu sein. Wer von 217
#: Titeln einen bewertet hat, hat nicht nichts gefunden — er hat nicht
#: nachgesehen. Beides klingt am Ende gleich, und genau deshalb muss der
#: Unterschied ausgerechnet und ausgesprochen werden.
MIN_DATA_COVERAGE = 0.5
MIN_SCORING_COVERAGE = 0.5

#: Zweite Runde für abgewiesene Abrufe: wenige gleichzeitig, nach einer Pause.
RETRY_WORKERS = 2
RETRY_PAUSE = 20.0

#: Ab diesem Anteil sieht ein Ausfall nach Drosselung aus und nicht mehr nach
#: einzelnen toten Titeln. Nur dann lohnt die langsame zweite Runde: Zwanzig
#: Sekunden Pause wegen eines delisteten Titels wären verschwendet, und die
#: Drosselung wartet man mit voller Nebenläufigkeit ohnehin nicht aus.
THROTTLE_HINT = 0.05


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

    @property
    def data_coverage(self) -> float | None:
        """Anteil des Universums, der überhaupt Fundamentaldaten lieferte."""
        if not self.universe_size:
            return None
        return self.fundamentals_ok / self.universe_size

    @property
    def scoring_coverage(self) -> float | None:
        """Anteil der Filter-Überlebenden, der tatsächlich bewertet wurde.

        Das ist das schärfere der beiden Maße: Wer die harten Filter passiert
        hat, *soll* bewertet werden. Fällt er stattdessen mit einem Fehler aus,
        fehlt er in der Trefferliste, ohne je geprüft worden zu sein.
        """
        if not self.passed_filters:
            return None
        return self.scored / self.passed_filters

    @property
    def trouble(self) -> str | None:
        """Warum dieser Lauf nichts über den Markt aussagt — oder None.

        Kein Wahrheitswert, sondern ein Satz: Wer die Warnung liest, soll ohne
        Rückfrage wissen, woran es lag.
        """
        if not self.universe_size:
            return "Das Universum ist leer — es wurde kein Titel geprüft."

        data = self.data_coverage
        if data is not None and data < MIN_DATA_COVERAGE:
            return (
                f"Nur {self.fundamentals_ok} von {self.universe_size} Titeln "
                f"lieferten überhaupt Daten ({data:.0%}). Die Datenquelle hat "
                "den Lauf nicht bedient."
            )

        scoring = self.scoring_coverage
        if scoring is not None and scoring < MIN_SCORING_COVERAGE:
            return (
                f"Von {self.passed_filters} Titeln, die die Filter passiert "
                f"haben, wurden nur {self.scored} bewertet ({scoring:.0%}) — "
                f"bei {len(self.errors)} Fehlern. Die übrigen sind nicht "
                "durchgefallen, sie wurden nie geprüft."
            )
        return None

    @property
    def degraded(self) -> bool:
        return self.trouble is not None


@dataclass
class ScreeningResult:
    candidates: list[Candidate]
    regime: MacroRegime
    stats: ScreeningStats
    #: Zufallsstichprobe aus den *aussortierten* Titeln — die Kontrollgruppe.
    #: Ohne sie könnte das Journal nur zeigen, wie sich die Treffer entwickelt
    #: haben, nicht ob die Auswahl überhaupt etwas taugt.
    control: list[Candidate] = field(default_factory=list)

    @property
    def trouble(self) -> str | None:
        return self.stats.trouble

    @property
    def degraded(self) -> bool:
        return self.stats.degraded


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
        self,
        config: Config,
        provider: MarketDataProvider,
        workers: int = 8,
        retry_workers: int = RETRY_WORKERS,
        retry_pause: float = RETRY_PAUSE,
    ) -> None:
        self.config = config
        self.provider = provider
        self.workers = max(1, workers)
        self.retry_workers = max(1, retry_workers)
        self.retry_pause = retry_pause

    # -- Datenbeschaffung -------------------------------------------------

    def _attempt(self, keys, call, workers: int) -> tuple[dict, dict[str, str]]:
        """Ruft `call` für alle Schlüssel parallel auf. Gibt Treffer und Fehler."""
        found: dict = {}
        failed: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call, key): key for key in keys}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    found[key] = future.result()
                except Exception as exc:
                    failed[key] = str(exc)
        return found, failed

    def _gather(self, keys, call, label: str, stats: ScreeningStats) -> dict:
        """Holt alles — und fasst bei Ausfällen einmal langsamer nach.

        Yahoo weist unter Last ab. Der DAX-Lauf mit 40 Titeln hatte null
        Fehler, der Lauf über alle 674 Titel am 5. August hatte 548: erst 332
        Fundamentaldaten-Abrufe, dann 216 von 217 Kurshistorien. Das ist kein
        Ausfall der Schnittstelle, sondern eine Drosselung — dieselben Titel
        antworten kurz darauf wieder.

        Deshalb eine zweite Runde für die Ausfälle: nach einer Pause und mit
        wenigen gleichzeitigen Abrufen statt acht. Bewusst nur eine — wer
        beliebig oft nachfasst, verwandelt eine harte Störung in einen Lauf,
        der ins Zeitlimit kriecht, statt sie zu melden.

        Langsam nachgefasst wird aber nur, wenn der Ausfall überhaupt nach
        Drosselung aussieht. Einzelne Titel fallen aus, weil sie delistet sind;
        dafür zwanzig Sekunden zu warten, hilft niemandem.
        """
        keys = list(keys)
        found, failed = self._attempt(keys, call, self.workers)
        if failed:
            throttled = len(failed) > max(3, len(keys) * THROTTLE_HINT)
            workers = self.retry_workers if throttled else self.workers
            log.warning(
                "%s: %d von %d Abrufen fehlgeschlagen — zweite Runde mit %d "
                "gleichzeitigen Abrufen%s.",
                label, len(failed), len(keys), workers,
                f" nach {self.retry_pause:.0f} Sekunden Pause" if throttled else "",
            )
            if throttled and self.retry_pause:
                time.sleep(self.retry_pause)
            recovered, failed = self._attempt(list(failed), call, workers)
            found.update(recovered)
            if recovered:
                log.info(
                    "%s: %d Titel antworteten in der zweiten Runde.",
                    label, len(recovered),
                )

        for key, message in failed.items():
            stats.errors[key] = f"{label}: {message}"
        return found

    def _fetch_fundamentals(
        self, entries: list[UniverseEntry], stats: ScreeningStats
    ) -> dict[str, Fundamentals]:
        result = self._gather(
            [e.ticker for e in entries],
            self.provider.fundamentals,
            "Fundamentaldaten",
            stats,
        )
        stats.fundamentals_ok = len(result)
        return result

    def _fetch_histories(
        self, tickers: list[str], stats: ScreeningStats, period: str = "3y"
    ) -> dict[str, PriceHistory]:
        return self._gather(
            tickers,
            lambda t: self.provider.history(t, period),
            "Kurshistorie",
            stats,
        )

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
