"""Journal: hält fest, was der Screener wann vorgeschlagen hat — und wie es lief.

Das ist ausdrücklich **kein Backtest**. Ein Backtest würde Fundamentaldaten zum
damaligen Stand brauchen; was heute in einer Gratis-Datenbank steht, ist die
nachträglich korrigierte Fassung. Wer damit rückrechnet, weiß Dinge, die zum
Kaufzeitpunkt niemand wusste, und bekommt systematisch zu schöne Ergebnisse.

Stattdessen wird hier nach vorne gemessen: Jeder Lauf schreibt seine Treffer mit
Kurs und Datum fort, und spätere Läufe rechnen aus, wie sich diese Titel seither
gegenüber ihrem Index entwickelt haben. Das dauert Monate, bis es etwas aussagt —
aber es ist verzerrungsfrei, weil zum Zeitpunkt der Aufzeichnung niemand die
Antwort kannte.

Zwei Vorkehrungen gegen Selbstbetrug:

* **Entprellung.** Derselbe Titel steht oft wochenlang in Folge auf der Liste.
  Würde man jede Nennung als eigene Beobachtung zählen, entstünden aus einer
  einzigen Kursbewegung dreißig scheinbar unabhängige Datenpunkte. Für die
  Auswertung zählt deshalb pro Titel und Kalendermonat nur die erste Nennung.
* **Feste Auswertungsraster.** Die Gruppen stehen unten im Code fest. Wer so
  lange nach Schnittmengen sucht, bis eine gut aussieht, findet immer eine.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median

import pandas as pd

from broker.models import Candidate
from broker.providers.base import MarketDataProvider

log = logging.getLogger(__name__)

#: Auswertungsfenster in Kalendertagen.
HORIZONS: dict[str, int] = {"1M": 30, "3M": 91, "6M": 182, "12M": 365}

#: Score-Gruppen, vorab festgelegt.
SCORE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("55–60", 55.0, 60.0),
    ("60–65", 60.0, 65.0),
    ("65–70", 65.0, 70.0),
    ("70+", 70.0, 1000.0),
)

#: Unterhalb dieser Beobachtungszahl wird keine Kennzahl ausgewiesen.
MIN_SAMPLE = 10


@dataclass
class JournalEntry:
    """Eine Nennung eines Titels an einem Tag."""

    date: str
    ticker: str
    name: str
    region: str
    benchmark: str
    currency: str | None
    price: float
    total_score: float
    valuation_score: float
    quality_score: float
    technical_score: float
    macro_score: float
    trailing_pe: float | None = None
    forward_pe: float | None = None
    rsi14: float | None = None
    setup: str = ""
    llm_verdict: str = ""
    sector: str = ""
    #: "hit" = vorgeschlagen, "control" = zufällig aus den aussortierten
    #: Titeln gezogen. Ältere Zeilen haben das Feld nicht und gelten als
    #: Treffer — damals wurde nichts anderes aufgezeichnet.
    kind: str = "hit"

    @property
    def recorded_on(self) -> date:
        return datetime.strptime(self.date, "%Y-%m-%d").date()

    @property
    def is_control(self) -> bool:
        return self.kind == "control"


@dataclass
class BucketStats:
    """Ergebnis einer Gruppe für einen Zeithorizont."""

    label: str
    sample: int
    median_excess: float | None = None
    median_return: float | None = None
    hit_rate: float | None = None

    @property
    def reportable(self) -> bool:
        return self.sample >= MIN_SAMPLE


@dataclass
class PerformanceReport:
    horizon: str
    by_score: list[BucketStats] = field(default_factory=list)
    by_verdict: list[BucketStats] = field(default_factory=list)
    by_setup: list[BucketStats] = field(default_factory=list)
    total_observations: int = 0
    #: Treffer gegen Kontrollgruppe — die eigentliche Frage.
    hits: BucketStats | None = None
    control: BucketStats | None = None

    @property
    def reportable(self) -> bool:
        return self.total_observations >= MIN_SAMPLE

    @property
    def edge(self) -> float | None:
        """Vorsprung der Treffer gegenüber der Kontrollgruppe.

        Das ist die Zahl, für die das Journal überhaupt gebaut wurde: Schlägt
        die Auswahl den Zufall? Sie wird erst ausgewiesen, wenn *beide*
        Gruppen genug Beobachtungen haben — ein Vorsprung gegenüber drei
        Kontrolltiteln ist keiner.
        """
        if self.hits is None or self.control is None:
            return None
        if not (self.hits.reportable and self.control.reportable):
            return None
        if self.hits.median_excess is None or self.control.median_excess is None:
            return None
        return round(self.hits.median_excess - self.control.median_excess, 4)


class Journal:
    """Append-only-Datei im JSONL-Format, eine Zeile je Nennung."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # -- Schreiben --------------------------------------------------------

    def append(self, candidates: list[Candidate], benchmarks: dict[str, str],
               run_date: date | None = None, control: list[Candidate] | None = None,
               ) -> int:
        """Schreibt Treffer und Kontrollgruppe eines Laufs fort.

        Beide landen in derselben Datei und unterscheiden sich nur im Feld
        `kind`. Getrennte Dateien wären verlockend, würden aber die
        Entprellung und die Kursabfrage doppeln — und ein Auswertungsschritt,
        der nur eine der beiden Dateien liest, wäre lautlos falsch.
        """
        tagged = [(c, "hit") for c in candidates]
        tagged += [(c, "control") for c in (control or [])]
        if not tagged:
            return 0

        stamp = (run_date or date.today()).isoformat()
        existing = {(e.date, e.ticker) for e in self.entries()}
        seen: set[str] = set()

        rows: list[JournalEntry] = []
        for c, kind in tagged:
            if (stamp, c.ticker) in existing or c.ticker in seen:
                continue  # Lauf wurde am selben Tag wiederholt
            if c.technical.price is None:
                continue  # ohne Einstiegskurs ist der Eintrag wertlos
            seen.add(c.ticker)
            rows.append(
                JournalEntry(
                    kind=kind,
                    date=stamp,
                    ticker=c.ticker,
                    name=c.name,
                    region=_region_of(c),
                    benchmark=benchmarks.get(_region_of(c), ""),
                    currency=c.fundamentals.currency,
                    price=float(c.technical.price),
                    total_score=round(c.total_score, 2),
                    valuation_score=round(c.valuation.score, 2),
                    quality_score=round(c.quality.score, 2),
                    technical_score=round(c.technical.score, 2),
                    macro_score=round(c.macro_score, 2),
                    trailing_pe=c.valuation.trailing_pe,
                    forward_pe=c.valuation.forward_pe,
                    rsi14=c.technical.rsi14,
                    setup=c.technical.setup,
                    llm_verdict=(c.llm.verdict if c.llm else ""),
                    sector=c.sector,
                )
            )

        if not rows:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        return len(rows)

    # -- Lesen ------------------------------------------------------------

    def entries(self) -> list[JournalEntry]:
        if not self.path.is_file():
            return []
        result: list[JournalEntry] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                result.append(JournalEntry(**payload))
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("Journalzeile %d unlesbar, übersprungen: %s", line_number, exc)
        return result

    def appearances(self, lookback_runs: int = 20) -> dict[str, int]:
        """Wie oft stand ein Titel in den letzten N Läufen auf der Liste?

        Ein Titel, der über Wochen immer wieder auftaucht, ist ein stabileres
        Signal als einer, der einmalig aufblitzt — Letzteres geht oft auf einen
        Datenfehler zurück, der am Folgetag korrigiert ist.
        """
        entries = self.entries()
        if not entries:
            return {}
        run_dates = sorted({e.date for e in entries}, reverse=True)[:lookback_runs]
        recent = set(run_dates)
        counts: dict[str, int] = defaultdict(int)
        for entry in entries:
            if entry.date in recent:
                counts[entry.ticker] += 1
        return dict(counts)

    @property
    def run_count(self) -> int:
        return len({e.date for e in self.entries()})


def _region_of(candidate: Candidate) -> str:
    """Leitet die Region aus dem Ticker-Suffix ab."""
    ticker = candidate.ticker
    if ticker.endswith(".DE"):
        return "DE"
    if "." in ticker:
        return "EU"
    return "US"


# -- Auswertung ------------------------------------------------------------


def _price_on_or_before(close: pd.Series, target: date) -> float | None:
    """Letzter Schlusskurs am oder vor dem Zieldatum."""
    if close.empty:
        return None
    index = close.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    series = pd.Series(close.to_numpy(), index=index)
    mask = series.index <= pd.Timestamp(target)
    if not mask.any():
        return None
    return float(series[mask].iloc[-1])


def _deduplicate(entries: list[JournalEntry]) -> list[JournalEntry]:
    """Pro Titel und Kalendermonat nur die früheste Nennung.

    Ohne das erzeugt eine einzige Kursbewegung so viele scheinbar unabhängige
    Beobachtungen, wie der Titel Tage lang auf der Liste stand.
    """
    best: dict[tuple[str, str], JournalEntry] = {}
    for entry in sorted(entries, key=lambda e: e.date):
        key = (entry.ticker, entry.date[:7])
        best.setdefault(key, entry)
    return list(best.values())


@dataclass
class Observation:
    entry: JournalEntry
    own_return: float
    benchmark_return: float | None

    @property
    def excess(self) -> float | None:
        if self.benchmark_return is None:
            return None
        return self.own_return - self.benchmark_return


def collect_observations(
    entries: list[JournalEntry],
    provider: MarketDataProvider,
    horizon_days: int,
    today: date | None = None,
) -> list[Observation]:
    """Berechnet für alle hinreichend alten Einträge die Entwicklung im Fenster."""
    today = today or date.today()
    cutoff = today - timedelta(days=horizon_days)
    mature = [e for e in _deduplicate(entries) if e.recorded_on <= cutoff]
    if not mature:
        return []

    tickers = {e.ticker for e in mature} | {e.benchmark for e in mature if e.benchmark}
    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            closes[ticker] = provider.history(ticker, period="3y").close
        except Exception as exc:
            log.debug("Kurse für %s nicht verfügbar: %s", ticker, exc)

    observations: list[Observation] = []
    for entry in mature:
        close = closes.get(entry.ticker)
        if close is None:
            continue
        start_date = entry.recorded_on
        end_date = start_date + timedelta(days=horizon_days)

        start = _price_on_or_before(close, start_date)
        end = _price_on_or_before(close, end_date)
        if not start or not end or start <= 0:
            continue
        own = end / start - 1.0

        bench_return = None
        bench_close = closes.get(entry.benchmark)
        if bench_close is not None:
            b_start = _price_on_or_before(bench_close, start_date)
            b_end = _price_on_or_before(bench_close, end_date)
            if b_start and b_end and b_start > 0:
                bench_return = b_end / b_start - 1.0

        observations.append(Observation(entry, own, bench_return))
    return observations


def _stats(label: str, observations: list[Observation]) -> BucketStats:
    if not observations:
        return BucketStats(label=label, sample=0)

    returns = [o.own_return for o in observations]
    excesses = [o.excess for o in observations if o.excess is not None]

    return BucketStats(
        label=label,
        sample=len(observations),
        median_return=round(median(returns), 4),
        # Median statt Mittelwert: Aktienrenditen sind stark rechtsschief,
        # ein einzelner Verdoppler würde den Mittelwert dominieren.
        median_excess=round(median(excesses), 4) if excesses else None,
        hit_rate=(
            round(sum(1 for e in excesses if e > 0) / len(excesses), 3)
            if excesses
            else None
        ),
    )


def build_report(
    observations: list[Observation], horizon_label: str
) -> PerformanceReport:
    # Die Kontrollgruppe gehört in keine der Aufschlüsselungen: Ihre Scores
    # liegen definitionsgemäß unter der Schwelle, und ein LLM-Urteil hat sie
    # nie bekommen. Sie ist der Vergleichsmaßstab, nicht eine Gruppe unter
    # vielen.
    hits = [o for o in observations if not o.entry.is_control]
    control = [o for o in observations if o.entry.is_control]

    report = PerformanceReport(
        horizon=horizon_label,
        total_observations=len(hits),
        hits=_stats("Treffer", hits),
        control=_stats("Kontrollgruppe", control),
    )

    for label, low, high in SCORE_BUCKETS:
        group = [o for o in hits if low <= o.entry.total_score < high]
        report.by_score.append(_stats(label, group))

    for verdict in ("zyklisch-guenstig", "strukturell-billig", "unklar"):
        group = [o for o in hits if o.entry.llm_verdict == verdict]
        report.by_verdict.append(_stats(verdict, group))

    setups = sorted({o.entry.setup for o in hits if o.entry.setup})
    for setup in setups:
        group = [o for o in hits if o.entry.setup == setup]
        report.by_setup.append(_stats(setup, group))

    return report


def evaluate(
    journal: Journal,
    provider: MarketDataProvider,
    horizons: dict[str, int] | None = None,
    today: date | None = None,
) -> list[PerformanceReport]:
    """Wertet das Journal über alle Zeithorizonte aus."""
    entries = journal.entries()
    reports: list[PerformanceReport] = []
    for label, days in (horizons or HORIZONS).items():
        observations = collect_observations(entries, provider, days, today=today)
        reports.append(build_report(observations, label))
    return reports
