"""Drift-Prüfung: Was hat sich an den externen Quellen geändert?

Die Fehler, die dieses Werkzeug im Betrieb einholen, sind selten Programmier-
fehler. Es sind Änderungen draußen: Ein Index tauscht Mitglieder aus, Eurostat
benennt den Euroraum von EA20 in EA21 um, Yahoo ändert die Bezeichnung einer
Bilanzzeile. Jede einzelne davon ist harmlos und läuft still ins Leere — eine
Warnung im Log, die niemand liest, und eine Kennzahl, die ab dann fehlt.

Deshalb prüft dieses Modul die drei bekannten Driftarten aktiv nach:

1. **Tote Ticker** — Titel, die keine Kursdaten mehr liefern.
2. **Ausgefallene Makroreihen** — erwartete Reihen, die nicht mehr ankommen.
3. **Umbenannte Abschlusszeilen** — Felder, die im ganzen Universum leer sind.

Geprüft wird gegen die *lebenden* Schnittstellen, nicht gegen Logtext. Ein
Logformat ändert sich, sobald jemand eine Meldung umformuliert; die Frage
"antwortet diese Reihe noch?" bleibt dieselbe.

Nur die erste Driftart lässt sich mechanisch beheben — eine Zeile aus einer
CSV-Datei zu entfernen braucht kein Urteilsvermögen. Die anderen beiden werden
benannt, nicht repariert.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from broker.providers.base import MarketDataProvider
from broker.universe.loader import DATA_DIR, load_universe

log = logging.getLogger(__name__)

#: Fallen mehr als so viele Titel gleichzeitig aus, ist nicht das Universum
#: veraltet, sondern die Datenquelle gestört. Dann wäre es fatal, hunderte
#: Ticker zur Löschung vorzuschlagen.
OUTAGE_RATIO = 0.20

#: Stichprobe für die Abschlussfelder. Größer bringt wenig: Ein umbenanntes
#: Feld fehlt bei *allen* Titeln, nicht bei einzelnen.
STATEMENT_SAMPLE = 40

#: Diese Felder kommen aus den Abschlüssen und tragen den Drift-Verdacht.
STATEMENT_FIELDS = (
    "ebit",
    "interest_expense",
    "current_assets",
    "current_liabilities",
    "total_equity",
)


@dataclass
class Finding:
    """Ein Befund der Drift-Prüfung."""

    check: str
    subject: str
    message: str
    #: Mechanisch behebbar — ohne Urteilsvermögen, rein textuell.
    fixable: bool = False


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)
    #: Was tatsächlich geprüft wurde, damit ein leerer Bericht von einem
    #: übersprungenen Lauf zu unterscheiden ist.
    checked: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.fixable]

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return "Keine Abweichungen gefunden."
        by_check: dict[str, int] = {}
        for finding in self.findings:
            by_check[finding.check] = by_check.get(finding.check, 0) + 1
        parts = [f"{count}× {check}" for check, count in sorted(by_check.items())]
        return ", ".join(parts)


# -- 1. Tote Ticker --------------------------------------------------------


def _outage_finding(failed: int, total: int) -> list[Finding]:
    """Störung, nicht Drift. Hier nichts zur Entfernung vorzuschlagen ist die
    wichtigste Nicht-Aktion des ganzen Moduls."""
    return [
        Finding(
            check="Datenquelle",
            subject="Kurshistorie",
            message=(
                f"{failed} von {total} Titeln ohne Kursdaten "
                f"({failed / total * 100:.0f}%). Das ist eine Störung der "
                "Datenquelle, keine Index-Änderung — es wird nichts zur "
                "Entfernung vorgeschlagen."
            ),
        )
    ]


def check_tickers(
    provider: MarketDataProvider,
    group: str = "all",
    workers: int = 8,
    retry_delay: float = 2.0,
) -> tuple[list[Finding], int, str | None]:
    """Fragt jeden Titel nach Kursdaten. Gibt Befunde, Anzahl und Abbruchgrund.

    Wer beim ersten Versuch nichts liefert, wird ein zweites Mal gefragt —
    einzeln und mit Pause. Ein einzelner fehlgeschlagener Abruf ist kein
    Beleg für ein Delisting: Beim ersten scharfen Lauf standen unter zwanzig
    gemeldeten Titeln fünf, die weiter gehandelt werden (BNY Mellon, Marsh &
    McLennan, Fiserv, Coterra, Schaeffler). Sie waren nicht tot, sondern
    kurz nicht erreichbar — parallele Abfragen laufen bei Yahoo gelegentlich
    ins Leere.
    """
    entries = load_universe(group)
    if not entries:
        return [], 0, "Universum leer"

    def probe(ticker: str) -> tuple[str, bool]:
        try:
            history = provider.history(ticker, period="1mo")
        except Exception:
            return ticker, False
        return ticker, not history.close.empty

    suspects: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, e.ticker) for e in entries]
        for future in as_completed(futures):
            ticker, alive = future.result()
            if not alive:
                suspects.append(ticker)

    # Störungsprüfung vor der zweiten Runde: Bei einem breiten Ausfall wären
    # alle Titel Verdachtsfälle, und die einzelnen Wiederholungen mit Pause
    # würden jedes Zeitlimit sprengen — für ein Ergebnis, das ohnehin
    # verworfen wird.
    if len(suspects) > len(entries) * OUTAGE_RATIO:
        return _outage_finding(len(suspects), len(entries)), len(entries), (
            "Ausfallquote zu hoch"
        )

    # Zweite Runde: nacheinander statt parallel, damit sich die Abfragen nicht
    # gegenseitig ausbremsen, und mit Pause zwischen den Versuchen.
    dead: list[str] = []
    for ticker in suspects:
        if retry_delay:
            time.sleep(retry_delay)
        if not probe(ticker)[1]:
            dead.append(ticker)

    if suspects and len(dead) < len(suspects):
        log.info(
            "%d von %d auffälligen Titeln antworteten im zweiten Versuch — "
            "sie gelten nicht als tot.",
            len(suspects) - len(dead), len(suspects),
        )

    if len(dead) > len(entries) * OUTAGE_RATIO:
        return _outage_finding(len(dead), len(entries)), len(entries), (
            "Ausfallquote zu hoch"
        )

    findings = [
        Finding(
            check="Toter Ticker",
            subject=ticker,
            message=f"{ticker} liefert keine Kursdaten mehr.",
            fixable=True,
        )
        for ticker in sorted(dead)
    ]
    return findings, len(entries), None


# -- 2. Ausgefallene Makroreihen -------------------------------------------


def check_macro_series(series: dict, expected: dict[str, str]) -> list[Finding]:
    """Vergleicht die angekommenen Reihen mit den erwarteten."""
    return [
        Finding(
            check="Makroreihe",
            subject=key,
            message=(
                f"{label} ({key}) kam nicht an. Bei Eurostat steckt dahinter "
                "meist ein geänderter Code — der Client protokolliert im "
                "Fehlerfall die gültige Codeliste."
            ),
        )
        for key, label in sorted(expected.items())
        if key not in series
    ]


def expected_macro_keys(with_fred: bool) -> dict[str, str]:
    from broker.macro.europe import ECB_SERIES, EUROSTAT_SERIES

    expected = {spec.key: spec.label for spec in EUROSTAT_SERIES}
    expected.update({spec.key: spec.label for spec in ECB_SERIES})
    expected["ez_yield_curve"] = "Euroraum-Renditekurve 10J–2J"
    if with_fred:
        from broker.macro.fred import SERIES

        expected.update({spec.key: spec.label for spec in SERIES})
    return expected


# -- 3. Umbenannte Abschlusszeilen -----------------------------------------


def check_statement_fields(
    provider: MarketDataProvider,
    group: str = "all",
    sample: int = STATEMENT_SAMPLE,
    workers: int = 8,
) -> tuple[list[Finding], int]:
    """Prüft, ob die Abschlussfelder im Universum überhaupt noch ankommen.

    Einzelne Lücken sind normal — kleine Nebenwerte liefern oft keine
    Abschlüsse. Ist ein Feld aber über die *ganze* Stichprobe leer, hat Yahoo
    die Zeile umbenannt und die Liste der Bezeichnungen in `providers/yahoo.py`
    braucht einen Eintrag.
    """
    entries = load_universe(group)[:sample]
    if not entries:
        return [], 0

    coverage = {name: 0 for name in STATEMENT_FIELDS}
    usable = 0

    def probe(ticker: str):
        try:
            return provider.fundamentals(ticker)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, e.ticker) for e in entries]
        for future in as_completed(futures):
            data = future.result()
            if data is None:
                continue
            usable += 1
            for name in STATEMENT_FIELDS:
                if getattr(data, name, None) is not None:
                    coverage[name] += 1

    if usable < sample // 2:
        # Zu wenig Grundlage: Aus drei Antworten lässt sich nicht schließen,
        # dass ein Feld verschwunden ist.
        return [], usable

    findings = [
        Finding(
            check="Abschlussfeld",
            subject=name,
            message=(
                f"'{name}' fehlt bei allen {usable} geprüften Titeln. Yahoo hat "
                "die Zeile vermutlich umbenannt — die Bezeichnungsliste in "
                "providers/yahoo.py (_statements) braucht einen weiteren Namen."
            ),
        )
        for name in STATEMENT_FIELDS
        if coverage[name] == 0
    ]
    return findings, usable


# -- Mechanische Korrektur -------------------------------------------------


def remove_tickers(tickers: list[str], data_dir: Path | None = None) -> dict[str, int]:
    """Entfernt Ticker aus den Universum-Dateien. Gibt Treffer je Datei zurück.

    Bewusst zeilenweise über den Rohtext statt über csv-Modul und Neuschreiben:
    So bleiben Reihenfolge, Kommentare und Zeilenenden der Datei unberührt, und
    der Pull Request zeigt genau die entfernten Zeilen — nichts sonst.
    """
    directory = data_dir or DATA_DIR
    wanted = set(tickers)
    removed: dict[str, int] = {}

    for path in sorted(directory.glob("*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines:
            continue
        header, rows = lines[0], lines[1:]
        kept = [r for r in rows if r.split(",")[0].strip() not in wanted]
        if len(kept) == len(rows):
            continue
        path.write_text(header + "".join(kept), encoding="utf-8")
        removed[path.name] = len(rows) - len(kept)

    return removed


# -- Gesamtlauf ------------------------------------------------------------


def run_doctor(
    provider: MarketDataProvider,
    series: dict,
    group: str = "all",
    with_fred: bool = True,
    skip_tickers: bool = False,
    skip_statements: bool = False,
    retry_delay: float = 2.0,
) -> DoctorReport:
    """Führt alle Prüfungen aus und sammelt die Befunde."""
    report = DoctorReport()

    if skip_tickers:
        report.skipped.append("Ticker")
    else:
        findings, count, aborted = check_tickers(
            provider, group, retry_delay=retry_delay
        )
        report.findings.extend(findings)
        report.checked["Ticker"] = count
        if aborted:
            report.skipped.append(f"Ticker-Korrektur ({aborted})")

    expected = expected_macro_keys(with_fred)
    report.findings.extend(check_macro_series(series, expected))
    report.checked["Makroreihen"] = len(expected)

    if skip_statements:
        report.skipped.append("Abschlussfelder")
    else:
        findings, usable = check_statement_fields(provider, group)
        report.findings.extend(findings)
        report.checked["Abschlussfelder"] = usable

    return report
