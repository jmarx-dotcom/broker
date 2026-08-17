"""Bestand an Fundamentaldaten, der einen Lauf überdauert.

Yahoo hat zwei Endpunkte mit sehr verschiedenen Grenzen. Kursdaten sind
großzügig — die Wartungsprüfung holt 674 Kurshistorien am Stück. Fundamentaldaten
(`.info`) sind hart gedeckelt: 342, 334 und 327 Titel an drei Tagen, und dieser
Deckel erholt sich nicht in Minuten. Warten half bei den Kursen, bei den
Kennzahlen nicht.

Ein Universum von 674 Titeln passt damit nicht in einen Lauf. Es passt aber in
zwei oder drei, wenn der Lauf sich merkt, was er beim letzten Mal geholt hat:
Jeder Lauf frischt die ältesten `REFRESH_BUDGET` Titel auf und nimmt den Rest
aus diesem Bestand.

Das ist vertretbar, weil die teuren Felder die trägen sind. Gewinn je Aktie,
Verschuldung, Eigenkapital, Branche ändern sich quartalsweise. Was sich täglich
bewegt, ist der Kurs — und der kommt aus dem billigen Endpunkt, taggenau, für
jeden Titel in jedem Lauf.

Zwei Grenzen hält der Bestand selbst ein:

* Einträge älter als `MAX_AGE_DAYS` werden nicht mehr herausgegeben. Fällt der
  Abruf über Tage aus, altert der Bestand aus und die Abdeckung sinkt — der
  Lauf meldet sich dann als unvollständig, statt mit Zahlen von letzter Woche
  einen vollständigen vorzutäuschen.
* Der Bestand ersetzt keinen Abruf, er überbrückt ihn. Deshalb wird immer
  zuerst geholt und nur der Rest ergänzt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from broker.models import Fundamentals

log = logging.getLogger(__name__)

#: So viele Titel frischt ein Lauf auf. Unter dem beobachteten Deckel von rund
#: 330 — mit Abstand, weil der Deckel schwankt und die Kurshistorien der
#: Filter-Überlebenden noch dazukommen.
REFRESH_BUDGET = 280

#: Ab diesem Alter gilt ein Eintrag als unbrauchbar. Eine Woche deckt auch ein
#: langes Wochenende mit gestörten Läufen ab, ohne dass Kennzahlen aus einem
#: anderen Quartal in die Bewertung geraten.
MAX_AGE_DAYS = 7

#: Zeitreihen aus dem Modell — brauchen eigene Behandlung beim Speichern.
_SERIES_FIELDS = (
    "quarterly_eps",
    "quarterly_revenue",
    "quarterly_net_income",
    "shares_history",
)


def _series_to_json(series: pd.Series | None) -> list | None:
    if series is None or series.empty:
        return None
    out = []
    for index, value in series.items():
        try:
            key = pd.Timestamp(index).date().isoformat()
        except Exception:
            key = str(index)
        out.append([key, None if pd.isna(value) else float(value)])
    return out


def _series_from_json(raw: list | None) -> pd.Series | None:
    if not raw:
        return None
    index = pd.to_datetime([entry[0] for entry in raw])
    return pd.Series([entry[1] for entry in raw], index=index)


def to_json(data: Fundamentals) -> dict:
    payload: dict = {}
    for field_ in fields(Fundamentals):
        value = getattr(data, field_.name)
        if field_.name in _SERIES_FIELDS:
            payload[field_.name] = _series_to_json(value)
        else:
            payload[field_.name] = value
    return payload


def from_json(payload: dict) -> Fundamentals:
    kwargs = {}
    for field_ in fields(Fundamentals):
        value = payload.get(field_.name)
        kwargs[field_.name] = (
            _series_from_json(value) if field_.name in _SERIES_FIELDS else value
        )
    return Fundamentals(**kwargs)


@dataclass
class StoredEntry:
    data: Fundamentals
    fetched_at: date

    def age_days(self, today: date) -> int:
        return (today - self.fetched_at).days


class FundamentalsStore:
    """Ein Eintrag je Titel, als JSON Lines auf der Platte."""

    def __init__(self, path: Path | str, max_age_days: int = MAX_AGE_DAYS) -> None:
        self.path = Path(path)
        self.max_age_days = max_age_days
        self.entries: dict[str, StoredEntry] = {}

    # -- Laden und Speichern ------------------------------------------------

    def load(self) -> int:
        """Liest den Bestand. Ein unlesbarer Eintrag wird übersprungen, nicht
        der ganze Bestand verworfen — sonst kostet ein einziges kaputtes Feld
        den Vorrat aller Titel."""
        self.entries = {}
        if not self.path.is_file():
            return 0

        broken = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                entry = StoredEntry(
                    data=from_json(payload["data"]),
                    fetched_at=date.fromisoformat(payload["fetched_at"]),
                )
            except Exception:
                broken += 1
                continue
            self.entries[entry.data.ticker] = entry

        if broken:
            log.warning("%d unlesbare Einträge im Bestand übersprungen.", broken)
        return len(self.entries)

    def save(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "ticker": ticker,
                    "fetched_at": entry.fetched_at.isoformat(),
                    "data": to_json(entry.data),
                },
                ensure_ascii=False,
            )
            for ticker, entry in sorted(self.entries.items())
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    # -- Zugriff ------------------------------------------------------------

    def put(self, data: Fundamentals, day: date | None = None) -> None:
        self.entries[data.ticker] = StoredEntry(
            data=data, fetched_at=day or date.today()
        )

    def get(self, ticker: str, today: date | None = None) -> Fundamentals | None:
        entry = self.entries.get(ticker)
        if entry is None:
            return None
        if entry.age_days(today or date.today()) > self.max_age_days:
            return None
        return entry.data

    def age_of(self, ticker: str, today: date | None = None) -> int | None:
        entry = self.entries.get(ticker)
        if entry is None:
            return None
        return entry.age_days(today or date.today())

    def refresh_order(self, tickers: list[str], today: date | None = None) -> list[str]:
        """Sortiert nach Dringlichkeit: unbekannt zuerst, dann das Älteste.

        Bei gleichem Alter entscheidet der Name — damit ein wiederholter Lauf
        am selben Tag dieselbe Auswahl trifft und nicht bei jedem Versuch eine
        andere Hälfte des Universums auffrischt.
        """
        reference = today or date.today()
        never = date.min

        def key(ticker: str) -> tuple[date, str]:
            entry = self.entries.get(ticker)
            return (entry.fetched_at if entry else never, ticker)

        return sorted(tickers, key=key)

    def drop_unknown(self, tickers: list[str]) -> int:
        """Entfernt Einträge, die nicht mehr im Universum stehen."""
        wanted = set(tickers)
        gone = [t for t in self.entries if t not in wanted]
        for ticker in gone:
            del self.entries[ticker]
        return len(gone)

    def prune(self, today: date | None = None) -> int:
        """Wirft ausgealterte Einträge weg, damit die Datei nicht zumüllt."""
        reference = today or date.today()
        limit = reference - timedelta(days=self.max_age_days)
        stale = [t for t, e in self.entries.items() if e.fetched_at < limit]
        for ticker in stale:
            del self.entries[ticker]
        return len(stale)
