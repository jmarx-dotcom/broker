"""Lädt das zu screenende Aktien-Universum aus den mitgelieferten CSV-Snapshots.

Die Listen sind Momentaufnahmen und veralten — Index-Zugehörigkeiten ändern
sich mehrmals im Jahr. `broker universe refresh` zieht aktuelle Listen, sobald
Netzwerkzugriff besteht; bis dahin ist der Snapshot die Grundlage. Ein Ticker,
den es nicht mehr gibt, kostet nur eine Warnung im Log.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

#: Gruppennamen, die auf der Kommandozeile erlaubt sind.
INDEX_GROUPS: dict[str, tuple[str, ...]] = {
    "dax": ("DAX",),
    "mdax": ("MDAX",),
    "sdax": ("SDAX",),
    "germany": ("DAX", "MDAX", "SDAX"),
    "estoxx": ("ESTOXX",),
    "europe": ("DAX", "MDAX", "SDAX", "ESTOXX"),
    "sp500": ("SP500",),
    "us": ("SP500",),
    "all": ("DAX", "MDAX", "SDAX", "ESTOXX", "SP500"),
}


@dataclass(frozen=True)
class UniverseEntry:
    ticker: str
    index: str
    region: str


def _read_csv(path: Path) -> list[UniverseEntry]:
    if not path.is_file():
        return []
    entries: list[UniverseEntry] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            entries.append(
                UniverseEntry(
                    ticker=ticker,
                    index=(row.get("index") or "").strip(),
                    region=(row.get("region") or "").strip(),
                )
            )
    return entries


def available_indices() -> set[str]:
    return {e.index for e in _all_entries()}


def _all_entries() -> list[UniverseEntry]:
    entries: list[UniverseEntry] = []
    for name in ("europe.csv", "us.csv"):
        entries.extend(_read_csv(DATA_DIR / name))
    return entries


def load_universe(
    group: str = "all", extra_tickers: list[str] | None = None
) -> list[UniverseEntry]:
    """Gibt die Titel der gewünschten Gruppe zurück, dedupliziert und sortiert.

    `group` ist ein Schlüssel aus INDEX_GROUPS oder eine kommagetrennte Liste
    von Index-Namen (z. B. "DAX,SP500").
    """
    key = group.strip().lower()
    if key in INDEX_GROUPS:
        wanted = set(INDEX_GROUPS[key])
    else:
        wanted = {part.strip().upper() for part in group.split(",") if part.strip()}
        if not wanted:
            raise ValueError(f"Unbekannte Universum-Gruppe: {group!r}")

    seen: set[str] = set()
    result: list[UniverseEntry] = []
    for entry in _all_entries():
        if entry.index in wanted and entry.ticker not in seen:
            seen.add(entry.ticker)
            result.append(entry)

    for ticker in extra_tickers or []:
        t = ticker.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(UniverseEntry(ticker=t, index="WATCHLIST", region=""))

    result.sort(key=lambda e: e.ticker)
    return result
