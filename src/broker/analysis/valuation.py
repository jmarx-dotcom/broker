"""KGV-Analyse.

Der Kern des Tools — und die Stelle, an der ein naiver Screener scheitert. Ein
niedriges KGV ist für sich genommen kein Kaufsignal: der Markt preist damit
meistens einbrechende Gewinne ein, bevor sie in den Zahlen stehen. "Günstig"
heißt hier deshalb immer *relativ*:

  1. gegen die eigene KGV-Historie (Perzentil der letzten Jahre)
  2. gegen den Median der Branche
  3. Trailing gegen Forward (fällt das erwartete KGV, wächst der Gewinn)
  4. Gewinnrendite gegen die Anleiherendite (lohnt sich Aktienrisiko überhaupt)

Erst die Kombination trennt "billig, weil unbeliebt" von "billig, weil kaputt".
"""

from __future__ import annotations

import math
from statistics import median

import numpy as np
import pandas as pd

from broker.models import Fundamentals, PriceHistory, ValuationResult

#: Obergrenze für ein noch sinnvoll interpretierbares KGV.
PE_CAP = 100.0


def build_pe_history(
    history: PriceHistory, quarterly_eps: pd.Series | None
) -> pd.Series | None:
    """Historische KGV-Reihe aus Kursen und rollierendem 12-Monats-EPS.

    Gibt None zurück, wenn zu wenig Quartale vorliegen — ohne mindestens
    ~2 Jahre Historie ist ein Perzentil-Vergleich bedeutungslos.
    """
    if quarterly_eps is None or len(quarterly_eps) < 8:
        return None

    ttm = quarterly_eps.sort_index().rolling(4).sum().dropna()
    if ttm.empty:
        return None

    close = history.close
    if close.empty:
        return None

    # Zeitzonen angleichen, sonst scheitert das Reindexing.
    close_index = close.index
    if getattr(close_index, "tz", None) is not None:
        close_index = close_index.tz_localize(None)
    close = pd.Series(close.to_numpy(), index=close_index)

    ttm_index = ttm.index
    if getattr(ttm_index, "tz", None) is not None:
        ttm_index = ttm_index.tz_localize(None)
    ttm = pd.Series(ttm.to_numpy(), index=ttm_index)

    # EPS gilt ab Veröffentlichung bis zum nächsten Quartal.
    aligned = ttm.reindex(close.index.union(ttm.index)).ffill().reindex(close.index)
    aligned = aligned[aligned > 0]
    if aligned.empty:
        return None

    pe = (close.reindex(aligned.index) / aligned).dropna()
    pe = pe[(pe > 0) & (pe < PE_CAP)]
    return pe if len(pe) >= 60 else None


def sector_median_pe(
    fundamentals: list[Fundamentals], min_peers: int = 4
) -> dict[str, float]:
    """Median-KGV je Sektor über das gesamte Universum.

    Sektoren mit zu wenigen Vergleichswerten fallen raus — ein "Branchenmedian"
    aus zwei Titeln ist kein Median, sondern Rauschen.
    """
    buckets: dict[str, list[float]] = {}
    for f in fundamentals:
        pe = f.trailing_pe
        if not f.sector or pe is None or pe <= 0 or pe >= PE_CAP:
            continue
        buckets.setdefault(f.sector, []).append(pe)

    return {
        sector: float(median(values))
        for sector, values in buckets.items()
        if len(values) >= min_peers
    }


def _score_percentile(percentile: float) -> float:
    """Perzentil 0 (billigst je) -> 100 Punkte, Perzentil 100 -> 0 Punkte."""
    return max(0.0, min(100.0, 100.0 - percentile))


def _score_ratio_below_one(ratio: float) -> float:
    """Verhältnis < 1 ist günstig. 0.5 -> 100, 1.0 -> 50, 1.5+ -> 0."""
    return max(0.0, min(100.0, 150.0 - 100.0 * ratio))


def analyze_valuation(
    fundamentals: Fundamentals,
    history: PriceHistory,
    sector_medians: dict[str, float] | None = None,
    bond_yield: float | None = None,
) -> ValuationResult:
    """Bewertet einen Titel. Score 0-100, höher = günstiger relativ zum Kontext."""
    notes: list[str] = []
    components: list[tuple[float, float]] = []  # (Score, Gewicht)

    trailing_pe = fundamentals.trailing_pe
    forward_pe = fundamentals.forward_pe

    # Ein negatives KGV bedeutet Verlust — dann ist die ganze Kennzahl sinnlos.
    if trailing_pe is not None and trailing_pe <= 0:
        notes.append("Negatives KGV (Verlust) — KGV-Vergleich nicht aussagekräftig.")
        trailing_pe = None
    if forward_pe is not None and forward_pe <= 0:
        forward_pe = None

    result = ValuationResult(
        score=0.0, trailing_pe=trailing_pe, forward_pe=forward_pe, notes=notes
    )

    # 1) KGV gegen die eigene Historie -----------------------------------
    pe_history = build_pe_history(history, fundamentals.quarterly_eps)
    if pe_history is not None and trailing_pe is not None:
        percentile = float((pe_history < trailing_pe).mean() * 100.0)
        own_median = float(pe_history.median())
        result.pe_percentile_own_history = round(percentile, 1)
        if own_median > 0:
            result.pe_vs_own_median = round(trailing_pe / own_median, 2)
        components.append((_score_percentile(percentile), 0.35))

        if percentile <= 20:
            notes.append(
                f"KGV im untersten Fünftel der eigenen Historie "
                f"(Perzentil {percentile:.0f})."
            )
        elif percentile >= 80:
            notes.append(
                f"KGV im obersten Fünftel der eigenen Historie "
                f"(Perzentil {percentile:.0f}) — teuer gegen sich selbst."
            )
    else:
        notes.append("Zu wenig Gewinnhistorie für einen Vergleich mit dem eigenen KGV.")

    # 2) KGV gegen den Branchenmedian ------------------------------------
    medians = sector_medians or {}
    sector_pe = medians.get(fundamentals.sector or "")
    if sector_pe and trailing_pe is not None and sector_pe > 0:
        ratio = trailing_pe / sector_pe
        result.sector_median_pe = round(sector_pe, 1)
        result.pe_vs_sector_median = round(ratio, 2)
        components.append((_score_ratio_below_one(ratio), 0.25))
        if ratio <= 0.7:
            notes.append(
                f"KGV {(1 - ratio) * 100:.0f}% unter dem Branchenmedian "
                f"({sector_pe:.1f})."
            )
    else:
        notes.append("Kein belastbarer Branchenmedian verfügbar.")

    # 3) Forward gegen Trailing ------------------------------------------
    if trailing_pe is not None and forward_pe is not None:
        ratio = forward_pe / trailing_pe
        # Forward deutlich unter Trailing = Analysten erwarten Gewinnwachstum.
        components.append((_score_ratio_below_one(ratio), 0.20))
        if ratio <= 0.85:
            notes.append(
                "Erwartetes KGV klar unter dem aktuellen — Analysten rechnen "
                "mit steigenden Gewinnen."
            )
        elif ratio >= 1.15:
            notes.append(
                "Erwartetes KGV über dem aktuellen — Analysten rechnen mit "
                "fallenden Gewinnen. Klassische Value-Falle."
            )
    elif forward_pe is None:
        notes.append("Keine Analystenschätzung (Forward-KGV) verfügbar.")

    # 4) Gewinnrendite gegen Anleiherendite ------------------------------
    base_pe = forward_pe or trailing_pe
    if base_pe:
        earnings_yield = 1.0 / base_pe
        result.earnings_yield = round(earnings_yield, 4)
        if bond_yield is not None:
            excess = earnings_yield - bond_yield
            result.excess_yield_vs_bond = round(excess, 4)
            # 0% Aufschlag -> 40 Punkte, 6% Aufschlag -> 100 Punkte.
            components.append((max(0.0, min(100.0, 40.0 + excess * 1000.0)), 0.20))
            if excess < 0:
                notes.append(
                    "Gewinnrendite unter der Anleiherendite — für das Aktienrisiko "
                    "gibt es keine Prämie."
                )

    # 5) PEG als Zusatzinformation (fließt nicht in den Score) ------------
    growth = fundamentals.earnings_growth
    if base_pe and growth and growth > 0:
        peg = base_pe / (growth * 100.0)
        if math.isfinite(peg):
            result.peg = round(peg, 2)

    if not components:
        result.score = 0.0
        notes.append("Keine belastbare Bewertungsgrundlage — Titel nicht bewertbar.")
        return result

    total_weight = sum(weight for _, weight in components)
    result.score = float(
        np.clip(sum(score * weight for score, weight in components) / total_weight, 0, 100)
    )
    return result
