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


def _winsorize(values: list[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    """Stutzt Extremwerte auf die Randperzentile, statt sie zu entfernen.

    Ein einzelner Titel mit KGV 95 verschiebt den Median einer kleinen Branche
    spürbar. Die harte Kappung bei PE_CAP fängt nur absurde Werte; hier werden
    die Ränder auf ein realistisches Maß zurückgeholt, ohne die Stichprobe zu
    verkleinern — bei acht Vergleichswerten zählt jeder einzelne.
    """
    if len(values) < 5:
        return values  # zu klein, um Ränder sinnvoll zu bestimmen
    ordered = sorted(values)
    low = float(np.quantile(ordered, lower))
    high = float(np.quantile(ordered, upper))
    return [min(max(v, low), high) for v in values]


#: Die Bewertungsmaße, für die Branchenmediane gebildet werden.
#: (Schlüssel, Zugriff, plausible Ober-/Untergrenze, "kleiner ist günstiger")
METRICS: tuple[tuple[str, str, float, float, bool], ...] = (
    ("pe", "trailing_pe", 0.0, PE_CAP, True),
    ("ev_ebitda", "ev_to_ebitda", 0.0, 60.0, True),
    ("pb", "price_to_book", 0.0, 25.0, True),
    ("fcf_yield", "fcf_yield", -1.0, 1.0, False),  # größer ist günstiger
)


def _metric_value(f: Fundamentals, attribute: str) -> float | None:
    value = getattr(f, attribute, None)
    return value if isinstance(value, (int, float)) else None


def sector_medians(
    fundamentals: list[Fundamentals], min_peers: int = 4
) -> dict[str, dict[str, float]]:
    """Branchenmediane je Bewertungsmaß: {Sektor: {Maß: Median}}.

    Sektoren mit zu wenigen Vergleichswerten fallen raus — ein "Branchenmedian"
    aus zwei Titeln ist kein Median, sondern Rauschen.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for f in fundamentals:
        if not f.sector:
            continue
        for key, attribute, low, high, _ in METRICS:
            value = _metric_value(f, attribute)
            if value is None or not (low < value < high):
                continue
            buckets.setdefault(f.sector, {}).setdefault(key, []).append(value)

    result: dict[str, dict[str, float]] = {}
    for sector, metrics in buckets.items():
        medians = {
            key: float(median(_winsorize(values)))
            for key, values in metrics.items()
            if len(values) >= min_peers
        }
        if medians:
            result[sector] = medians
    return result


def sector_median_pe(
    fundamentals: list[Fundamentals], min_peers: int = 4
) -> dict[str, float]:
    """Nur die KGV-Mediane — schmale Sicht auf sector_medians()."""
    return {
        sector: metrics["pe"]
        for sector, metrics in sector_medians(fundamentals, min_peers).items()
        if "pe" in metrics
    }


def _score_percentile(percentile: float) -> float:
    """Perzentil 0 (billigst je) -> 100 Punkte, Perzentil 100 -> 0 Punkte."""
    return max(0.0, min(100.0, 100.0 - percentile))


def _score_ratio_below_one(ratio: float) -> float:
    """Verhältnis < 1 ist günstig. 0.5 -> 100, 1.0 -> 50, 1.5+ -> 0."""
    return max(0.0, min(100.0, 150.0 - 100.0 * ratio))


def _compare_to_sector(
    value: float | None, sector_median: float | None, lower_is_cheaper: bool
) -> tuple[float | None, float | None, bool | None]:
    """Gibt (Verhältnis, Score, ist_guenstig) für ein Maß gegen die Branche zurück."""
    if value is None or sector_median is None or sector_median == 0:
        return None, None, None

    if lower_is_cheaper:
        if value <= 0:
            return None, None, None
        ratio = value / sector_median
        return round(ratio, 2), _score_ratio_below_one(ratio), ratio < 1.0

    # Größer ist günstiger (Cashflow-Rendite): Verhältnis umdrehen, damit
    # dieselbe Punkteskala gilt.
    if sector_median <= 0:
        # Branche schreibt im Median negativen Cashflow — ein positiver Wert
        # ist dann günstig, aber das Verhältnis wäre nicht interpretierbar.
        return None, (80.0 if value > 0 else 20.0), value > 0
    ratio = sector_median / value if value > 0 else None
    if ratio is None:
        return None, 10.0, False
    return round(value / sector_median, 2), _score_ratio_below_one(ratio), value > sector_median


def analyze_valuation(
    fundamentals: Fundamentals,
    history: PriceHistory,
    sector_medians: dict[str, float] | dict[str, dict[str, float]] | None = None,
    bond_yield: float | None = None,
) -> ValuationResult:
    """Bewertet einen Titel. Score 0-100, höher = günstiger relativ zum Kontext.

    `sector_medians` akzeptiert beide Formen: das schmale {Sektor: KGV-Median}
    und das breite {Sektor: {Maß: Median}}.
    """
    notes: list[str] = []
    components: list[tuple[float, float]] = []  # (Score, Gewicht)

    # Beide Eingabeformen auf die breite Form vereinheitlichen.
    medians_by_sector: dict[str, dict[str, float]] = {}
    for sector, value in (sector_medians or {}).items():
        medians_by_sector[sector] = (
            value if isinstance(value, dict) else {"pe": float(value)}
        )
    own_medians = medians_by_sector.get(fundamentals.sector or "", {})

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
        components.append((_score_percentile(percentile), 0.25))

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
    sector_pe = own_medians.get("pe")
    if sector_pe and trailing_pe is not None and sector_pe > 0:
        ratio = trailing_pe / sector_pe
        result.sector_median_pe = round(sector_pe, 1)
        result.pe_vs_sector_median = round(ratio, 2)
        components.append((_score_ratio_below_one(ratio), 0.20))
        if ratio <= 0.7:
            notes.append(
                f"KGV {(1 - ratio) * 100:.0f}% unter dem Branchenmedian "
                f"({sector_pe:.1f})."
            )
    else:
        notes.append("Kein belastbarer Branchenmedian verfügbar.")

    # 2b) Die übrigen Bewertungsmaße gegen die Branche --------------------
    # Das KGV allein ist anfällig für Einmaleffekte im Gewinn. EV/EBITDA ist
    # unabhängig von Kapitalstruktur und Abschreibungen, die Cashflow-Rendite
    # ist schwerer zu schönen, und das KBV greift auch dort, wo der Gewinn
    # gerade nichts aussagt.
    cheap_flags: list[bool] = []
    if trailing_pe is not None and sector_pe:
        cheap_flags.append(trailing_pe < sector_pe)

    result.ev_to_ebitda = (
        None if fundamentals.ev_to_ebitda is None else round(fundamentals.ev_to_ebitda, 1)
    )
    ratio, score, is_cheap = _compare_to_sector(
        fundamentals.ev_to_ebitda, own_medians.get("ev_ebitda"), True
    )
    result.ev_ebitda_vs_sector = ratio
    if score is not None:
        components.append((score, 0.15))
    if is_cheap is not None:
        cheap_flags.append(is_cheap)

    result.fcf_yield = (
        None if fundamentals.fcf_yield is None else round(fundamentals.fcf_yield, 4)
    )
    ratio, score, is_cheap = _compare_to_sector(
        fundamentals.fcf_yield, own_medians.get("fcf_yield"), False
    )
    result.fcf_yield_vs_sector = ratio
    if score is not None:
        components.append((score, 0.10))
    if is_cheap is not None:
        cheap_flags.append(is_cheap)

    result.price_to_book = fundamentals.price_to_book
    ratio, score, is_cheap = _compare_to_sector(
        fundamentals.price_to_book, own_medians.get("pb"), True
    )
    result.pb_vs_sector = ratio
    if score is not None:
        components.append((score, 0.10))
    if is_cheap is not None:
        cheap_flags.append(is_cheap)

    result.cheap_measures = sum(1 for flag in cheap_flags if flag)
    result.comparable_measures = len(cheap_flags)
    if result.comparable_measures >= 3:
        if result.cheap_measures == result.comparable_measures:
            notes.append(
                f"Alle {result.comparable_measures} vergleichbaren "
                "Bewertungsmaße liegen unter dem Branchenschnitt — die "
                "Bewertung ist breit niedrig, nicht nur beim KGV."
            )
        elif result.cheap_measures <= 1:
            notes.append(
                f"Nur {result.cheap_measures} von {result.comparable_measures} "
                "Bewertungsmaßen sind günstig. Ein einzelnes niedriges Maß "
                "geht oft auf Einmaleffekte zurück."
            )

    # 3) Forward gegen Trailing ------------------------------------------
    if trailing_pe is not None and forward_pe is not None:
        ratio = forward_pe / trailing_pe
        # Forward deutlich unter Trailing = Analysten erwarten Gewinnwachstum.
        components.append((_score_ratio_below_one(ratio), 0.10))
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
            components.append((max(0.0, min(100.0, 40.0 + excess * 1000.0)), 0.10))
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
