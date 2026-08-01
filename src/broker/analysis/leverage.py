"""Risikorechnung für Hebelprodukte auf einen Basiswert.

Was dieses Modul *nicht* tut: Hebelprodukte screenen. Ein Knock-out hat kein
KGV, keine Bilanz und keinen Sektor — sein Preis ist eine mechanische Funktion
des Basiswerts, der Finanzierungskosten und des Emittenten-Spreads. Das
Fundamentalmodul ist auf ihn nicht anwendbar, und die Produktdaten der
deutschen Emittenten liegen in keiner der hier genutzten Quellen.

Was es tut: Für einen Basiswert, den der Screener gefunden hat, ausrechnen,
welcher Hebel überhaupt zu dessen Schwankungsbreite passt und was er kostet.
Alle drei Rechnungen brauchen nur die Volatilität, die ohnehin schon berechnet
wird:

  * **Volatilitätsdrag** — der am meisten unterschätzte Effekt bei
    Faktor-Zertifikaten. Ein Faktor-4-Papier auf einen seitwärts schwankenden
    Basiswert verliert *mechanisch* Geld, auch wenn der Basiswert am Ende
    unverändert dasteht.
  * **Knock-out-Wahrscheinlichkeit** — nicht, ob der Kurs am Ende unter der
    Schwelle liegt, sondern ob er sie *irgendwann unterwegs* berührt. Das ist
    deutlich wahrscheinlicher, und es ist der Unterschied zwischen "These war
    richtig" und "Totalverlust".
  * **Maximal sinnvoller Hebel** aus Schwankungsbreite und Verlusttoleranz.

Alle Zahlen setzen ein Standardmodell voraus (geometrische Brownsche Bewegung
mit konstanter Volatilität). Echte Kurse springen und haben schwerere Ränder —
die tatsächlichen Knock-out-Wahrscheinlichkeiten liegen daher eher *über* den
berechneten. Sie sind eine Untergrenze, keine Prognose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TRADING_DAYS = 252


# -- Normalverteilung ohne scipy -------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse Standardnormalverteilung (Acklam-Approximation, ~1e-9 genau)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss zwischen 0 und 1 liegen")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# -- Die drei Kernrechnungen -----------------------------------------------


def volatility_drag(factor: float, volatility: float, days: int) -> float:
    """Erwarteter Verlust eines Faktor-Zertifikats bei *unverändertem* Basiswert.

    Ein Papier mit täglichem Reset auf den Faktor k wächst im Log um
    ``k·µ − ½k²σ²``, der Basiswert um ``µ − ½σ²``. Die Differenz zum
    k-fachen des Basiswerts ist ``−½·σ²·k·(k−1)`` pro Jahr — unabhängig von
    der Richtung und allein durch die Schwankung verursacht.

    Gibt die einfache Rendite zurück (−0.18 = 18 % Verlust).
    """
    if volatility <= 0 or days <= 0 or factor <= 1:
        return 0.0
    years = days / TRADING_DAYS
    log_drag = -0.5 * volatility**2 * factor * (factor - 1.0) * years
    return math.expm1(log_drag)


def knockout_probability(
    barrier_distance: float, volatility: float, days: int
) -> float | None:
    """Wahrscheinlichkeit, die Schwelle *innerhalb* der Haltedauer zu berühren.

    `barrier_distance` ist der relative Abstand zur Barriere (0.15 = 15 %
    unter dem aktuellen Kurs).

    Gerechnet wird ohne Drift. Das ist Absicht: Die Richtung ist unbekannt,
    und eine unterstellte positive Drift würde das Risiko kleinrechnen. Unter
    dieser Annahme gilt das Spiegelungsprinzip, also
    ``P = 2·Φ(−|ln(1−d)| / (σ√T))``.
    """
    if not 0 < barrier_distance < 1 or volatility <= 0 or days <= 0:
        return None
    years = days / TRADING_DAYS
    log_distance = abs(math.log(1.0 - barrier_distance))
    z = log_distance / (volatility * math.sqrt(years))
    return min(1.0, 2.0 * _norm_cdf(-z))


def barrier_for_probability(
    target_probability: float, volatility: float, days: int
) -> float | None:
    """Umkehrung: Welcher Abstand hält die Knock-out-Gefahr unter `target`?"""
    if not 0 < target_probability < 1 or volatility <= 0 or days <= 0:
        return None
    years = days / TRADING_DAYS
    z = -_norm_ppf(target_probability / 2.0)
    log_distance = z * volatility * math.sqrt(years)
    return 1.0 - math.exp(-log_distance)


def max_sensible_leverage(
    volatility: float, days: int, loss_tolerance: float, confidence: float = 0.95
) -> float | None:
    """Größter Hebel, bei dem der Verlust die Toleranz mit `confidence` einhält.

    `loss_tolerance` ist der Anteil des Einsatzes, den man zu verlieren bereit
    ist (0.3 = 30 %). Gerechnet wird gegen die ungünstige Seite der Verteilung,
    nicht gegen den Erwartungswert.
    """
    if volatility <= 0 or days <= 0 or not 0 < loss_tolerance < 1:
        return None
    years = days / TRADING_DAYS
    z = _norm_ppf(confidence)
    adverse_move = -math.expm1(-z * volatility * math.sqrt(years))
    if adverse_move <= 0:
        return None
    return loss_tolerance / adverse_move


def financing_cost(factor: float, annual_rate: float, days: int) -> float:
    """Finanzierungskosten bezogen auf das eingesetzte Kapital.

    Bei Hebel k ist (k−1) mal der Einsatz fremdfinanziert. Der Emittent stellt
    dafür Referenzzins plus Aufschlag in Rechnung.
    """
    if factor <= 1 or days <= 0:
        return 0.0
    return (factor - 1.0) * annual_rate * (days / TRADING_DAYS)


# -- Zusammenfassung --------------------------------------------------------


@dataclass
class LeverageAssessment:
    ticker: str
    volatility: float
    days: int
    factor: float
    #: Verlust allein durch Schwankung, bei unverändertem Basiswert.
    drag: float = 0.0
    financing: float = 0.0
    total_holding_cost: float = 0.0
    #: Knock-out-Wahrscheinlichkeit bei typischen Barriere-Abständen.
    knockout_risk: dict[str, float] = field(default_factory=dict)
    safe_barrier_10pct: float | None = None
    max_leverage: float | None = None
    notes: list[str] = field(default_factory=list)


def assess_leverage(
    ticker: str,
    volatility: float | None,
    days: int = 60,
    factor: float = 3.0,
    annual_financing_rate: float = 0.06,
    loss_tolerance: float = 0.30,
) -> LeverageAssessment | None:
    """Fasst die Hebelrechnungen für einen Basiswert zusammen."""
    if volatility is None or volatility <= 0:
        return None

    result = LeverageAssessment(
        ticker=ticker, volatility=volatility, days=days, factor=factor
    )

    result.drag = volatility_drag(factor, volatility, days)
    result.financing = financing_cost(factor, annual_financing_rate, days)
    result.total_holding_cost = result.drag - result.financing

    for label, distance in (("10 %", 0.10), ("20 %", 0.20), ("30 %", 0.30)):
        probability = knockout_probability(distance, volatility, days)
        if probability is not None:
            result.knockout_risk[label] = probability

    result.safe_barrier_10pct = barrier_for_probability(0.10, volatility, days)
    result.max_leverage = max_sensible_leverage(volatility, days, loss_tolerance)

    # Klartext dazu, damit die Zahlen nicht falsch gelesen werden.
    if result.drag <= -0.10:
        result.notes.append(
            f"Allein durch die Schwankung verliert ein Faktor-{factor:g}-Papier "
            f"über {days} Handelstage rund {abs(result.drag) * 100:.0f} % — "
            f"auch wenn der Basiswert am Ende unverändert notiert."
        )
    near = result.knockout_risk.get("20 %")
    if near is not None and near >= 0.25:
        result.notes.append(
            f"Bei 20 % Barriere-Abstand liegt die Knock-out-Wahrscheinlichkeit "
            f"bei {near * 100:.0f} %. Die These kann aufgehen und die Position "
            f"trotzdem vorher ausgestoppt sein."
        )
    if result.max_leverage is not None and result.max_leverage < factor:
        result.notes.append(
            f"Zur Schwankungsbreite dieses Titels passt bei {loss_tolerance * 100:.0f} % "
            f"Verlusttoleranz höchstens Hebel {result.max_leverage:.1f} — "
            f"gerechnet wurde mit {factor:g}."
        )
    if volatility > 0.50:
        result.notes.append(
            f"Volatilität von {volatility * 100:.0f} % p.a. — in dieser "
            f"Größenordnung sind Hebelprodukte auf Sicht von Wochen eher eine "
            f"Wette auf den Zeitpunkt als auf die Bewertung."
        )
    return result
