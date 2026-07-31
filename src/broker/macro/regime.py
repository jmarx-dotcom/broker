"""Verdichtet die Makro-Zeitreihen zu einem Regime und Sektor-Scores.

Bewusst deterministisch und nachvollziehbar: Wer den Score eines Sektors nicht
versteht, kann ihn hier Zeile für Zeile nachrechnen. Die LLM-Schicht setzt
darauf auf, ersetzt diesen Teil aber nicht.
"""

from __future__ import annotations

import numpy as np

from broker.macro.sensitivity import SECTOR_SENSITIVITY, sector_label
from broker.models import MacroRegime, MacroSeries


def _direction(change: float | None, threshold: float) -> str:
    if change is None:
        return "neutral"
    if change > threshold:
        return "steigend"
    if change < -threshold:
        return "fallend"
    return "neutral"


def _factor(change: float | None, scale: float) -> float:
    """Skaliert eine Veränderung auf [-1, 1]."""
    if change is None:
        return 0.0
    return float(np.clip(change / scale, -1.0, 1.0))


def neutral_regime(reason: str = "") -> MacroRegime:
    """Regime ohne Daten: alle Sektoren neutral bei 50 Punkten."""
    return MacroRegime(
        sector_scores={sector: 50.0 for sector in SECTOR_SENSITIVITY},
        summary=reason or "Keine Makrodaten verfügbar — Makro-Teil neutral bewertet.",
        live=False,
    )


def build_regime(series: dict[str, MacroSeries]) -> MacroRegime:
    """Baut aus den Zeitreihen Richtungsaussagen und Sektor-Scores."""
    if not series:
        return neutral_regime()

    def get(key: str) -> MacroSeries | None:
        return series.get(key)

    def change_3m(key: str) -> float | None:
        s = get(key)
        return None if s is None else s.change_3m

    def value(key: str) -> float | None:
        s = get(key)
        return None if s is None else s.value

    # Zinsniveau: Leitzins bevorzugt, sonst 10-jährige Rendite.
    rate_change = change_3m("fed_funds")
    if rate_change is None:
        rate_change = change_3m("us_10y")

    curve = value("yield_curve")
    inflation_change = change_3m("us_cpi")
    unemployment_change = change_3m("us_unemployment")
    oil_change = change_3m("oil_brent")
    vix = value("vix")

    rate_direction = _direction(rate_change, 0.25)  # Prozentpunkte
    inflation_trend = _direction(inflation_change, 0.005)  # 0,5 % in 3 Monaten

    if curve is None:
        curve_shape = "neutral"
    elif curve < 0:
        curve_shape = "invers"
    elif curve < 0.5:
        curve_shape = "flach"
    else:
        curve_shape = "steil"

    # Wachstumssignal: steigende Arbeitslosigkeit und inverse Kurve als Bremse.
    growth_points = 0.0
    if unemployment_change is not None:
        growth_points -= float(np.clip(unemployment_change / 0.5, -1.0, 1.0))
    if curve is not None:
        growth_points += float(np.clip(curve / 1.5, -1.0, 1.0)) * 0.5
    if growth_points > 0.3:
        growth_signal = "expansiv"
    elif growth_points < -0.3:
        growth_signal = "restriktiv"
    else:
        growth_signal = "neutral"

    if vix is None:
        risk_appetite = "neutral"
    elif vix > 25:
        risk_appetite = "risk-off"
    elif vix < 15:
        risk_appetite = "risk-on"
    else:
        risk_appetite = "neutral"

    # Faktorwerte für die Sektor-Verrechnung.
    factors = {
        "rates_up": _factor(rate_change, 1.0),
        "curve_steep": 0.0 if curve is None else float(np.clip(curve / 1.5, -1.0, 1.0)),
        "inflation_up": _factor(inflation_change, 0.02),
        "growth_up": float(np.clip(growth_points, -1.0, 1.0)),
        "oil_up": _factor(oil_change, 0.30),
        "risk_off": 0.0 if vix is None else float(np.clip((vix - 18.0) / 12.0, -1.0, 1.0)),
    }

    sector_scores: dict[str, float] = {}
    for sector, sensitivity in SECTOR_SENSITIVITY.items():
        impact = sum(sensitivity.get(f, 0.0) * v for f, v in factors.items())
        # Impact liegt praktisch in [-3, 3]; auf 0-100 um 50 herum abbilden.
        sector_scores[sector] = float(np.clip(50.0 + impact * 16.0, 0.0, 100.0))

    return MacroRegime(
        series=series,
        rate_direction=rate_direction,
        curve_shape=curve_shape,
        inflation_trend=inflation_trend,
        growth_signal=growth_signal,
        risk_appetite=risk_appetite,
        sector_scores=sector_scores,
        summary=_summarize(
            rate_direction, curve_shape, inflation_trend, growth_signal,
            risk_appetite, sector_scores,
        ),
        live=True,
    )


def _summarize(
    rate_direction: str,
    curve_shape: str,
    inflation_trend: str,
    growth_signal: str,
    risk_appetite: str,
    sector_scores: dict[str, float],
) -> str:
    parts = [
        f"Zinsen {rate_direction}",
        f"Renditekurve {curve_shape}",
        f"Inflation {inflation_trend}",
        f"Wachstumssignal {growth_signal}",
        f"Risikoneigung {risk_appetite}",
    ]
    ranked = sorted(sector_scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        best = ", ".join(sector_label(s) for s, _ in ranked[:3])
        worst = ", ".join(sector_label(s) for s, _ in ranked[-3:])
        parts.append(f"Rückenwind: {best}")
        parts.append(f"Gegenwind: {worst}")
    return ". ".join(parts) + "."


def bond_yield_for(regime: MacroRegime, region: str | None) -> float | None:
    """Passende Anleiherendite als Dezimalzahl (0.042 = 4,2 %)."""
    key = "ez_10y" if (region or "").upper() not in ("US", "") else "us_10y"
    series = regime.series.get(key) or regime.series.get("us_10y")
    if series is None or series.value is None:
        return None
    return series.value / 100.0
