"""Sektor-Sensitivität gegenüber Makro-Faktoren.

Die Tabelle ist eine bewusst grobe, fest kodierte Heuristik über breit belegte
Zusammenhänge — steigende Zinsen belasten zinssensitive Sektoren wie Immobilien
und Versorger, während Banken von einer steileren Kurve profitieren. Sie ersetzt
keine Analyse einzelner Unternehmen und soll das auch nicht: sie fließt mit nur
10 % in den Gesamtscore ein und dient primär als Kontext im Report.

Werte: +1 = profitiert stark, 0 = neutral, -1 = leidet stark.
Sektornamen folgen der Yahoo-/GICS-Nomenklatur.
"""

from __future__ import annotations

#: Faktoren: rates_up, curve_steep, inflation_up, growth_up, oil_up, risk_off
SECTOR_SENSITIVITY: dict[str, dict[str, float]] = {
    "Technology": {
        "rates_up": -0.7, "curve_steep": 0.0, "inflation_up": -0.4,
        "growth_up": 0.8, "oil_up": -0.1, "risk_off": -0.7,
    },
    "Financial Services": {
        "rates_up": 0.5, "curve_steep": 0.8, "inflation_up": 0.1,
        "growth_up": 0.6, "oil_up": 0.0, "risk_off": -0.5,
    },
    "Real Estate": {
        "rates_up": -0.9, "curve_steep": -0.3, "inflation_up": 0.2,
        "growth_up": 0.4, "oil_up": -0.1, "risk_off": -0.3,
    },
    "Utilities": {
        "rates_up": -0.7, "curve_steep": -0.2, "inflation_up": -0.2,
        "growth_up": 0.0, "oil_up": -0.2, "risk_off": 0.6,
    },
    "Consumer Defensive": {
        "rates_up": -0.2, "curve_steep": 0.0, "inflation_up": -0.3,
        "growth_up": 0.1, "oil_up": -0.2, "risk_off": 0.7,
    },
    "Consumer Cyclical": {
        "rates_up": -0.6, "curve_steep": 0.1, "inflation_up": -0.6,
        "growth_up": 0.9, "oil_up": -0.4, "risk_off": -0.7,
    },
    "Healthcare": {
        "rates_up": -0.2, "curve_steep": 0.0, "inflation_up": -0.1,
        "growth_up": 0.2, "oil_up": 0.0, "risk_off": 0.5,
    },
    "Industrials": {
        "rates_up": -0.3, "curve_steep": 0.2, "inflation_up": -0.3,
        "growth_up": 0.8, "oil_up": -0.3, "risk_off": -0.5,
    },
    "Energy": {
        "rates_up": 0.0, "curve_steep": 0.1, "inflation_up": 0.6,
        "growth_up": 0.5, "oil_up": 0.9, "risk_off": -0.1,
    },
    "Basic Materials": {
        "rates_up": -0.3, "curve_steep": 0.2, "inflation_up": 0.4,
        "growth_up": 0.8, "oil_up": 0.1, "risk_off": -0.5,
    },
    "Communication Services": {
        "rates_up": -0.4, "curve_steep": 0.0, "inflation_up": -0.2,
        "growth_up": 0.5, "oil_up": -0.1, "risk_off": -0.3,
    },
}

#: Deutsche Bezeichnungen für den Report.
SECTOR_LABELS_DE: dict[str, str] = {
    "Technology": "Technologie",
    "Financial Services": "Finanzwesen",
    "Real Estate": "Immobilien",
    "Utilities": "Versorger",
    "Consumer Defensive": "Basiskonsum",
    "Consumer Cyclical": "Zyklischer Konsum",
    "Healthcare": "Gesundheit",
    "Industrials": "Industrie",
    "Energy": "Energie",
    "Basic Materials": "Grundstoffe",
    "Communication Services": "Kommunikation",
}


def sector_label(sector: str | None) -> str:
    if not sector:
        return "Unbekannt"
    return SECTOR_LABELS_DE.get(sector, sector)
