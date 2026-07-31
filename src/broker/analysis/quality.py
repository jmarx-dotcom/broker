"""Bilanz- und Trendqualität — der Value-Fallen-Filter.

Ohne diesen Schritt findet ein KGV-Screener zuverlässig genau die Titel, die
zu Recht billig sind: hoch verschuldete Unternehmen mit schrumpfendem Umsatz
und einer Dividende, die sie sich nicht leisten können. Hier werden solche
Fälle mit Punktabzug und expliziten Warnungen versehen, statt sie stillschweigend
durchzulassen.
"""

from __future__ import annotations

import numpy as np

from broker.models import Fundamentals, QualityResult


def analyze_quality(fundamentals: Fundamentals) -> QualityResult:
    """Score 0-100, höher = solidere Substanz."""
    f = fundamentals
    red_flags: list[str] = []
    notes: list[str] = []
    components: list[tuple[float, float]] = []

    # Verschuldung ---------------------------------------------------------
    nd_ebitda = f.net_debt_to_ebitda
    if nd_ebitda is not None:
        if nd_ebitda <= 0:
            debt_score = 100.0  # Nettoliquidität
        elif nd_ebitda >= 5:
            debt_score = 0.0
        else:
            debt_score = 100.0 - nd_ebitda * 20.0
        components.append((debt_score, 0.30))
        if nd_ebitda > 4:
            red_flags.append(
                f"Nettoverschuldung beim {nd_ebitda:.1f}-fachen EBITDA — "
                "wenig Puffer bei Gewinnrückgang."
            )
        elif nd_ebitda <= 0:
            notes.append("Nettoliquidität — mehr Cash als Schulden.")
    else:
        notes.append("Keine belastbaren Verschuldungsdaten.")

    # Umsatzentwicklung ----------------------------------------------------
    if f.revenue_growth is not None:
        # -20% -> 0 Punkte, 0% -> 50, +20% -> 100
        components.append((float(np.clip(50.0 + f.revenue_growth * 250.0, 0, 100)), 0.20))
        if f.revenue_growth < -0.10:
            red_flags.append(
                f"Umsatz schrumpft um {abs(f.revenue_growth) * 100:.0f}% — "
                "das Geschäft selbst steht unter Druck."
            )

    # Gewinnentwicklung ----------------------------------------------------
    if f.earnings_growth is not None:
        components.append((float(np.clip(50.0 + f.earnings_growth * 125.0, 0, 100)), 0.20))
        if f.earnings_growth < -0.25:
            red_flags.append(
                f"Gewinn bricht um {abs(f.earnings_growth) * 100:.0f}% ein — "
                "das aktuelle KGV ist damit optisch zu niedrig."
            )

    # Kapitalrendite -------------------------------------------------------
    if f.return_on_equity is not None:
        components.append((float(np.clip(f.return_on_equity * 400.0, 0, 100)), 0.15))
        if f.return_on_equity < 0:
            red_flags.append("Negative Eigenkapitalrendite.")

    # Marge ----------------------------------------------------------------
    if f.profit_margin is not None:
        components.append((float(np.clip(f.profit_margin * 500.0, 0, 100)), 0.15))
        if f.profit_margin < 0:
            red_flags.append("Negative Nettomarge — das Unternehmen schreibt Verluste.")

    # Cashflow und Ausschüttung: keine eigenen Punkte, aber harte Warnungen.
    fcf_positive = None if f.free_cashflow is None else f.free_cashflow > 0
    if fcf_positive is False:
        red_flags.append("Negativer freier Cashflow.")

    if f.payout_ratio is not None and f.payout_ratio > 1.0:
        red_flags.append(
            f"Ausschüttungsquote {f.payout_ratio * 100:.0f}% — die Dividende wird "
            "aus der Substanz gezahlt und ist gefährdet."
        )

    if not components:
        return QualityResult(
            score=50.0,
            red_flags=red_flags,
            notes=["Zu wenig Fundamentaldaten für eine Qualitätsbewertung."],
        )

    total_weight = sum(w for _, w in components)
    score = sum(s * w for s, w in components) / total_weight

    # Jede rote Flagge kostet zusätzlich, damit sich Probleme nicht
    # gegenseitig wegmitteln.
    score *= max(0.4, 1.0 - 0.12 * len(red_flags))

    return QualityResult(
        score=float(np.clip(score, 0, 100)),
        net_debt_to_ebitda=None if nd_ebitda is None else round(nd_ebitda, 2),
        revenue_growth=f.revenue_growth,
        earnings_growth=f.earnings_growth,
        return_on_equity=f.return_on_equity,
        profit_margin=f.profit_margin,
        payout_ratio=f.payout_ratio,
        free_cashflow_positive=fcf_positive,
        red_flags=red_flags,
        notes=notes,
    )
