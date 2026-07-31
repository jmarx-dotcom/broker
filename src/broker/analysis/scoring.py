"""Verrechnet die Einzelscores zum Gesamtscore."""

from __future__ import annotations

import numpy as np

from broker.config import Weights
from broker.models import QualityResult, TechnicalResult, ValuationResult


def combine_scores(
    valuation: ValuationResult,
    technical: TechnicalResult,
    quality: QualityResult,
    macro_score: float,
    weights: Weights,
) -> float:
    """Gewichteter Gesamtscore 0-100.

    Die Qualität wirkt zusätzlich als Multiplikator: ein Titel mit sehr
    schwacher Substanz soll nicht über einen hohen Bewertungsscore nach oben
    gemittelt werden. Genau dieser Effekt macht naive Value-Screener nutzlos.
    """
    total_weight = weights.value + weights.quality + weights.technical + weights.macro
    if total_weight <= 0:
        raise ValueError("Die Summe der Gewichte muss positiv sein.")

    base = (
        valuation.score * weights.value
        + quality.score * weights.quality
        + technical.score * weights.technical
        + macro_score * weights.macro
    ) / total_weight

    # Qualitätsdämpfung: unter 40 Punkten wird spürbar abgewertet.
    if quality.score < 40:
        base *= 0.75 + 0.25 * (quality.score / 40.0)

    return float(np.clip(base, 0, 100))
