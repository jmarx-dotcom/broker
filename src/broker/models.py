"""Datenmodelle, die zwischen den Schichten wandern.

Bewusst als schlichte Dataclasses gehalten: die Provider-Schicht füllt sie,
die Analyse-Schicht liest sie. Alle Kennzahlen sind optional, weil jede
Datenquelle Lücken hat — die Analyse muss damit umgehen, nicht abstürzen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class Fundamentals:
    """Fundamentaldaten eines Titels zum Abrufzeitpunkt."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None

    trailing_eps: float | None = None
    forward_eps: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None

    total_debt: float | None = None
    total_cash: float | None = None
    ebitda: float | None = None
    free_cashflow: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    return_on_equity: float | None = None
    profit_margin: float | None = None

    #: Quartals-EPS (Index: Periodenende, Wert: EPS) für die KGV-Historie.
    quarterly_eps: pd.Series | None = None

    @property
    def net_debt(self) -> float | None:
        if self.total_debt is None:
            return None
        return self.total_debt - (self.total_cash or 0.0)

    @property
    def net_debt_to_ebitda(self) -> float | None:
        nd, e = self.net_debt, self.ebitda
        if nd is None or e is None or e <= 0:
            return None
        return nd / e

    @property
    def enterprise_value(self) -> float | None:
        """Unternehmenswert: Marktkapitalisierung plus Nettoverschuldung.

        Der Vorteil gegenüber der reinen Marktkapitalisierung: Zwei Firmen mit
        gleichem Börsenwert sind unterschiedlich teuer, wenn die eine schuldenfrei
        ist und die andere auf Krediten sitzt — ein Käufer übernimmt die Schulden
        mit.
        """
        if self.market_cap is None:
            return None
        return self.market_cap + (self.net_debt or 0.0)

    @property
    def ev_to_ebitda(self) -> float | None:
        """Robuster als das KGV: unabhängig von Kapitalstruktur und Abschreibungen."""
        ev, e = self.enterprise_value, self.ebitda
        if ev is None or e is None or e <= 0 or ev <= 0:
            return None
        return ev / e

    @property
    def fcf_yield(self) -> float | None:
        """Freier Cashflow je Euro Börsenwert — schwerer zu schönen als der Gewinn."""
        if self.free_cashflow is None or not self.market_cap:
            return None
        return self.free_cashflow / self.market_cap


@dataclass
class PriceHistory:
    """Tägliche OHLCV-Daten. `frame` hat mindestens die Spalten Close und Volume."""

    ticker: str
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def close(self) -> pd.Series:
        return self.frame["Close"].dropna()

    @property
    def volume(self) -> pd.Series:
        return self.frame["Volume"].dropna()

    @property
    def last_close(self) -> float | None:
        c = self.close
        return float(c.iloc[-1]) if len(c) else None


@dataclass
class ValuationResult:
    """Ergebnis der KGV-Analyse. `score` ist 0-100, höher = günstiger."""

    score: float
    trailing_pe: float | None = None
    forward_pe: float | None = None
    pe_percentile_own_history: float | None = None
    pe_vs_own_median: float | None = None
    pe_vs_sector_median: float | None = None
    sector_median_pe: float | None = None
    peg: float | None = None
    earnings_yield: float | None = None
    excess_yield_vs_bond: float | None = None

    #: Weitere Bewertungsmaße, jeweils absolut und relativ zur Branche.
    ev_to_ebitda: float | None = None
    ev_ebitda_vs_sector: float | None = None
    fcf_yield: float | None = None
    fcf_yield_vs_sector: float | None = None
    price_to_book: float | None = None
    pb_vs_sector: float | None = None

    #: Wie viele der unabhängigen Maße halten den Titel für günstig?
    cheap_measures: int = 0
    comparable_measures: int = 0

    notes: list[str] = field(default_factory=list)

    @property
    def breadth(self) -> float | None:
        """Anteil der Maße, die "günstig" sagen. Das eigentliche Gütesiegel.

        Ein Titel, der nur beim KGV billig aussieht, hat meist Einmaleffekte im
        Gewinn. Erst wenn KGV, EV/EBITDA, KBV und Cashflow-Rendite in dieselbe
        Richtung zeigen, ist die Bewertung wirklich niedrig.
        """
        if self.comparable_measures == 0:
            return None
        return self.cheap_measures / self.comparable_measures


@dataclass
class TechnicalResult:
    """Ergebnis der Chart-Analyse. `score` ist 0-100, höher = besseres Setup."""

    score: float
    price: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    above_sma200: bool | None = None
    golden_cross: bool | None = None
    rsi14: float | None = None
    macd_histogram: float | None = None
    drawdown_from_52w_high: float | None = None
    upside_to_52w_high: float | None = None
    distance_to_52w_low: float | None = None
    annualized_volatility: float | None = None
    atr14: float | None = None
    atr_percent: float | None = None
    bollinger_percent_b: float | None = None
    bollinger_bandwidth: float | None = None
    stochastic_k: float | None = None
    stochastic_d: float | None = None
    volume_trend: float | None = None
    relative_strength_6m: float | None = None
    setup: str = "unklar"
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityResult:
    """Bilanz- und Trendqualität — der Value-Fallen-Filter. `score` 0-100."""

    score: float
    net_debt_to_ebitda: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    return_on_equity: float | None = None
    profit_margin: float | None = None
    payout_ratio: float | None = None
    free_cashflow_positive: bool | None = None
    red_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DataFlag:
    """Ein Befund der Plausibilitätsprüfung.

    `severe` bedeutet: Die Kennzahlen widersprechen sich so deutlich, dass die
    darauf aufbauende Bewertung nicht belastbar ist.
    """

    check: str
    message: str
    severe: bool = False


@dataclass
class DataQualityResult:
    flags: list[DataFlag] = field(default_factory=list)

    @property
    def severe_flags(self) -> list[DataFlag]:
        return [f for f in self.flags if f.severe]

    @property
    def trustworthy(self) -> bool:
        return not self.severe_flags

    @property
    def penalty(self) -> float:
        """Faktor auf den Gesamtscore. 1.0 = keine Abwertung."""
        if self.severe_flags:
            return 0.6
        if self.flags:
            return 0.9
        return 1.0


@dataclass
class MacroSeries:
    """Eine Makro-Zeitreihe mit aktuellem Wert und Veränderung."""

    key: str
    label: str
    value: float | None
    change_3m: float | None = None
    change_12m: float | None = None
    unit: str = ""
    as_of: date | None = None


@dataclass
class MacroRegime:
    """Verdichtetes Makrobild plus Sektor-Einschätzungen."""

    series: dict[str, MacroSeries] = field(default_factory=dict)
    rate_direction: str = "neutral"  # steigend | fallend | neutral
    curve_shape: str = "neutral"  # invers | flach | steil
    inflation_trend: str = "neutral"  # steigend | fallend | neutral
    growth_signal: str = "neutral"  # expansiv | restriktiv | neutral
    risk_appetite: str = "neutral"  # risk-on | risk-off | neutral
    #: Sektor -> Score 0-100 (50 = neutral)
    sector_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    live: bool = False

    def score_for(self, sector: str | None) -> float:
        if not sector:
            return 50.0
        return self.sector_scores.get(sector, 50.0)


@dataclass
class NewsItem:
    title: str
    publisher: str | None = None
    published: date | None = None
    url: str | None = None
    summary: str | None = None


@dataclass
class LLMContext:
    """Vom LLM erzeugte Einordnung eines Kandidaten."""

    cheap_because: str = ""
    macro_alignment: str = ""
    key_risks: list[str] = field(default_factory=list)
    verdict: str = "unklar"  # zyklisch-guenstig | strukturell-billig | unklar
    confidence: str = "mittel"  # hoch | mittel | niedrig
    summary: str = ""
    error: str | None = None


@dataclass
class Candidate:
    """Ein bewerteter Titel mit allem, was der Report braucht."""

    ticker: str
    fundamentals: Fundamentals
    valuation: ValuationResult
    technical: TechnicalResult
    quality: QualityResult
    macro_score: float = 50.0
    total_score: float = 0.0
    data_quality: DataQualityResult = field(default_factory=DataQualityResult)
    #: Hebelrechnung auf den Basiswert (analysis.leverage.LeverageAssessment).
    leverage: Any | None = None
    news: list[NewsItem] = field(default_factory=list)
    llm: LLMContext | None = None

    @property
    def name(self) -> str:
        return self.fundamentals.name or self.ticker

    @property
    def sector(self) -> str:
        return self.fundamentals.sector or "Unbekannt"

    def to_dict(self) -> dict[str, Any]:
        """Flache Darstellung für JSON-Export und LLM-Prompt."""
        f, v, t, q = self.fundamentals, self.valuation, self.technical, self.quality
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "country": f.country,
            "currency": f.currency,
            "market_cap": f.market_cap,
            "total_score": round(self.total_score, 1),
            "valuation": {
                "score": round(v.score, 1),
                "trailing_pe": v.trailing_pe,
                "forward_pe": v.forward_pe,
                "pe_percentile_own_history": v.pe_percentile_own_history,
                "pe_vs_own_median": v.pe_vs_own_median,
                "pe_vs_sector_median": v.pe_vs_sector_median,
                "sector_median_pe": v.sector_median_pe,
                "peg": v.peg,
                "earnings_yield": v.earnings_yield,
                "excess_yield_vs_bond": v.excess_yield_vs_bond,
                "ev_to_ebitda": v.ev_to_ebitda,
                "ev_ebitda_vs_sector": v.ev_ebitda_vs_sector,
                "fcf_yield": v.fcf_yield,
                "fcf_yield_vs_sector": v.fcf_yield_vs_sector,
                "price_to_book": v.price_to_book,
                "pb_vs_sector": v.pb_vs_sector,
                "cheap_measures": v.cheap_measures,
                "comparable_measures": v.comparable_measures,
                "notes": v.notes,
            },
            "technical": {
                "score": round(t.score, 1),
                "price": t.price,
                "rsi14": t.rsi14,
                "above_sma200": t.above_sma200,
                "golden_cross": t.golden_cross,
                "drawdown_from_52w_high": t.drawdown_from_52w_high,
                "distance_to_52w_low": t.distance_to_52w_low,
                "annualized_volatility": t.annualized_volatility,
                "relative_strength_6m": t.relative_strength_6m,
                "setup": t.setup,
                "notes": t.notes,
            },
            "quality": {
                "score": round(q.score, 1),
                "net_debt_to_ebitda": q.net_debt_to_ebitda,
                "revenue_growth": q.revenue_growth,
                "earnings_growth": q.earnings_growth,
                "return_on_equity": q.return_on_equity,
                "profit_margin": q.profit_margin,
                "free_cashflow_positive": q.free_cashflow_positive,
                "red_flags": q.red_flags,
            },
            "macro_score": round(self.macro_score, 1),
            "data_quality": {
                "trustworthy": self.data_quality.trustworthy,
                "flags": [
                    {"check": f.check, "message": f.message, "severe": f.severe}
                    for f in self.data_quality.flags
                ],
            },
            "llm": None
            if self.llm is None
            else {
                "cheap_because": self.llm.cheap_because,
                "macro_alignment": self.llm.macro_alignment,
                "key_risks": self.llm.key_risks,
                "verdict": self.llm.verdict,
                "confidence": self.llm.confidence,
                "summary": self.llm.summary,
            },
        }
